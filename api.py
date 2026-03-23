import inspect
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _resolve_ocr_cpu_threads() -> int:
    raw = os.environ.get("OCR_CPU_THREADS")
    if raw is not None and raw.strip() != "":
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    n = os.cpu_count()
    return max(1, n if n else 1)


def _apply_native_thread_env(threads: int) -> None:
    # OpenMP / BLAS: must be set before Paddle / NumPy heavy paths load.
    # See PaddleOCR #8784 (mkldnn / threading); Docker image libs + env (#10147, Paddle #54482).
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        os.environ.setdefault(key, str(threads))
    # Paddle CPU math kernels (read before paddle native libs init).
    os.environ.setdefault("FLAGS_cpu_math_num_threads", str(threads))
    # Intel OpenMP / oneDNN in containers: avoid bad pinning; low spin (typical Docker/K8s win).
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_AFFINITY", "disabled")


_OCR_CPU_THREADS = _resolve_ocr_cpu_threads()
_apply_native_thread_env(_OCR_CPU_THREADS)

import magic
import pypdfium2 as pdfium
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from paddleocr import PaddleOCRVL


def _kwargs_for_call(func, kw: dict) -> dict:
    """Pass only arguments the callable accepts (supports **kwargs)."""
    try:
        sig = inspect.signature(func)
        params = sig.parameters.values()
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return dict(kw)
        names = {p.name for p in params} - {"self"}
        return {k: v for k, v in kw.items() if k in names}
    except (TypeError, ValueError):
        return {}


def _paddleocr_vl_perf_kwargs() -> dict:
    # PaddleOCR-VL: enable_mkldnn + cpu_threads + use_queues (PaddleOCR 3.x docs).
    # Classic PaddleOCR CPU speedup: enable_mkldnn=True (github.com/PaddlePaddle/PaddleOCR/issues/8784).
    out = {
        "enable_mkldnn": _env_bool("OCR_ENABLE_MKLDNN", True),
        "cpu_threads": _OCR_CPU_THREADS,
        "use_queues": _env_bool("OCR_USE_QUEUES", True),
    }
    dev = os.environ.get("OCR_DEVICE", "").strip()
    if dev:
        out["device"] = dev
    return out


_BASE_PERF = _paddleocr_vl_perf_kwargs()
PADDLEOCR_VL_INIT_KWARGS = _kwargs_for_call(PaddleOCRVL.__init__, _BASE_PERF)
PADDLEOCR_VL_PREDICT_KWARGS = _kwargs_for_call(PaddleOCRVL.predict, _BASE_PERF)


DB_PATH = os.environ.get("DB_PATH", "/data/ocr.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/data/uploads")
DPI = int(os.environ.get("OCR_DPI", "200"))
API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(request: Request):
    if not API_KEY:
        return
    if request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_pages INTEGER DEFAULT 0,
                processed_pages INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                page_num INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(job_id, page_num)
            );
        """)
        now = time.time()
        stale = db.execute("SELECT id FROM jobs WHERE status = 'processing'").fetchall()
        for row in stale:
            db.execute("DELETE FROM pages WHERE job_id = ?", (row["id"],))
        db.execute(
            "UPDATE jobs SET status = 'queued', processed_pages = 0, updated_at = ? WHERE status = 'processing'",
            (now,),
        )


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()



def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    return clean


def strip_image_tags(text: str) -> str:
    return re.sub(r"!\[.*?\]\(.*?\)\n*", "", text)



class OCRWorker:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._model = None
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()

    def cancel_job(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.add(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled

    def _clear_cancelled(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.discard(job_id)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _load_model(self):
        if self._model is None:
            print(
                f"Loading PaddleOCR-VL (cpu_threads={_OCR_CPU_THREADS}, "
                f"enable_mkldnn={_BASE_PERF.get('enable_mkldnn')}, "
                f"use_queues={_BASE_PERF.get('use_queues')})..."
            )
            self._model = PaddleOCRVL(**PADDLEOCR_VL_INIT_KWARGS)
            print("Model loaded.")
        return self._model

    def _run(self):
        ocr = self._load_model()
        while not self._stop.is_set():
            job = self._pick_next_job()
            if job is None:
                time.sleep(1)
                continue
            self._process_job(ocr, job)

    def _pick_next_job(self):
        with get_db() as db:
            row = db.execute(
                "SELECT id, filename FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                now = time.time()
                db.execute(
                    "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return dict(row)
        return None

    def _process_job(self, ocr, job):
        job_id = job["id"]
        pdf_path = Path(UPLOAD_DIR) / job_id / "input.pdf"

        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            total_pages = len(pdf)

            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET total_pages = ?, updated_at = ? WHERE id = ?",
                    (total_pages, time.time(), job_id),
                )

            scale = DPI / 72
            for page_idx in range(total_pages):
                if self._stop.is_set():
                    return
                if self._is_cancelled(job_id):
                    self._clear_cancelled(job_id)
                    with get_db() as db:
                        db.execute(
                            "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                            (time.time(), job_id),
                        )
                    print(f"[{job_id[:8]}] Job cancelled at page {page_idx + 1}/{total_pages}")
                    return

                page = pdf[page_idx]
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_image.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    result = ocr.predict(input=tmp_path, **PADDLEOCR_VL_PREDICT_KWARGS)

                    markdown_parts = []
                    for res in result:
                        md_data = res._to_markdown(pretty=False)
                        if isinstance(md_data, dict):
                            text = md_data.get("markdown_texts") or md_data.get("markdown") or ""
                            if text:
                                markdown_parts.append(text)

                    if not markdown_parts:
                        markdown_parts = [self._extract_text(result)]

                    page_markdown = "\n\n".join(markdown_parts)
                    page_markdown = strip_html(page_markdown)
                    page_markdown = strip_image_tags(page_markdown)

                    now = time.time()
                    with get_db() as db:
                        db.execute(
                            "INSERT INTO pages (job_id, page_num, markdown, created_at) VALUES (?, ?, ?, ?)",
                            (job_id, page_idx + 1, page_markdown, now),
                        )
                        db.execute(
                            "UPDATE jobs SET processed_pages = ?, updated_at = ? WHERE id = ?",
                            (page_idx + 1, now, job_id),
                        )

                    print(f"[{job_id[:8]}] Page {page_idx + 1}/{total_pages} done")

                finally:
                    os.unlink(tmp_path)

            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', updated_at = ? WHERE id = ?",
                    (time.time(), job_id),
                )
            print(f"[{job_id[:8]}] Job completed ({total_pages} pages)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                    (str(e), time.time(), job_id),
                )

    def _extract_text(self, result):
        texts = []
        for res in result:
            if hasattr(res, "rec_text"):
                texts.append(res.rec_text)
            elif hasattr(res, "text"):
                texts.append(res.text)
            else:
                texts.append(str(res))
        return "\n".join(texts)


worker = OCRWorker()


app = FastAPI(title="PaddleOCR API", version="1.0.0", dependencies=[Depends(verify_api_key)])


@app.on_event("startup")
def startup():
    init_db()
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    worker.start()


@app.on_event("shutdown")
def shutdown():
    worker.stop()


@app.post("/ocr")
async def submit_job(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    mime = magic.from_buffer(content, mime=True)
    if mime != "application/pdf":
        raise HTTPException(400, f"File is not a valid PDF (detected: {mime})")

    job_id = uuid.uuid4().hex
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)

    pdf_path = job_dir / "input.pdf"
    pdf_path.write_bytes(content)

    now = time.time()
    with get_db() as db:
        db.execute(
            "INSERT INTO jobs (id, filename, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)",
            (job_id, file.filename, now, now),
        )

    return {"job_id": job_id, "filename": file.filename, "status": "queued"}


@app.get("/ocr/{job_id}")
def get_job_status(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "error": job["error"],
    }


@app.get("/ocr/{job_id}/pages/{page_num}")
def get_page(job_id: str, page_num: int):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        page = db.execute(
            "SELECT * FROM pages WHERE job_id = ? AND page_num = ?",
            (job_id, page_num),
        ).fetchone()

    if not page:
        if page_num > job["total_pages"] and job["total_pages"] > 0:
            raise HTTPException(404, f"Page {page_num} does not exist (total: {job['total_pages']})")
        raise HTTPException(202, f"Page {page_num} not yet processed")

    return {
        "job_id": job_id,
        "page_num": page["page_num"],
        "markdown": page["markdown"],
    }


@app.get("/ocr/{job_id}/result")
def get_full_result(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        pages = db.execute(
            "SELECT page_num, markdown FROM pages WHERE job_id = ? ORDER BY page_num",
            (job_id,),
        ).fetchall()

    return {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "pages": [{"page_num": p["page_num"], "markdown": p["markdown"]} for p in pages],
    }


@app.post("/ocr/{job_id}/cancel")
def cancel_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("queued", "processing"):
            raise HTTPException(400, f"Job cannot be cancelled (status: {job['status']})")
        if job["status"] == "queued":
            db.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
        else:
            worker.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelling" if job["status"] == "processing" else "cancelled"}


@app.delete("/ocr/{job_id}")
def delete_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        db.execute("DELETE FROM pages WHERE job_id = ?", (job_id,))
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    job_dir = Path(UPLOAD_DIR) / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)

    return {"status": "deleted"}


@app.get("/jobs")
def list_jobs():
    with get_db() as db:
        jobs = db.execute(
            "SELECT id, filename, status, total_pages, processed_pages, created_at FROM jobs ORDER BY created_at DESC"
        ).fetchall()

    return {
        "jobs": [
            {
                "job_id": j["id"],
                "filename": j["filename"],
                "status": j["status"],
                "total_pages": j["total_pages"],
                "processed_pages": j["processed_pages"],
            }
            for j in jobs
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
