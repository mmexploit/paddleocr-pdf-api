FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISABLE_MODEL_SOURCE_CHECK=True
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# CPU inference in Docker: ship BLAS/LAPACK runtimes so NumPy/Paddle can use optimized libs
# (PaddleOCR #10147 / Paddle #54482 used OpenBLAS + LAPACK; avoid bogus CUDA LD_LIBRARY_PATH on CPU).
# KMP_* defaults pair with api.py setdefault for MKL-DNN / Intel OpenMP in containers.
ENV KMP_BLOCKTIME=0
ENV KMP_AFFINITY=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 libgomp1 libmagic1 \
    libopenblas0-pthread \
    liblapack3 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir "paddlex[ocr]"

COPY api.py .

VOLUME /data

EXPOSE 8000

CMD ["python3", "api.py"]
