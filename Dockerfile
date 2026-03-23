# CPU image ships PaddlePaddle 3.2.x with a supported Python (cp310+). Older
# paddle:2.5.x bases use Python 3.8, which has no paddlepaddle 3.2 wheels on PyPI.
FROM paddlepaddle/paddle:3.2.0

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# System dependencies (fixed + updated)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "paddlex[ocr]"

# Copy app
COPY api.py .

# Volume (optional for Coolify persistence)
VOLUME ["/data"]

# Expose port
EXPOSE 8000

# Start app
CMD ["python", "api.py"]
