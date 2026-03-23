FROM paddlepaddle/paddle:2.5.0rc0-cpu

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

# Upgrade pip first (important for Paddle)
RUN pip install --no-cache-dir --upgrade pip

# Install PaddlePaddle (CPU)
RUN pip install --no-cache-dir paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

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
