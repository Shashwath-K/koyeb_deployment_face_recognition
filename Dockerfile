# Multi-stage build for PyTorch + Flask application
# Stage 1: Builder (install dependencies)
FROM python:3.10-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with PyTorch from CPU-only index
# Using --user flag to install in user directory for easier copying
RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PostgreSQL client libraries (for psycopg2)
    libpq-dev \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Add user packages to PATH and PYTHONPATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Copy application code
COPY . .

# Clean up Python bytecode and cache files
RUN find /root/.local -name "*.pyc" -delete && \
    find /root/.local -type d -name "__pycache__" -delete

# Verify PyTorch installation (CPU-only)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'PyTorch built with CUDA: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"None\"}')"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /root/.local
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/api/health', timeout=5); exit(0) if r.status_code == 200 else exit(1)"

# Run with gunicorn (optimized for CPU/memory)
CMD ["gunicorn", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "2", \
    "--threads", "2", \
    "--timeout", "300", \
    "--worker-class", "sync", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:app"]