# Use Python 3.10 explicitly (3.13 has compatibility issues)
FROM python:3.10-slim

LABEL maintainer="Face Attendance API"
LABEL version="1.0"

WORKDIR /app

# Install system dependencies in one RUN layer to reduce image size
RUN apt-get update && apt-get install -y \
    # Python build dependencies
    build-essential \
    python3-dev \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PostgreSQL client
    libpq-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with specific PyTorch CPU version
RUN pip install --no-cache-dir \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Verify installations
RUN python -c "import torch; print(f'PyTorch {torch.__version__} installed'); \
    print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import cv2; print(f'OpenCV {cv2.__version__} installed')" && \
    python -c "import flask; print(f'Flask {flask.__version__} installed')"

# Copy application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:5000/health', timeout=2); exit(0) if r.status_code == 200 else exit(1)" || exit 1

# Run with gunicorn (optimized for memory)
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "2", \
    "--threads", "2", \
    "--timeout", "300", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:app"]