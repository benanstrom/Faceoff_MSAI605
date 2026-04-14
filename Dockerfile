# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install facenet-pytorch torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

# Install the package in editable mode
RUN pip install -e .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default command: show CLI help
CMD ["python", "scripts/infer_pairs.py", "--help"]