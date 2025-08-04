# syntax=docker/dockerfile:1

# -----------------------------------------------------
# Base image with slim Python runtime
# -----------------------------------------------------
FROM python:3.9-slim AS runtime

# Install system-level dependencies required by RDKit and friends
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libboost-all-dev \
       libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Set working directory
# -----------------------------------------------------
WORKDIR /app

# -----------------------------------------------------
# Python dependencies
# -----------------------------------------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# Copy backend source code and core library
#   - backend code ends up at /app/main.py (same layout used by existing backend Dockerfile)
#   - project core Python modules live in /app/src so `import src.<module>` works
# -----------------------------------------------------
COPY website/backend/ ./
COPY src/ ./src

# -----------------------------------------------------
# Expose API port & start server
# -----------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
