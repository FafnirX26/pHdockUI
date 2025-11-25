FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for RDKit and other scientific libraries
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libcairo2 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY website/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code from root
COPY src/ /app/src/

# Copy backend application code
COPY website/backend/ /app/

# Expose port (Railway will override with $PORT)
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
