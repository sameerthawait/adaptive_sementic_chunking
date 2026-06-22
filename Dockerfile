# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ollama==0.4.4 --no-deps

# Download NLTK datasets
RUN python -m nltk.downloader punkt punkt_tab

# Copy frontend build output from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy backend source code and script entrypoint
COPY src/ ./src/
COPY main.py .

# Expose port 8000 for the app
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV CHROMA_PERSIST_DIR=/app/data/chromadb

# Healthcheck to monitor API status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
