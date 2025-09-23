# Dockerfile optimizat pentru AI Educational PDF Embeddings
FROM python:3.9-slim

# Setează directorul de lucru
WORKDIR /app

# Instalează dependențele de sistem necesare
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiază requirements și instalează dependențele Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiază codul aplicației
COPY pdf_converter_working.py .
COPY railway_api.py .
COPY test_pdf_working.py .

# Creează directoare necesare
RUN mkdir -p /app/embeddings_db
RUN mkdir -p /app/uploads
RUN mkdir -p /app/new_pdfs
RUN mkdir -p /app/processed_pdfs

# Copiază embeddings-urile dacă există (opțional - pentru pre-populated)
# COPY embeddings_db/ ./embeddings_db/

# Setează variabilele de mediu
ENV PYTHONPATH=/app
ENV EMBEDDINGS_PATH=/app/embeddings_db
ENV PORT=8000

# Expune portul
EXPOSE 8000

# Healthcheck pentru Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Rulează aplicația
CMD ["python", "railway_api.py"]