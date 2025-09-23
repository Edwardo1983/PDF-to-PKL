FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY pdf_converter_working.py .
COPY railway_api.py .

RUN mkdir -p /app/embeddings_db
RUN mkdir -p /app/uploads

ENV PYTHONPATH=/app
ENV EMBEDDINGS_PATH=/app/embeddings_db
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "railway_api.py"]