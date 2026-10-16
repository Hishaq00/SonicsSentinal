# SonicSentinel AI container image
#   docker build -t sonicsentinel .
#   docker run -p 8000:8000 -e SECRET_KEY=change-me -v sonic-data:/data sonicsentinel
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////data/sonicsentinel.db STORAGE_DIR=/data/storage PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data/storage
VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} wsgi:app"]
