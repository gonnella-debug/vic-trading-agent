FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bust cache: 2026-04-04-v2
CMD ["sh", "-c", "uvicorn vic:app --host 0.0.0.0 --port ${PORT:-8000}"]
