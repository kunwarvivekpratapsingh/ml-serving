FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY model-c/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY model-c/train_model.py .
COPY model-c/app.py .

RUN python train_model.py

EXPOSE 8003

CMD ["gunicorn", "--bind", "0.0.0.0:8003", "--workers", "1", "--threads", "2", "--timeout", "30", "app:app"]
