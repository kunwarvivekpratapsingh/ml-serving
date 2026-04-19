# ML Model Serving on AWS EC2 Free Tier

Deploy multiple dockerized ML models on a single AWS EC2 t3.micro instance with
Nginx reverse proxy, Prometheus monitoring, Grafana dashboard, and prediction logging.

## Architecture

```
Client Request
      │
      ▼
┌─────────────┐
│  Nginx      │  Port 80 — routes /model-a, /model-b, /model-c
└─────┬───────┘
      │
  ┌───┼───────────────┐
  ▼   ▼               ▼
Model A  Model B   Model C     ← Flask + Gunicorn (each in own container)
:8001    :8002     :8003
  │       │         │
  └───┬───┘─────────┘
      ▼
  Prometheus (:9090)            ← Scrapes /metrics every 15s
      │
      ▼
  Grafana (:3000)               ← Pre-built dashboard
```

**7 containers** running simultaneously, ~700 MB total RAM.

## Quick start

### Prerequisites
- AWS account (free tier)
- EC2 instance: t3.micro, Ubuntu 22.04/24.04, 20GB storage
- Security group: ports 22, 80, 3000, 9090 open

### Deploy (one command)

```bash
# Upload project to EC2
scp -i your-key.pem -r ml-serving-project/ ubuntu@YOUR_EC2_IP:~/

# SSH in
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Deploy everything
cd ~/ml-serving-project
chmod +x deploy.sh
./deploy.sh
```

First build takes 5-10 minutes. Subsequent starts are instant.

## API reference

All models follow the same contract:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model-{x}/predict` | POST | Run inference |
| `/model-{x}/health` | GET | Health check |
| `/model-{x}/metrics` | GET | Prometheus metrics |

### Model A — Iris classifier (Dr. Priya Sharma)

```bash
curl -X POST http://YOUR_IP/model-a/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:
```json
{
  "model": "iris_classifier",
  "version": "1.0",
  "prediction": {
    "species": "setosa",
    "class": 0,
    "probabilities": {"setosa": 0.98, "versicolor": 0.01, "virginica": 0.01}
  },
  "latency_ms": 12.5
}
```

### Model B — House price predictor (Rahul Mehta)

```bash
curl -X POST http://YOUR_IP/model-b/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [3.5, 25.0, 5.2, 1.0, 1200, 3.0, 37.5, -122.0]}'
```

Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

### Model C — Sentiment analyzer (Ananya Reddy)

```bash
curl -X POST http://YOUR_IP/model-c/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This product is absolutely wonderful"}'
```

## Monitoring

- **Grafana**: `http://YOUR_IP:3000` (admin / mlserving)
- **Prometheus**: `http://YOUR_IP:9090`

The Grafana dashboard is pre-built with:
- Total request count
- Average latency across all models
- Error rate
- Model deployment registry (scientist, version, type)
- Requests per second (per model)
- P95 latency (per model)
- Container CPU and memory usage

## Container inventory

| # | Container | Image | RAM | Port |
|---|-----------|-------|-----|------|
| 1 | nginx | nginx:alpine | 32 MB | 80 |
| 2 | model-a | python:3.11-slim | 80 MB | 8001 |
| 3 | model-b | python:3.11-slim | 100 MB | 8002 |
| 4 | model-c | python:3.11-slim | 90 MB | 8003 |
| 5 | prometheus | prom/prometheus | 60 MB | 9090 |
| 6 | grafana | grafana/grafana | 100 MB | 3000 |
| 7 | cadvisor | google/cadvisor | 40 MB | 8080 |

Total: ~700 MB (fits in 1 GB with swap safety net)

## Management commands

```bash
# Check all containers
sudo docker compose ps

# View logs
sudo docker compose logs -f              # all containers
sudo docker compose logs model-a         # specific model

# Restart one model (after updating code)
sudo docker compose up -d --build model-a

# Free RAM (stop Grafana when not needed)
sudo docker compose stop grafana
sudo docker compose start grafana

# Stop everything
sudo docker compose down

# Full rebuild from scratch
sudo docker compose down
sudo docker compose up -d --build
```

## Deploying a new model version

1. Update the training script or model code
2. Update `MODEL_X_VERSION` in `.env`
3. Rebuild just that container:
```bash
sudo docker compose up -d --build model-a
```
Other models continue serving without interruption.

## Project structure

```
ml-serving-project/
├── docker-compose.yml          # All 7 services
├── .env                        # Versions, scientist names
├── deploy.sh                   # One-command EC2 setup
├── README.md
├── nginx/
│   └── nginx.conf              # Routes /model-a, /model-b, /model-c
├── shared/                     # Common code (copied into each image)
│   ├── __init__.py
│   ├── metrics.py              # Prometheus counters + Info metric
│   ├── middleware.py            # Request timing, error handling
│   └── logger.py               # SQLite prediction logger
├── model-a/                    # Iris classifier
│   ├── Dockerfile
│   ├── app.py
│   ├── train_model.py
│   └── requirements.txt
├── model-b/                    # House price predictor
│   ├── Dockerfile
│   ├── app.py
│   ├── train_model.py
│   └── requirements.txt
├── model-c/                    # Sentiment analyzer
│   ├── Dockerfile
│   ├── app.py
│   ├── train_model.py
│   └── requirements.txt
└── monitoring/
    ├── prometheus.yml           # Scrape targets
    ├── alerts.yml               # Alert rules
    └── grafana/provisioning/
        ├── datasources/
        │   └── prometheus.yml   # Auto-connects to Prometheus
        └── dashboards/
            ├── dashboards.yml
            └── ml-serving.json  # Pre-built dashboard
```
