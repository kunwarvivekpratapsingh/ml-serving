#!/bin/bash
set -e

echo "========================================="
echo "  ML Serving Project — EC2 Deploy Script"
echo "========================================="

# --- Step 1: Install Docker ---
if ! command -v docker &> /dev/null; then
    echo "[1/4] Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io docker-compose-v2 > /dev/null 2>&1
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "  Docker installed."
    echo "  NOTE: Using 'sudo docker compose' for this session."
    echo "  Log out and back in to use docker without sudo."
else
    echo "[1/4] Docker already installed."
fi

# --- Step 2: Set up swap ---
if [ ! -f /swapfile ]; then
    echo "[2/4] Setting up 2GB swap file..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile > /dev/null
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
    echo "  Swap enabled."
else
    echo "[2/4] Swap already configured."
fi

# --- Step 3: Create data directories ---
echo "[3/4] Creating data directories..."
sudo mkdir -p /data/models /data/logs
sudo chown -R "$USER:$USER" /data

# --- Step 4: Build and start ---
echo "[4/4] Building and starting containers..."
echo "  This will take 5-10 minutes on t3.micro (first build)."
echo ""

sudo docker compose up -d --build

echo ""
echo "========================================="
echo "  Deployment complete!"
echo "========================================="
echo ""

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_EC2_IP")

echo "  Model endpoints:"
echo "    Iris classifier:    POST http://${PUBLIC_IP}/model-a/predict"
echo "    House price:        POST http://${PUBLIC_IP}/model-b/predict"
echo "    Sentiment:          POST http://${PUBLIC_IP}/model-c/predict"
echo ""
echo "  Health checks:"
echo "    http://${PUBLIC_IP}/model-a/health"
echo "    http://${PUBLIC_IP}/model-b/health"
echo "    http://${PUBLIC_IP}/model-c/health"
echo ""
echo "  Monitoring:"
echo "    Grafana dashboard:  http://${PUBLIC_IP}:3000  (admin / mlserving)"
echo "    Prometheus:         http://${PUBLIC_IP}:9090"
echo ""
echo "  Test commands:"
echo "    curl -s -X POST http://localhost/model-a/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"features\": [5.1, 3.5, 1.4, 0.2]}'"
echo ""
echo "    curl -s -X POST http://localhost/model-b/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"features\": [3.5, 25.0, 5.2, 1.0, 1200, 3.0, 37.5, -122.0]}'"
echo ""
echo "    curl -s -X POST http://localhost/model-c/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"text\": \"This product is absolutely wonderful\"}'"
echo ""
echo "  Management:"
echo "    sudo docker compose ps              — check running containers"
echo "    sudo docker compose logs -f         — tail all logs"
echo "    sudo docker compose logs model-a    — logs for specific model"
echo "    sudo docker compose stop grafana    — free 100MB RAM"
echo "    sudo docker compose restart model-a — restart one model"
echo "    sudo docker compose down            — stop everything"
