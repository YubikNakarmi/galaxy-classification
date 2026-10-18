#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/mlflow-setup.log) 2>&1

MLFLOW_USER="bishwa"
MLFLOW_PASS="CHANGE_ME"
APP_DIR="/opt/mlflow"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl openssl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

PUBLIC_IP=$(curl -s -H Metadata:true --max-time 5 \
  "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/publicIpAddress?api-version=2021-02-01&format=text" || true)
if [ -z "$PUBLIC_IP" ]; then
  PUBLIC_IP=$(curl -s --max-time 5 https://ifconfig.me || true)
fi
echo "Public IP: $PUBLIC_IP"

mkdir -p "$APP_DIR/nginx"
cd "$APP_DIR"

printf "%s:%s\n" "$MLFLOW_USER" "$(openssl passwd -apr1 "$MLFLOW_PASS")" > nginx/.htpasswd

# nginx config
cat > nginx/nginx.conf <<'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 0;

    auth_basic           "MLflow";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://mlflow:5000;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF

# docker compose file
cat > docker-compose.yml <<EOF
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.13.0-full
    container_name: mlflow
    expose:
      - "5000"
    volumes:
      - mlflow_data:/mlflow
    command: >
      mlflow server
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --default-artifact-root mlflow-artifacts:/
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
      --host 0.0.0.0
      --port 5000
      --allowed-hosts "*"
      --cors-allowed-origins "http://$PUBLIC_IP:5000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  nginx:
    image: nginx:stable
    container_name: mlflow-nginx
    ports:
      - "5000:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/.htpasswd:/etc/nginx/.htpasswd:ro
    depends_on:
      mlflow:
        condition: service_healthy
    restart: unless-stopped

volumes:
  mlflow_data:
EOF

docker compose up -d
echo "MLflow is up at http://$PUBLIC_IP:5000"