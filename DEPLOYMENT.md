# HatCat Deployment Guide

This guide provides instructions for deploying HatCat and its OpenWebUI fork (hatcat-ui) as a complete deception detection system.

## Overview

The deployment consists of three main components:

1. **HatCat Backend** - FastAPI server providing divergence analysis via OpenAI-compatible API
2. **OpenWebUI Fork (hatcat-ui)** - Web interface with real-time divergence visualization
3. **Model Artifacts** - Trained SUMO classifiers (~3 GB) and concept layer definitions (~74 MB)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              User's Browser                         │
└─────────────────────┬───────────────────────────────┘
                      │
                      │ HTTP (port 5173 dev / 3000 prod)
                      │
┌─────────────────────▼───────────────────────────────┐
│         hatcat-ui Frontend (SvelteKit)              │
│   - Real-time divergence visualization             │
│   - Token-level concept highlighting               │
│   - Interactive concept graph                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      │ HTTP (port 8080)
                      │ API requests
                      │
┌─────────────────────▼───────────────────────────────┐
│         hatcat-ui Backend (OpenWebUI Fork)          │
│   - Chat history and user management               │
│   - Proxy to HatCat backend                        │
│   - Session management                             │
└─────────────────────┬───────────────────────────────┘
                      │
                      │ HTTP (port 8765)
                      │ OpenAI-compatible API
                      │
┌─────────────────────▼───────────────────────────────┐
│              HatCat Backend                         │
│   - Gemma-3-4B model (GPU)                         │
│   - SUMO activation lenses                         │
│   - Embedding centroids                            │
│   - Dynamic divergence analysis                    │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 24 GB VRAM (for Gemma-3-4B with full divergence detection)
  - Recommended: RTX 4090, RTX A5000, A6000, or better
  - Minimum: RTX 3090 (24GB) or RTX 4060 Ti 16GB with aggressive optimization
  - **Reality check**: Full pipeline uses 18-22 GB VRAM (model + activations + lenses + generation)
- **RAM**: 32+ GB system memory (16 GB minimum)
- **Storage**: 40+ GB (15 GB for model, 3 GB for classifiers, 15+ GB for Docker images, 7+ GB for concept layers)
- **CPU**: 8+ cores recommended (4 cores minimum)

### Software Requirements

- Docker 24.0+ and Docker Compose 2.20+
- NVIDIA Container Toolkit (for GPU support)
- Git

## Quick Start (Development)

For local development without Docker, use the included startup script:

```bash
# Clone repositories (if not already cloned)
git clone https://github.com/yourusername/HatCat.git
git clone https://github.com/yourusername/hatcat-ui.git

# Ensure hatcat-ui is either:
# - In the same parent directory as HatCat (../hatcat-ui)
# - Inside the HatCat directory (./hatcat-ui)

# Run the startup script
cd HatCat
./start_hatcat_ui.sh
```

The script will:
1. Clean up any existing processes on ports 8765, 8080, 5173
2. Start HatCat Backend on port 8765
3. Start OpenWebUI Backend on port 8080
4. Start OpenWebUI Frontend on port 5173
5. Provide connection instructions

Access the UI at http://localhost:5173

## Production Deployment (Docker Compose)

### 1. Clone Repositories

```bash
# Clone HatCat backend
git clone https://github.com/yourusername/HatCat.git
cd HatCat

# Clone hatcat-ui fork
cd ..
git clone https://github.com/yourusername/hatcat-ui.git
```

### 2. Create Docker Compose Configuration

Create `docker-compose.yml` in a parent directory containing both repos:

```yaml
version: '3.8'

services:
  hatcat-backend:
    build:
      context: ./HatCat
      dockerfile: Dockerfile
    container_name: hatcat-backend
    ports:
      - "8765:8765"
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_HUB_OFFLINE=0  # Set to 1 for offline deployments
    volumes:
      # Mount trained model artifacts
      - ./HatCat/results:/app/results:ro
      - ./HatCat/data:/app/data:ro
      # Cache HuggingFace models to avoid re-downloading
      - huggingface-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8765/v1/models"]
      interval: 30s
      timeout: 10s
      retries: 3

  hatcat-ui:
    build:
      context: ./hatcat-ui
      dockerfile: Dockerfile
    container_name: hatcat-ui
    ports:
      - "3000:8080"
    environment:
      # Point to HatCat backend
      - OPENAI_API_BASE_URLS=http://hatcat-backend:8765/v1
      - OPENAI_API_KEYS=sk-hatcat-dummy
      # OpenWebUI configuration
      - WEBUI_NAME=HatCat Divergence Monitor
      - ENABLE_SIGNUP=false
      - DEFAULT_MODELS=hatcat-divergence
    volumes:
      - openwebui-data:/app/backend/data
    depends_on:
      hatcat-backend:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  huggingface-cache:
  openwebui-data:
```

### 3. Launch Stack

```bash
docker-compose up -d
```

Access the UI at http://localhost:3000

## Manual Deployment

### HatCat Backend

#### 1. Create Dockerfile

Create `HatCat/Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip3 install poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev dependencies for production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY results/ ./results/

# Expose API port
EXPOSE 8765

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8765/v1/models || exit 1

# Run FastAPI server
CMD ["uvicorn", "src.ui.openwebui.server:app", "--host", "0.0.0.0", "--port", "8765"]
```

#### 2. Build and Run

```bash
cd HatCat

# Build image
docker build -t hatcat-backend:latest .

# Run container
docker run -d \
  --name hatcat-backend \
  --gpus all \
  -p 8765:8765 \
  -v $(pwd)/results:/app/results:ro \
  -v $(pwd)/data:/app/data:ro \
  -v hatcat-hf-cache:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  hatcat-backend:latest
```

### OpenWebUI Fork (hatcat-ui)

The hatcat-ui repository already contains a Dockerfile. Build and run:

```bash
cd hatcat-ui

# Build image
docker build -t hatcat-ui:latest .

# Run container
docker run -d \
  --name hatcat-ui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URLS=http://hatcat-backend:8765/v1 \
  -e OPENAI_API_KEYS=sk-hatcat-dummy \
  -e WEBUI_NAME="HatCat Divergence Monitor" \
  -e ENABLE_SIGNUP=false \
  --link hatcat-backend:hatcat-backend \
  -v openwebui-data:/app/backend/data \
  hatcat-ui:latest
```

## Model Artifacts

### Packaging Trained Classifiers

The trained model artifacts need to be included in the deployment:

```bash
# From HatCat directory
tar -czf hatcat-artifacts-v1.tar.gz \
  results/sumo_classifiers/ \
  data/concept_graph/abstraction_layers/
```

Size breakdown:
- `results/sumo_classifiers/`: ~3.0 GB (5,582 concept classifiers + centroids)
- `data/concept_graph/abstraction_layers/`: ~74 MB (SUMO-WordNet hierarchy)

### Artifact Storage Options

**Option 1: Include in Docker Image**
- Pros: Single deployable artifact
- Cons: Large image size (~18 GB)

**Option 2: External Volume Mount** (Recommended)
- Pros: Smaller base image, easier updates
- Cons: Requires separate artifact distribution

**Option 3: Download on First Run**
- Pros: Minimal image size
- Cons: Slow first startup, requires external hosting

### Distribution via Cloud Storage

For production deployments, host artifacts externally:

```bash
# Example: Upload to S3
aws s3 cp hatcat-artifacts-v1.tar.gz s3://your-bucket/hatcat/

# In Dockerfile or entrypoint script:
# wget https://your-bucket.s3.amazonaws.com/hatcat/hatcat-artifacts-v1.tar.gz
# tar -xzf hatcat-artifacts-v1.tar.gz -C /app/
```

## Environment Variables

### HatCat Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_CUDA_ALLOC_CONF` | - | Set to `expandable_segments:True` for dynamic VRAM |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to prevent model downloads |
| `HATCAT_BASE_LAYERS` | `0` | Comma-separated layer IDs to preload |
| `HATCAT_KEEP_TOP_K` | `100` | Number of concepts to track per layer |
| `HATCAT_MODEL_NAME` | `google/gemma-3-4b-pt` | HuggingFace model ID |
| `HATCAT_PORT` | `8765` | API server port |

### hatcat-ui

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_BASE_URLS` | - | HatCat backend URL (e.g., `http://hatcat-backend:8765/v1`) |
| `OPENAI_API_KEYS` | - | Dummy API key (required but not validated) |
| `WEBUI_NAME` | `Open WebUI` | Custom branding name |
| `ENABLE_SIGNUP` | `true` | Set to `false` for closed deployments |
| `DEFAULT_MODELS` | - | Pre-select `hatcat-divergence` model |

## Cloud Deployment

### AWS EC2 with GPU

**Instance Type**: `g5.2xlarge` (1x A10G GPU, 24GB VRAM, 8 vCPUs, 32GB RAM)

**Note**: `g4dn.xlarge` (T4 16GB) is **insufficient** for full HatCat pipeline (needs 18-22 GB VRAM)

```bash
# Launch Ubuntu 22.04 Deep Learning AMI
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Clone repos and launch
git clone https://github.com/yourusername/HatCat.git
git clone https://github.com/yourusername/hatcat-ui.git
docker-compose up -d

# Configure security group to allow:
# - Port 3000 (HTTPS recommended in production)
# - SSH (port 22) for management
```

**Estimated Cost**: ~$1.21/hour (~$870/month for continuous deployment)

**Budget alternative**: Use spot instances at ~$0.40/hour (~$290/month, 67% savings)

### Google Cloud Platform (GCP)

**Instance Type**: `n1-standard-8` with 1x A100 40GB or 1x L4 24GB

```bash
# Create instance with Container-Optimized OS + GPU
gcloud compute instances create hatcat-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# SSH and install NVIDIA drivers
gcloud compute ssh hatcat-vm
sudo cos-extensions install gpu
sudo mount --bind /var/lib/nvidia /var/lib/nvidia
sudo mount -o remount,exec /var/lib/nvidia

# Deploy with Docker Compose
# ... (same as AWS)
```

**Estimated Cost**:
- L4 (24GB): ~$0.60/hour (~$430/month)
- A100 (40GB): ~$2.50/hour (~$1,800/month)

**Budget alternative**: Use preemptible L4 at ~$0.20/hour (~$145/month, 67% savings)

### Azure VM

**Instance Type**: `NC24ads_A100_v4` (1x A100 80GB, 24 vCPUs, 220GB RAM) or `NC6s_v3` (1x V100 16GB - insufficient)

```bash
# Create resource group and VM
az group create --name hatcat-rg --location eastus
az vm create \
  --resource-group hatcat-rg \
  --name hatcat-vm \
  --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install NVIDIA drivers and Docker
# ... (similar to AWS setup)
```

**Estimated Cost**: ~$3.67/hour (~$2,640/month for A100)

**Budget alternative**: Azure doesn't have good mid-range GPU options. Consider AWS/GCP instead.

### Self-Hosted / On-Premises

Requirements:
- Linux server with NVIDIA GPU
- Docker + NVIDIA Container Toolkit
- Static IP or dynamic DNS for remote access
- Reverse proxy (nginx/Caddy) for HTTPS

```nginx
# Example nginx config for HTTPS
server {
    listen 443 ssl http2;
    server_name hatcat.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/hatcat.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hatcat.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Production Considerations

### 1. Security

- **API Authentication**: Implement proper JWT-based auth in hatcat-ui
- **HTTPS**: Use reverse proxy with TLS certificates (Let's Encrypt)
- **Network Isolation**: Run services in private network, expose only UI via reverse proxy
- **Secrets Management**: Use Docker secrets or environment files (not committed to git)

### 2. Monitoring

```yaml
# Add Prometheus + Grafana to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=changeme
```

Monitor:
- GPU utilization (`nvidia-smi`)
- API latency and throughput
- Memory usage (watch for VRAM leaks)
- Divergence detection statistics

### 3. Scaling

**Horizontal Scaling** (Multiple Replicas):
- Load balancer (nginx/HAProxy) distributing to multiple HatCat backend instances
- Each instance requires dedicated GPU
- Shared volume for model artifacts (NFS/EFS)

**Vertical Scaling** (Larger GPU):
- Upgrade to A10G/A100 for faster inference
- Reduce batch processing time for real-time analysis

### 4. Backup and Recovery

```bash
# Backup OpenWebUI data (chat history, users)
docker run --rm \
  -v openwebui-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/openwebui-backup-$(date +%Y%m%d).tar.gz /data

# Restore
docker run --rm \
  -v openwebui-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar xzf /backup/openwebui-backup-20250111.tar.gz -C /
```

## Testing the Deployment

### 1. Health Checks

```bash
# Backend health (check if models endpoint responds)
curl http://localhost:8765/v1/models

# Frontend health
curl http://localhost:5173/health  # Development
curl http://localhost:3000/health  # Production Docker
```

### 2. API Test

```bash
# Test divergence analysis API
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hatcat-divergence",
    "messages": [{"role": "user", "content": "Tell me about AI safety"}],
    "stream": false,
    "max_tokens": 100
  }'
```

### 3. End-to-End Test

**Development:**
1. Open browser to http://localhost:5173
2. Sign in (first user becomes admin)
3. Go to Admin Settings → Connections
4. Add OpenAI Connection:
   - Base URL: `http://localhost:8765/v1`
   - API Key: `sk-test` (any value works)
5. Select the HatCat model from the model dropdown
6. Send message: "Explain how you work"
7. Verify tokens are color-coded by divergence
8. Check concept panel shows detected concepts

**Production Docker:**
1. Open browser to http://localhost:3000
2. Follow steps 2-8 above (use same backend URL if using docker-compose)

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce model precision or batch size

```python
# In src/openwebui/server.py, change:
torch_dtype=torch.bfloat16  # Already using bfloat16
# Or reduce keep_top_k to process fewer concepts
```

### Issue: Slow First Request

**Cause**: Model loading + CUDA initialization

**Solution**: Implement proper startup with preloading

```python
# Add startup event in server.py
@app.on_event("startup")
async def startup():
    await analyzer.initialize()
    # Warm up with dummy request
    await analyzer.analyze("test")
```

### Issue: Connection Refused Between Containers

**Solution**: Use Docker network instead of `--link`

```bash
docker network create hatcat-net
docker run --network hatcat-net --name hatcat-backend ...
docker run --network hatcat-net --name hatcat-ui \
  -e OPENAI_API_BASE_URLS=http://hatcat-backend:8765/v1 ...
```

### Issue: Model Not Appearing in UI

**Cause**: Connection between OpenWebUI and HatCat backend not configured

**Solution**: Manually add the connection in OpenWebUI:
1. Navigate to Admin Settings → Connections
2. Add OpenAI Connection with:
   - Base URL: `http://localhost:8765/v1` (or appropriate backend URL)
   - API Key: any value (e.g., `sk-test`)
3. Refresh model list

## Updating the Deployment

### Update HatCat Backend Code

```bash
cd HatCat
git pull
docker-compose build hatcat-backend
docker-compose up -d hatcat-backend
```

### Update Model Artifacts

```bash
# Re-train classifiers for new layers
./.venv/bin/python scripts/train_sumo_classifiers.py --layers 6 --train-text-lenses

# Restart backend to load new models
docker-compose restart hatcat-backend
```

### Update hatcat-ui

```bash
cd hatcat-ui
git pull
docker-compose build hatcat-ui
docker-compose up -d hatcat-ui
```

## Cost Optimization

### 1. Use Spot/Preemptible Instances

- **AWS**: Spot instances (~70% discount)
- **GCP**: Preemptible VMs (~80% discount)
- **Azure**: Spot VMs (~90% discount)

**Caveat**: May be terminated with 30-second notice

### 2. Auto-Scaling Schedule

Stop instances during low-usage hours:

```bash
# Cron job to stop at midnight, start at 8am
0 0 * * * docker-compose down
0 8 * * * docker-compose up -d
```

### 3. Smaller Model for Testing

For development/demo deployments, use smaller model:

```yaml
environment:
  - HATCAT_MODEL_NAME=google/gemma-2b  # Smaller, less VRAM
```

## License and Attribution

- **HatCat**: [Your License]
- **OpenWebUI**: Open WebUI License (see hatcat-ui/LICENSE)
- **Gemma Models**: Gemma Terms of Use

## Support and Contributing

- **Issues**: https://github.com/yourusername/HatCat/issues
- **Discussions**: https://github.com/yourusername/HatCat/discussions
- **Documentation**: https://hatcat.readthedocs.io (if applicable)

## Next Steps

After successful deployment:

1. **Training Layer 6**: Complete training for finest-grained concepts
2. **Benchmark Suite**: Establish baseline divergence patterns
3. **Fine-tuning**: Adjust detection thresholds based on use case
4. **Visualization Enhancements**: Custom divergence heatmaps and concept graphs
5. **Research Applications**: Publish findings on deception detection patterns
