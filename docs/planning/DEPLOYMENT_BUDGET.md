# Budget-Friendly HatCat Deployment

Cost-effective deployment strategies for HatCat under $50/month, with options as low as $0 for self-hosting.

## Cost Breakdown Analysis

**Why is cloud GPU expensive?**
- Dedicated GPU instances (T4/A10) cost $350-500/month
- Most of the cost is GPU rental, not compute/storage

**How to reduce costs:**
1. Use consumer GPU hardware (self-host)
2. Share GPU resources (serverless/spot)
3. Deploy to free/low-cost CPU-only instances for demos
4. Optimize model size

---

## Option 1: Self-Hosted (Free - $0/month)

**Best for**: Personal use, research, development

### Requirements
- GPU: NVIDIA RTX 4090 (24GB) or RTX A5000 (24GB) recommended
- Minimum: RTX 4060 Ti 16GB (will need aggressive optimization)
- Can be your gaming PC, workstation, or dedicated server
- Existing hardware you already own

**Memory Reality Check**: Full HatCat divergence detection uses **18-22 GB VRAM**.
If you only have 12-16 GB, you'll need quantization (see Strategy 2 below).

### Setup

```bash
# Install NVIDIA drivers + Docker + NVIDIA Container Toolkit
# (One-time setup)

# Clone and deploy
git clone https://github.com/yourusername/HatCat.git
git clone https://github.com/yourusername/hatcat-ui.git
cd HatCat
docker-compose up -d
```

### Access Options

**Local Network Only** (Most secure, free):
- Access at `http://localhost:3000`
- No internet exposure, zero hosting cost

**Remote Access via Tailscale** (Free for personal use):
```bash
# Install Tailscale on host and client devices
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# Access from anywhere via Tailscale IP
# e.g., http://100.64.0.1:3000
```

**Remote Access via Cloudflare Tunnel** (Free):
```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Create tunnel (requires Cloudflare account)
cloudflared tunnel create hatcat
cloudflared tunnel route dns hatcat hatcat.yourdomain.com

# Run tunnel
cloudflared tunnel run hatcat --url http://localhost:3000
```

**Cost**: $0/month (electricity ~$10-20/month if running 24/7)

**Pros**:
- No recurring cloud fees
- Full control over hardware
- Low latency for local use
- Can turn off when not in use

**Cons**:
- Requires upfront hardware investment (~$400-800 for used RTX 3060)
- Responsible for maintenance
- Home internet bandwidth limits
- Not suitable for public/production deployment

---

## Option 2: Vast.ai / RunPod Serverless ($10-30/month)

**Best for**: Intermittent use, demos, testing

### Vast.ai On-Demand

Pay only when running:
- **RTX 3090** (24GB): ~$0.20-0.35/hour
- **RTX 4090** (24GB): ~$0.40-0.60/hour
- **A4000** (16GB): ~$0.15-0.25/hour

**Example usage**:
- 2 hours/day × 30 days = 60 hours/month
- 60 hours × $0.25/hour = **$15/month**

```bash
# Install Vast.ai CLI
pip install vastai

# Find cheap RTX 3090 instance
vastai search offers 'gpu_name=RTX_3090 num_gpus=1' --order 'dph_total+'

# Launch instance with Docker image
vastai create instance <OFFER_ID> \
  --image nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --disk 50 \
  --ssh

# SSH in and run docker-compose
ssh -p <PORT> root@<IP>
# ... deploy HatCat
```

**Access**: Direct IP access, or tunnel via Cloudflare/ngrok

### RunPod Serverless

Deploy as serverless endpoint (pay per request):
- **Cold start**: 10-30 seconds (first request)
- **Warm requests**: ~2-5 seconds
- **Pricing**: ~$0.00015/second of GPU time

**Example costs**:
- 100 queries/day × 5 seconds/query = 500 seconds/day
- 500 sec/day × 30 days × $0.00015 = **$2.25/month**

Setup requires:
1. Create RunPod account
2. Build custom Docker image with HatCat
3. Deploy as serverless endpoint
4. Call via API from hatcat-ui

**Cost**: $10-30/month for light-moderate use

**Pros**:
- Only pay when using
- No idle costs
- Easy to scale up/down
- Good GPU availability

**Cons**:
- Cold start latency
- Requires SSH management (Vast.ai)
- Instance may be terminated (preemptible)

---

## Option 3: Oracle Cloud Free Tier ($0/month, CPU-only demo)

**Best for**: CPU-only demo, testing UI without GPU

Oracle Cloud offers **ALWAYS FREE** ARM instances:
- 4 OCPU + 24 GB RAM (Ampere A1)
- 200 GB storage
- 10 TB/month bandwidth

### Limitations
- **No GPU** - Use quantized/small model (Gemma-2B or CPU inference)
- Slower inference (~5-10 seconds per response)
- Suitable for demos, not real-time analysis

### Setup

```bash
# Create Oracle Cloud account (free tier)
# Launch ARM instance (4 OCPU, 24GB RAM)

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose

# Deploy with CPU-only model
git clone https://github.com/yourusername/HatCat.git
cd HatCat

# Modify docker-compose.yml to remove GPU requirements
# Change model to smaller variant
# Deploy
docker-compose up -d
```

**Cost**: $0/month (forever, as long as within free tier limits)

**Pros**:
- Completely free
- Reliable uptime
- Good for UI testing and demos

**Cons**:
- Very slow inference without GPU
- May need model quantization
- Limited to small models

---

## Option 4: Google Colab Pro ($10/month)

**Best for**: Experimentation, research sessions, intermittent use

Colab Pro provides:
- T4/V100 GPU access
- ~24 hour session limits
- Background execution

### Setup

Create Colab notebook:

```python
# Install dependencies
!git clone https://github.com/yourusername/HatCat.git
%cd HatCat
!pip install -e .

# Run server in background
import subprocess
proc = subprocess.Popen(['python', 'src/openwebui/server.py'])

# Expose via ngrok
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Access at: {public_url}")
```

**Cost**: $10/month (Colab Pro)

**Pros**:
- Very cheap GPU access
- Easy to share notebooks
- No setup required

**Cons**:
- Not for production
- Session timeouts
- Can't run 24/7
- Requires manual restarts

---

## Option 5: Shared Server / Split Costs ($5-20/month per person)

**Best for**: Research groups, multiple users

Rent a single GPU instance and share costs:
- 4 people × $100/month instance = **$25/person**
- Implement user quotas/rate limiting

```yaml
# Multi-user deployment with quotas
services:
  hatcat-backend:
    # ... (same as main deployment)
    environment:
      - RATE_LIMIT_PER_USER=100req/hour

  hatcat-ui:
    # Enable signup for team members
    environment:
      - ENABLE_SIGNUP=true
      - ENABLE_USER_RATE_LIMITING=true
```

---

## Option 6: Model Optimization (Reduce GPU requirements)

### Strategy 1: Use Smaller Base Model

**Gemma-2B** instead of Gemma-3-4B:
- VRAM: 4-6 GB (vs 20 GB for full detection pipeline)
- Inference: 2-3x faster
- Trade-off: Lower accuracy (~5-10% drop in F1)

**Note**: Gemma-3-4B with full divergence detection typically uses **18-22 GB VRAM** due to:
- Base model: ~8 GB (bfloat16)
- Activation captures: ~4-6 GB
- Lens inference: ~2-4 GB
- Text generation buffers: ~4-6 GB

Re-train classifiers for Gemma-2B:
```bash
./.venv/bin/python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 \
  --model google/gemma-2b \
  --train-text-lenses \
  --use-adaptive
```

**New GPU options** (for Gemma-2B + quantization):
- RTX 3060 Ti (8GB): $0.12/hour on Vast.ai - requires heavy optimization
- RTX 4060 (8GB): $0.15/hour on Vast.ai - marginal, will OOM easily
- RTX 3090 (24GB): $0.25/hour on Vast.ai - recommended minimum for full pipeline

### Strategy 2: 4-bit Quantization

Reduce model size by 4x using bitsandbytes:

```python
# In src/openwebui/server.py
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
```

**Result**:
- Gemma-3-4B: 20 GB → **8-10 GB VRAM** (base model ~3 GB, lenses/activations ~5-7 GB)
- Slight quality loss (~2-3%)
- Can run on 12-16 GB GPUs with careful memory management

### Strategy 3: CPU-Only Inference (Free GPUs)

Use ONNX Runtime or llama.cpp for CPU inference:

```bash
# Convert model to ONNX
pip install optimum
optimum-cli export onnx \
  --model google/gemma-3-4b-pt \
  --task text-generation \
  gemma-3-4b-onnx/

# Use ONNX runtime (CPU)
# Inference: ~10-30 seconds per response
```

**Enables**:
- Oracle Cloud free tier
- Any cheap VPS ($5-10/month)
- Old laptops/desktops

---

## Option 7: Hybrid Deployment ($0-5/month)

**Backend**: Self-hosted (free)
**Frontend**: Vercel/Netlify (free)

### Architecture

```
User → Vercel (hatcat-ui frontend) → Home server (HatCat backend)
                                       ↓
                                   Cloudflare Tunnel (free)
```

### Setup

1. **Backend at home** (free):
   - Run HatCat on home GPU
   - Expose via Cloudflare Tunnel

2. **Frontend on Vercel** (free):
   - Deploy hatcat-ui static frontend
   - Configure backend URL

```bash
# Backend (home)
cloudflared tunnel create hatcat-api
cloudflared tunnel route dns hatcat-api api.hatcat.yourdomain.com
cloudflared tunnel run hatcat-api --url http://localhost:8000

# Frontend (Vercel)
cd hatcat-ui
npm run build
vercel deploy
# Set environment: OPENAI_API_BASE_URLS=https://api.hatcat.yourdomain.com
```

**Cost**: $0/month (domain optional, ~$10/year)

---

## Recommended Budget Tiers

### Tier 0: Free / Self-Host
- **Hardware**: Personal GPU PC (already owned)
- **Access**: Local network or Tailscale
- **Use case**: Personal research, development
- **Cost**: $0/month

### Tier 1: Casual Use ($10-15/month)
- **Platform**: Vast.ai on-demand (RTX 3090)
- **Usage**: 2 hours/day, turn off when idle
- **Use case**: Demos, intermittent testing
- **Cost**: ~$15/month

### Tier 2: Regular Use ($20-30/month)
- **Platform**: RunPod Serverless or Vast.ai reserved
- **Usage**: 100-200 requests/day
- **Use case**: Small team, regular research
- **Cost**: ~$25/month

### Tier 3: Light Production ($30-50/month)
- **Platform**: Vast.ai 24/7 RTX 3090
- **Usage**: Always-on, multiple users
- **Use case**: Small lab deployment
- **Cost**: ~$40-50/month

---

## Cost Comparison Table

| Option | Monthly Cost | GPU | Uptime | Best For |
|--------|-------------|-----|--------|----------|
| Self-hosted | $0 | Personal | Variable | Personal use |
| Vast.ai (2hr/day) | $15 | RTX 3090 | On-demand | Demos |
| RunPod Serverless | $2-25 | A4000 | On-demand | API use |
| Oracle Free Tier | $0 | None (CPU) | 24/7 | UI testing |
| Colab Pro | $10 | T4/V100 | Sessions | Experiments |
| Vast.ai 24/7 | $40-50 | RTX 3090 | 24/7 | Small prod |
| AWS g4dn.xlarge | $380 | T4 | 24/7 | Enterprise |

---

## Optimization Checklist

To minimize costs:

- [ ] Use Gemma-2B instead of Gemma-3-4B (saves 50% VRAM)
- [ ] Apply 4-bit quantization (saves 75% VRAM)
- [ ] Train only layers 0-2 (saves 80% training time)
- [ ] Use on-demand/spot instances (saves 70-90% vs reserved)
- [ ] Turn off instances when not in use
- [ ] Implement auto-shutdown after idle time
- [ ] Use CPU inference for non-critical demos
- [ ] Cache model downloads to avoid re-downloading
- [ ] Share instance costs with team members

---

## Auto-Shutdown Script (Save Money)

Automatically stop instance after idle time:

```python
#!/usr/bin/env python3
"""Auto-shutdown after 30 minutes of no API requests"""
import time
import subprocess
from pathlib import Path

IDLE_TIMEOUT = 1800  # 30 minutes
LAST_REQUEST_FILE = Path("/tmp/hatcat_last_request")

while True:
    time.sleep(60)

    if not LAST_REQUEST_FILE.exists():
        continue

    last_request = float(LAST_REQUEST_FILE.read_text())
    idle_time = time.time() - last_request

    if idle_time > IDLE_TIMEOUT:
        print(f"Idle for {idle_time/60:.1f} minutes, shutting down...")
        subprocess.run(["sudo", "shutdown", "-h", "now"])
        break
```

Add to server.py:
```python
@app.middleware("http")
async def track_requests(request, call_next):
    Path("/tmp/hatcat_last_request").write_text(str(time.time()))
    return await call_next(request)
```

---

## FAQ

**Q: Can I run HatCat on Apple Silicon (M1/M2)?**
A: Yes, but requires MPS backend setup and may be slower. No CUDA support.

**Q: Can I run multiple models on one GPU?**
A: Not practically. Even quantized, HatCat uses 8-10 GB VRAM. You'd need 40+ GB VRAM (A100) for multiple instances.

**Q: What's the minimum GPU for production?**
A: RTX 3090/4090 24GB for full Gemma-3-4B pipeline. RTX 4060 Ti 16GB minimum with quantization. 8-12 GB cards will struggle even with optimization.

**Q: Can I use Google Colab free tier?**
A: Yes, but very limited (session timeouts, slower GPUs). Colab Pro ($10) recommended.

**Q: How much bandwidth does HatCat use?**
A: ~1-5 MB per request (tokens + metadata). 100 requests/day = ~150-500 MB/day.

---

## Next Steps

1. **Start with self-hosting** if you have a GPU
2. **Try Vast.ai on-demand** for cloud testing
3. **Implement quantization** to reduce requirements
4. **Monitor actual usage** and right-size your deployment
5. **Consider shared costs** if deploying for a team

For production deployments, see main [DEPLOYMENT.md](DEPLOYMENT.md).
