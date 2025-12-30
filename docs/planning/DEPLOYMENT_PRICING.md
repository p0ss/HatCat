# HatCat Deployment: Real-World Pricing Comparison

**TL;DR**: Cloud GPU hosting is expensive. Best options are self-hosting ($0/mo) or Vast.ai spot instances ($15-50/mo).

## Memory Requirements Reality Check

HatCat with Gemma-3-4B uses **18-22 GB VRAM** during inference:
- Base model (bfloat16): ~8 GB
- Activation captures (hidden states): ~4-6 GB
- Lens inference (2,000+ classifiers): ~2-4 GB
- Generation buffers (KV cache): ~4-6 GB

**This means you need 24 GB GPU minimum.**

---

## Cloud Provider Pricing (24/7 Deployment)

### AWS EC2

| Instance | GPU | VRAM | On-Demand | Spot | Monthly (On-Demand) | Monthly (Spot) |
|----------|-----|------|-----------|------|---------------------|----------------|
| g4dn.xlarge | T4 | 16 GB | $0.526/hr | $0.158/hr | ❌ **Too small** | ❌ **Too small** |
| g5.xlarge | A10G | 24 GB | $1.006/hr | $0.302/hr | $724/mo | $217/mo |
| g5.2xlarge | A10G | 24 GB | $1.212/hr | $0.364/hr | $872/mo | $262/mo |
| p3.2xlarge | V100 | 16 GB | $3.06/hr | $0.918/hr | ❌ **Too small** | ❌ **Too small** |
| p4d.24xlarge | A100 | 40 GB | $32.77/hr | $9.83/hr | $23,594/mo 😱 | $7,078/mo |

**Best AWS option**: g5.xlarge spot at **$217/month**

### Google Cloud Platform (GCP)

| Instance | GPU | VRAM | On-Demand | Preemptible | Monthly (On-Demand) | Monthly (Preemptible) |
|----------|-----|------|-----------|-------------|---------------------|----------------------|
| n1-standard-4 + T4 | T4 | 16 GB | $0.44/hr | $0.132/hr | ❌ **Too small** | ❌ **Too small** |
| n1-standard-4 + L4 | L4 | 24 GB | $0.69/hr | $0.207/hr | $497/mo | $149/mo |
| n1-standard-8 + L4 | L4 | 24 GB | $0.87/hr | $0.261/hr | $626/mo | $188/mo |
| a2-highgpu-1g | A100 | 40 GB | $2.93/hr | $0.879/hr | $2,110/mo | $633/mo |

**Best GCP option**: n1-standard-4 + L4 preemptible at **$149/month**

### Azure

| Instance | GPU | VRAM | Price | Monthly |
|----------|-----|------|-------|---------|
| NC6s_v3 | V100 | 16 GB | $3.06/hr | ❌ **Too small** |
| NCasT4_v3 | T4 | 16 GB | $0.526/hr | ❌ **Too small** |
| NC24ads_A100_v4 | A100 | 80 GB | $3.67/hr | $2,642/mo 😱 |

**Best Azure option**: ❌ **None** - Azure has no affordable 24GB GPU option

---

## Budget Cloud Providers (On-Demand GPU Rental)

### Vast.ai (Community GPU Rental)

| GPU | VRAM | Typical Price | 24/7 Monthly | 2hr/day Monthly |
|-----|------|--------------|--------------|-----------------|
| RTX 3090 | 24 GB | $0.20-0.35/hr | $144-252/mo | $12-21/mo |
| RTX 4090 | 24 GB | $0.40-0.70/hr | $288-504/mo | $24-42/mo |
| RTX A5000 | 24 GB | $0.30-0.50/hr | $216-360/mo | $18-30/mo |
| RTX 6000 Ada | 48 GB | $0.60-1.00/hr | $432-720/mo | $36-60/mo |

**Best Vast.ai option**: RTX 3090 spot at **$0.25/hr**
- **24/7**: ~$180/month
- **8 hours/day**: ~$60/month
- **2 hours/day**: ~$15/month

### RunPod (Serverless/Dedicated)

| GPU | VRAM | Serverless | Dedicated | 24/7 Monthly |
|-----|------|------------|-----------|--------------|
| RTX 3090 | 24 GB | $0.00034/sec | $0.34/hr | $245/mo |
| RTX 4090 | 24 GB | $0.00044/sec | $0.44/hr | $317/mo |
| RTX A6000 | 48 GB | $0.00079/sec | $0.79/hr | $569/mo |

**Best RunPod option**: RTX 3090 serverless
- Light use (100 req/day, 5 sec each): **$5-10/month**
- Heavy use (500 req/day): **$25-50/month**

### Lambda Labs (Dedicated Instances)

| GPU | VRAM | Price | Monthly |
|-----|------|-------|---------|
| RTX 6000 Ada | 48 GB | $0.80/hr | $576/mo |
| A10 | 24 GB | $0.60/hr | $432/mo |
| A100 | 40 GB | $1.29/hr | $929/mo |

**Best Lambda option**: A10 at **$432/month**

---

## Self-Hosted (One-Time Hardware Cost)

### Consumer GPUs (New)

| GPU | VRAM | Price | Suitable? |
|-----|------|-------|-----------|
| RTX 4060 Ti 16GB | 16 GB | $500 | ⚠️ Marginal (needs heavy optimization) |
| RTX 4070 Ti | 12 GB | $800 | ❌ Too small |
| RTX 4080 | 16 GB | $1,200 | ⚠️ Marginal |
| RTX 4090 | 24 GB | $1,600 | ✅ **Recommended** |
| RTX 6000 Ada | 48 GB | $7,000 | ✅ Overkill but great |

### Consumer GPUs (Used Market)

| GPU | VRAM | Used Price | Suitable? |
|-----|------|------------|-----------|
| RTX 3090 | 24 GB | $600-800 | ✅ **Best value** |
| RTX 3090 Ti | 24 GB | $800-1,000 | ✅ Good |
| Titan RTX | 24 GB | $800-1,200 | ✅ Good |
| RTX A5000 | 24 GB | $1,500-2,000 | ✅ Professional option |

**Best self-hosted option**: Used RTX 3090 at **$700**
- Break-even vs Vast.ai 24/7: ~4 months
- Break-even vs AWS spot: ~3 months
- Electricity: ~$10-20/month (300W × 24/7 × $0.12/kWh)

---

## Cost Comparison Summary (24/7 Deployment)

| Option | Monthly Cost | Setup Effort | Reliability | Best For |
|--------|--------------|--------------|-------------|----------|
| Self-hosted RTX 3090 | $0* + electricity | High | ★★★★★ | Personal/lab use |
| Vast.ai RTX 3090 | $180 | Low | ★★★☆☆ | Budget 24/7 |
| GCP L4 preemptible | $149 | Medium | ★★☆☆☆ | Budget cloud |
| AWS g5.xlarge spot | $217 | Medium | ★★★☆☆ | Budget AWS |
| RunPod serverless | $5-50 | Low | ★★★★☆ | Intermittent |
| AWS g5.xlarge on-demand | $724 | Low | ★★★★★ | Production |
| GCP L4 on-demand | $497 | Medium | ★★★★★ | Production |
| Lambda A10 | $432 | Low | ★★★★☆ | Dedicated |

*Assumes you already own the GPU; add $700 if buying used RTX 3090

---

## Recommended Deployment Strategy by Budget

### $0/month - Self-Host
- Buy used RTX 3090 ($700 one-time)
- Host on personal hardware
- Use Cloudflare Tunnel for remote access (free)
- **Total**: $700 upfront, $10-20/mo electricity

### $15-30/month - Vast.ai Part-Time
- Rent RTX 3090 on-demand
- 2-4 hours/day at $0.25/hour
- Turn off when not in use
- **Best for**: Demos, testing, development

### $50-100/month - RunPod Serverless
- Pay per request
- Handles 200-400 requests/day
- Auto-scales to zero when idle
- **Best for**: API-based usage, intermittent

### $150-200/month - Spot/Preemptible
- GCP L4 preemptible ($149/mo)
- AWS g5.xlarge spot ($217/mo)
- Vast.ai RTX 3090 reserved ($180/mo)
- **Best for**: Budget 24/7, can handle occasional restarts

### $400-500/month - Stable Cloud
- Lambda A10 ($432/mo)
- GCP L4 on-demand ($497/mo)
- **Best for**: Small production deployment

### $700+/month - Enterprise
- AWS g5 on-demand ($724/mo)
- Add load balancer, monitoring, backups
- **Best for**: Public-facing production

---

## How to Minimize Costs

### 1. Use Spot/Preemptible Instances (Save 70%)
- Set up auto-restart on termination
- Save checkpoints frequently
- Use health checks to detect failures

### 2. Auto-Shutdown When Idle (Save 50-80%)
```python
# Shutdown after 30 min idle
if idle_time > 1800:
    subprocess.run(["shutdown", "-h", "now"])
```

### 3. Use Smaller Model (Save 60% VRAM)
- Retrain with Gemma-2B instead of Gemma-3-4B
- Reduces VRAM from 20GB → 8GB
- Enables $0.15/hr GPUs (RTX 3060 Ti)

### 4. Apply 4-bit Quantization (Save 50% VRAM)
- Use bitsandbytes quantization
- Reduces VRAM by ~50% (20GB → 10GB)
- Minor quality loss (~2-3%)

### 5. Limit Concept Layers (Save 40% Memory)
- Load only layers 0-2 (not 0-5)
- Fewer lenses = less memory
- Faster inference

### 6. Share Costs (Save 75%)
- Split instance with 3 friends
- $180/mo ÷ 4 = $45/person
- Implement per-user quotas

---

## Monthly Cost Calculator

**Formula**: `hourly_rate × hours_per_day × 30 days`

Examples:
- **Vast.ai part-time**: $0.25/hr × 2 hr/day × 30 = **$15/month**
- **Vast.ai half-day**: $0.25/hr × 12 hr/day × 30 = **$90/month**
- **Vast.ai 24/7**: $0.25/hr × 24 hr/day × 30 = **$180/month**
- **AWS spot 24/7**: $0.302/hr × 24 × 30 = **$217/month**
- **GCP preempt 24/7**: $0.207/hr × 24 × 30 = **$149/month**

---

## Reality Check: What Does $X Get You?

### $15/month
- Vast.ai RTX 3090 for 2 hours/day
- ~60 hours total
- Good for: Daily testing, demos

### $50/month
- Vast.ai RTX 3090 for 7 hours/day
- RunPod serverless 300-500 requests/day
- Good for: Regular development

### $100/month
- Vast.ai RTX 3090 for 14 hours/day
- Good for: Business hours deployment

### $180/month
- Vast.ai RTX 3090 24/7
- Good for: Always-on for small team

### $500/month
- GCP L4 on-demand 24/7
- Good for: Production deployment

---

## Final Recommendation

**For most users**: Start with **Vast.ai on-demand** at $0.25/hr
- Only pay when using
- No commitment
- Scale up if needed

**For researchers**: Buy a **used RTX 3090** for $700
- Break-even in 4 months vs Vast.ai 24/7
- Full control
- No recurring costs (except electricity)

**For production**: Use **GCP L4 preemptible** at $149/mo
- Best price/performance
- Managed infrastructure
- Implement auto-restart

**Avoid**: AWS/GCP/Azure on-demand unless budget > $500/mo or need SLA

---

## Questions?

**Q: Why is cloud GPU so expensive?**
A: GPU shortage + AI boom. Demand outpaces supply. Consumer GPUs have better price/performance.

**Q: Can I use free tiers?**
A: Google Colab Pro ($10/mo) works for experiments but not 24/7. No true free GPU options exist.

**Q: What about CPU-only?**
A: Possible but very slow (10-30 sec per response). Not practical for real-time detection.

**Q: Is self-hosting worth it?**
A: If you plan to run 24/7 for 4+ months, yes. Otherwise use cloud on-demand.

**Q: What's the absolute minimum?**
A: $15/month on Vast.ai (2 hours/day). $0 if you already own 24GB GPU.
