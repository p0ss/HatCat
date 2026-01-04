# HatCat Cloud Deployment

Deploy HatCat on Modal with scale-to-zero GPU.

## Quick Start

```bash
cd /home/poss/Documents/Code/HatCat

# Install modal (in deploy group)
poetry install --with deploy

# Authenticate with Modal
poetry run modal setup

# Deploy
poetry run modal deploy scripts/deploy/modal_app.py
```

That's it. The deployment:
- Clones HatCat and hatcat-ui from GitHub
- Installs deps from `requirements.txt` and `pyproject.toml`
- Downloads lens pack from HuggingFace
- Caches models in a persistent volume
- Scales to zero when idle (5 min timeout)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │     Modal Container (T4 GPU)        │
                    │                                     │
Users ──────────────│──▶ OpenWebUI ──▶ HatCat Backend    │
                    │     (UI)          │                 │
                    │                   ├── Gemma 3 4B    │
                    │                   ├── Gemma 2B      │
                    │                   └── Lens Pack     │
                    └─────────────────────────────────────┘
```

Everything in one container. Scales to zero together.

## Cost

| Usage | Monthly Cost |
|-------|-------------|
| Idle | $0 |
| Light (1hr/day) | ~$20 |
| Medium (3hr/day) | ~$55 |
| Heavy (8hr/day) | ~$150 |

T4 GPU: $0.59/hr, billed per second.

## Access Control

OpenWebUI has built-in auth:
- First user to sign up becomes admin
- Admin can invite users or enable open registration
- User data persists in `/data` volume

## Spending Limits

```bash
modal config set spending-limit 50  # $50/month cap
```

## Keep Warm (Optional)

Uncomment the `keep_warm` function to ping every 15 min during business hours, reducing cold start latency.

## Files

- `modal_app.py` - Complete deployment (HatCat + OpenWebUI)
