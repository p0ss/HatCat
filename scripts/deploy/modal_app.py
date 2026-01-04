"""
Modal deployment for HatCat - complete stack.

Deploys both:
- HatCat GPU backend (Gemma + lens pack + divergence analysis)
- OpenWebUI frontend (chat interface)

Usage:
    modal deploy scripts/deploy/modal_app.py
"""

import modal

app = modal.App("hatcat")

# =============================================================================
# Image: Everything needed for HatCat + OpenWebUI
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "build-essential")
    # Node.js for OpenWebUI frontend
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "git lfs install",
    )
    # Clone repos
    .run_commands(
        "git clone https://github.com/p0ss/HatCat.git /app",
        "git clone https://github.com/p0ss/HatCat-OpenWebUI.git /ui",
    )
    # Install HatCat deps from requirements.txt
    .run_commands(
        "pip install -r /app/requirements.txt && python -c \"import nltk; nltk.download('wordnet')\"",
    )
    # Download lens pack from HuggingFace
    .run_commands(
        "cd /app && git clone https://huggingface.co/HatCatFTW/lens-gemma-3-4b-first-light-v1 lens_packs/gemma-3-4b_first-light-v1-bf16"
    )
    # Install OpenWebUI from pyproject.toml + build frontend
    .run_commands(
        "cd /ui && pip install .",
        "cd /ui && npm ci",
        "cd /ui && npm run build",
    )
)

# Persistent volumes
model_volume = modal.Volume.from_name("hatcat-models", create_if_missing=True)
data_volume = modal.Volume.from_name("hatcat-data", create_if_missing=True)

# =============================================================================
# Configuration
# =============================================================================

# Users authenticate via OpenWebUI's built-in auth
# First user to sign up becomes admin

# =============================================================================
# Combined Server
# =============================================================================

@app.cls(
    image=image,
    gpu="T4",  # 16GB VRAM
    timeout=600,
    scaledown_window=300,  # Scale to zero after 5 min
    volumes={
        "/root/.cache/huggingface": model_volume,
        "/data": data_volume,
    },
)
class HatCat:

    @modal.enter()
    def startup(self):
        import os
        import sys
        import subprocess

        # Start HatCat backend on port 8765
        sys.path.insert(0, "/app")
        os.chdir("/app")

        # Start HatCat server in background
        self.hatcat_proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "src.ui.openwebui.server:app",
             "--host", "0.0.0.0", "--port", "8765"],
            cwd="/app",
            env={**os.environ, "PYTHONPATH": "/app"}
        )

        # Configure OpenWebUI to use local HatCat
        os.environ["OPENAI_API_BASE_URL"] = "http://localhost:8765/v1"
        os.environ["OPENAI_API_KEY"] = "sk-local"
        os.environ["DATA_DIR"] = "/data"
        os.environ["DATABASE_URL"] = "sqlite:////data/webui.db"

        print("HatCat backend starting on :8765")
        print("OpenWebUI will connect to local HatCat")

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'hatcat_proc'):
            self.hatcat_proc.terminate()

    @modal.asgi_app()
    def web(self):
        import sys
        sys.path.insert(0, "/ui/backend")

        # Import OpenWebUI app
        from open_webui.main import app
        return app


# =============================================================================
# Optional: Keep warm
# =============================================================================

# @app.function(schedule=modal.Cron("*/15 8-20 * * *"))
# def keep_warm():
#     import requests
#     requests.get("https://YOUR-URL.modal.run/", timeout=60)


@app.local_entrypoint()
def main():
    print("""
HatCat Modal Deployment (Complete Stack)
=========================================

Deploy:
    modal deploy scripts/deploy/modal_app.py

Access:
    https://YOUR-WORKSPACE--hatcat-hatcat-web.modal.run

First user to sign up becomes admin.
OpenWebUI connects to HatCat backend automatically.

Cost: ~$0.59/hr when active, $0 when idle.
""")
