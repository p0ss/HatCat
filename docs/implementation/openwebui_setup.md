# OpenWebUI Integration Setup

## Prerequisites

You need **OpenWebUI** installed separately. This is the chat interface that will display the divergence visualization.

### Install OpenWebUI (if you don't have it)

**Option 1: Docker (Recommended)**
```bash
docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then open http://localhost:3000

**Option 2: pip**
```bash
pipx install open-webui
pipx run open-webui serve
```

Then open http://localhost:8080

user icon in top right" then "settings with gear icon" then "admin settings" then "

For more installation options, see: https://docs.openwebui.com/getting-started/
  1. Open http://localhost:8080 in your browser
  2. user icon in top right" 
  3. then "settings with gear icon" 
  4. then "admin settings"
  5. Navigate to Connections → OpenAI API
  6. Set Base URL: http://localhost:8765/v1
  7. Set API Key: sk-dummy (can be anything)
  8. Save settings
  9. The hatcat-divergence model should appear in the model selector
  
We also need to install the visualisation or you'll see nothing different from normal chat 

  1. go back to admin settings
  2. click the functions tab 
  3. click new function 
  4. paste in all the code from /src/openwebui/hatcat_visualizer.js
  5. save  
  
## Quick Start

The HatCat divergence server provides real-time token-by-token divergence visualization through an OpenAI-compatible API.

### 1. Start the Server

```bash
poetry run python src/openwebui/server.py
```

The server will:
- Load 5,582 dual lens pairs from `results/sumo_classifiers_adaptive_l0_5/`
- Load the Gemma-3-4b model on CUDA
- Start on `http://0.0.0.0:8765`

Expected output:
```
🎩 Initializing HatCat divergence analyzer...
✓ Loaded metadata for 121512 concepts across 7 layers
✓ Base layers loaded: 14 lenses
✓ Loaded 14 activation lenses
✓ Loaded 14 text lenses
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### 2. Test the Server

Health check:
```bash
curl http://localhost:8765/
```

Should return:
```json
{
  "name": "HatCat Divergence API",
  "status": "ready",
  "activation_lenses": 14,
  "text_lenses": 14
}
```

### 3. Connect OpenWebUI to HatCat Server

Now you have:
- **OpenWebUI running** at http://localhost:3000 (or :8080)
- **HatCat server running** at http://localhost:8765

#### Step-by-step Connection:

1. **Open OpenWebUI** in your browser (http://localhost:3000)

2. **Go to Settings** (click your profile icon → Settings)

3. **Navigate to Connections** → Admin Panel → Connections

4. **Add OpenAI API Connection**:
   - Click "+" to add new connection
   - **API Base URL**: `http://localhost:8765/v1`
   - **API Key**: `dummy` (any value works, not validated)
   - Click "Save"

5. **Select the Model**:
   - Go back to chat
   - Click model selector dropdown
   - You should see "hatcat-divergence"
   - Select it

6. **Start Chatting**!
   - Type a message
   - Watch tokens appear with color coding
   - Hover over tokens to see divergence details

#### Alternative: Use as Pipeline (Advanced)

If you prefer using OpenWebUI's pipeline system:

```bash
# Copy the pipeline file
cp src/openwebui/divergence_pipeline.py ~/.local/share/openwebui/pipelines/

# Restart OpenWebUI
docker restart open-webui  # if using Docker
# or restart the pip version
```

The pipeline will appear in Settings → Pipelines.

## What You'll See

When you chat with the model through OpenWebUI, each token will be:

### Color Coded by Divergence (Current Thresholds)
- 🟢 **Green** (div < 0.707): Model's thoughts align with text
- 🟡 **Yellow** (0.707 - 0.842): Moderate divergence
- 🔴 **Red** (div > 0.842): High divergence - model thinking differently than writing

### Hover Tooltips Show
- **Max divergence score** for that token
- **🧠 Model Thinks**: Top concepts detected in activations
- **📝 Model Writes**: Top concepts detected in text
- **⚠️ Top Divergences**: Biggest mismatches between thinking and writing

## Example Output

```
Query: "What is a physical object?"

Response (with metadata):
Token: " is"
  🔴 Divergence: 0.963
  🧠 Model Thinks: Object (0.96), ContentBearingPhysical (0.89)
  📝 Model Writes: (none)
  ⚠️ Top Divergences:
    - Object: Δ=0.963 (thinks:0.96, writes:0.00)
    - ContentBearingPhysical: Δ=0.888 (thinks:0.89, writes:0.00)
```

## Configuration

### Adjust Divergence Thresholds

Edit `src/openwebui/server.py` or `divergence_pipeline.py`:

```python
class Valves(BaseModel):
    DIVERGENCE_THRESHOLD_LOW: float = 0.707   # Green/yellow boundary
    DIVERGENCE_THRESHOLD_HIGH: float = 0.842  # Yellow/red boundary
```

Based on our analysis of 1,000 tokens:
- **Tertiles** (balanced): low=0.707, high=0.842 (current)
- **Quartiles** (more green): low=0.703, high=0.857
- **Strict** (mostly red): low=0.6, high=0.9

### Use Different Lens Layers

```python
class Valves(BaseModel):
    LENS_DIR: str = "results/sumo_classifiers_adaptive_l0_5"
    BASE_LAYERS: List[int] = [0]  # Change to [0, 1] for multi-layer
```

### Limit Generation Length

```python
class ChatCompletionRequest(BaseModel):
    max_tokens: int = 512  # Adjust default
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8765/
```

### List Models
```bash
GET http://localhost:8765/v1/models
```

Returns:
```json
{
  "object": "list",
  "data": [{
    "id": "hatcat-divergence",
    "object": "model",
    "created": 1234567890,
    "owned_by": "hatcat"
  }]
}
```

### Chat Completions (Streaming)
```bash
POST http://localhost:8765/v1/chat/completions
Content-Type: application/json

{
  "model": "hatcat-divergence",
  "messages": [
    {"role": "user", "content": "What is consciousness?"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 100
}
```

Returns Server-Sent Events stream with divergence metadata.

## Testing Locally

Use the test script:

```bash
poetry run python scripts/test_divergence_server.py
```

This will:
1. Query: "What is a physical object?"
2. Show each token with:
   - Color indicator (🟢🟡🔴)
   - Divergence score
   - Top divergent concepts
3. Display full response at the end

## Troubleshooting

### Server won't start - CUDA OOM

If you see `torch.OutOfMemoryError`, another process is using GPU memory.

**Solution**: Kill other GPU processes or use CPU mode:

Edit `src/openwebui/server.py`:
```python
# Change from:
device_map="cuda"

# To:
device_map="cpu"
```

Note: CPU mode is ~10x slower but works without GPU.

### Lenses not found

Error: `FileNotFoundError: results/sumo_classifiers_adaptive_l0_5`

**Solution**: You need to train the lenses first:

```bash
poetry run python scripts/train_sumo_classifiers.py \
  --layers 0 1 2 3 4 5 \
  --output-dir results/sumo_classifiers_adaptive_l0_5
```

This takes ~12 hours to train 5,582 dual lens pairs.

### Port already in use

Error: `Address already in use`

**Solution**: Kill the existing server:

```bash
# Find process using port 8765
lsof -ti:8765 | xargs kill -9

# Or change port in server.py:
uvicorn.run(app, host="0.0.0.0", port=8766)
```

### OpenWebUI can't connect

1. Check firewall allows port 8765
2. Try `http://127.0.0.1:8765` instead of `localhost`
3. If OpenWebUI is in Docker, use host IP: `http://172.17.0.1:8765`

## Performance

### Token Generation Speed
- **With divergence analysis**: ~0.5-1 tokens/sec
- **Without divergence**: ~10-20 tokens/sec

The slowdown is due to:
1. Running 14 activation lenses per token (MLP inference)
2. Running 14 text lenses per token (TF-IDF + LogReg)
3. Calculating divergences and formatting metadata

### Memory Usage
- **Model (Gemma-3-4b)**: ~18 GB GPU memory
- **Lenses (14 pairs)**: ~200 MB
- **Total**: ~18.2 GB

## Advanced Usage

### Multi-Layer Analysis

Load multiple lens layers for deeper concept detection:

```python
manager = DynamicLensManager(
    lenses_dir=Path('results/sumo_classifiers_adaptive_l0_5'),
    base_layers=[0, 1, 2],  # Multiple layers
    keep_top_k=200,          # Keep more lenses in memory
)
```

This will load:
- Layer 0: 14 concepts (always loaded)
- Layer 1: 276 concepts (loaded when layer 0 activates)
- Layer 2: 1,059 concepts (loaded when layer 1 activates)

### Batch Processing

For analyzing many prompts, use the test scripts instead of the server:

```bash
poetry run python scripts/analyze_divergence_distribution.py
poetry run python scripts/test_self_concept_divergence.py
```

These save results to JSON files for later analysis.

## Next Steps

- See `docs/dual_lens_divergence_detection.md` for technical details
- Explore `results/self_concept_divergence_test/` for example temporal slices
- Try implementing brightness=divergence, hue=concept-group visualization

## Files

- **Server**: `src/openwebui/server.py`
- **Pipeline**: `src/openwebui/divergence_pipeline.py`
- **Test script**: `scripts/test_divergence_server.py`
- **Lenses**: `results/sumo_classifiers_adaptive_l0_5/`
