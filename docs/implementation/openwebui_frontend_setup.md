# OpenWebUI Frontend Setup

Instructions for integrating the HatCat visualizer into OpenWebUI.

## Installation

### Option 1: Custom Function (Recommended)

1. **Start OpenWebUI** (if not already running):
   ```bash
   # Using Docker
   docker run -d -p 3000:8080 --name open-webui ghcr.io/open-webui/open-webui:main

   # Or if installed locally
   open-webui serve
   ```

2. **Open OpenWebUI** in your browser:
   - Navigate to `http://localhost:3000`
   - Sign in or create an account

3. **Add HatCat as a Custom Function**:
   - Click on your profile → **Settings**
   - Go to **Functions** tab
   - Click **+ Create New Function**
   - Copy the contents of `src/openwebui/hatcat_visualizer.js`
   - Paste into the function editor
   - Name it "HatCat Divergence Visualizer"
   - Click **Save**

4. **Configure the API endpoint**:
   - In the function code, verify `HATCAT_API_BASE` points to your server:
     ```javascript
     const HATCAT_API_BASE = 'http://localhost:8765/v1';
     ```

5. **Add HatCat as a Model**:
   - Go to **Settings** → **Models**
   - Click **+** to add external model
   - Enter connection details:
     - **Base URL**: `http://localhost:8765/v1`
     - **API Key**: (leave empty or use dummy value)
     - **Model ID**: `hatcat-divergence`
   - Click **Save**

### Option 2: Pipeline Function

Alternatively, you can use OpenWebUI's Pipeline system:

1. Create a new pipeline:
   - Go to **Settings** → **Pipelines**
   - Click **+ Create Pipeline**
   - Use the HatCat visualizer code as a pipeline filter

2. Configure pipeline to:
   - Intercept streaming responses
   - Process metadata and render colored tokens
   - Forward to OpenWebUI chat interface

## Usage

### Basic Chat with Divergence Visualization

1. **Select HatCat model** in the chat interface
2. **Send a message** - tokens will be colored based on concept divergence
3. **Hover over tokens** to see divergence details:
   - Top divergent concepts
   - Activation vs text lens scores
   - Concept palette
   - Active steerings

### Adding Steering

**Method 1: Right-click on tokens**
1. Right-click on any colored token
2. Select a concept from the menu
3. Click **Amplify** (+50%) or **Suppress** (-50%)
4. The steering is immediately applied to future generations

**Method 2: Steering Panel**
- View the steering panel on the right side
- Shows all active steerings with:
  - Concept name
  - Strength bar (visual indicator)
  - Source (user/model)
  - Remove button
- Click **Clear All** to remove all steerings

### Managing Steerings

**View active steerings:**
```bash
curl http://localhost:8765/v1/steering/list?session_id=default
```

**Add a steering manually:**
```bash
curl -X POST http://localhost:8765/v1/steering/add \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "default",
    "concept": "Proposition",
    "layer": 0,
    "strength": 0.5,
    "reason": "Amplify logical reasoning"
  }'
```

**Remove a steering:**
```bash
curl -X DELETE "http://localhost:8765/v1/steering/remove/Proposition?session_id=default"
```

**Clear all steerings:**
```bash
curl -X DELETE "http://localhost:8765/v1/steering/clear?session_id=default"
```

## Features

### Token Visualization
- **Sunburst colors**: Each token colored based on dominant concept from ontology
- **Color intensity**: Brightness indicates divergence magnitude
- **Hover tooltips**: Detailed divergence breakdown
- **Palette swatches**: Top 5 contributing concepts

### Divergence Details
Tooltip shows:
- **Max Divergence**: Highest divergence score
- **Top Concepts**: Up to 3 most divergent concepts with:
  - Activation lens probability
  - Text lens probability
  - Divergence score (absolute difference)
- **Concept Palette**: Visual breakdown of top 5 concepts
- **Active Steerings**: Currently applied concept amplifications/suppressions

### Steering Controls
- **Right-click menu**: Quick access to amplify/suppress concepts
- **Steering panel**: Manage all active steerings
- **Real-time updates**: Changes apply immediately to next generation
- **Visual indicators**: ⚡ emoji on tokens affected by steering

### Color Coding
- **Hue**: Position in concept ontology (sunburst chart)
- **Saturation**: Concept clustering/specificity
- **Lightness**: Divergence magnitude (darker = higher divergence)
- **Text color**: Automatically contrasted for readability (black/white)

## Troubleshooting

### Tokens not colored
- Check that HatCat server is running: `curl http://localhost:8765/`
- Verify model endpoint in OpenWebUI settings
- Check browser console for errors

### Steering not working
- Verify steering was added: `curl http://localhost:8765/v1/steering/list`
- Check session_id matches between requests
- Look for steering indicator (⚡) on tokens
- Restart generation after adding steering

### API connection errors
- Ensure HatCat server is accessible from OpenWebUI
- If using Docker, check network configuration
- Update `HATCAT_API_BASE` in visualizer code if needed

### CORS issues
- HatCat server has CORS enabled (`allow_origins=["*"]`)
- If issues persist, check browser console
- May need to run OpenWebUI and HatCat on same host

## Advanced Configuration

### Custom Session IDs
Pass `session_id` in chat completion request:
```javascript
{
  "model": "hatcat-divergence",
  "messages": [...],
  "session_id": "user_123",  // Custom session
  "stream": true
}
```

### Steering Presets
Create steering presets for common use cases:

**Amplify logical reasoning:**
```bash
curl -X POST http://localhost:8765/v1/steering/add -d '{
  "concept": "Proposition",
  "strength": 0.7,
  "layer": 0
}'
```

**Suppress temporal concepts:**
```bash
curl -X POST http://localhost:8765/v1/steering/add -d '{
  "concept": "Process",
  "strength": -0.5,
  "layer": 0
}'
```

### Keyboard Shortcuts (Future)
- `Ctrl+H`: Toggle help panel
- `Ctrl+S`: Toggle steering panel
- `Ctrl+K`: Clear all steerings
- `Escape`: Close panels/menus

## Next Steps

- **Phase 5**: Enhanced steering UI with sunburst concept chart
- **Phase 6**: Allow model to request steering changes via tool calls
- **Phase 7**: Documentation and sharing features

## Support

For issues or questions:
- GitHub: https://github.com/p0ss/HatCat/issues
- Documentation: `/docs/` directory in repository
