# OpenWebUI Fork Plan for HatCat Visualization

## Strategy

Fork OpenWebUI with minimal changes to add token-level divergence visualization, then submit upstream as an optional feature.

## Architecture

### Modified Components

1. **`src/lib/components/chat/Messages/ResponseMessage.svelte`**
   - Check for `metadata.divergence` in streaming chunks
   - Render tokens with inline styles based on color
   - Add tooltip on hover

2. **`src/lib/components/chat/Messages/TokenTooltip.svelte`** (new)
   - Reusable tooltip component for divergence details
   - Shows top concepts, activation/text scores, palette

3. **`src/lib/stores/settings.js`**
   - Add setting to enable/disable HatCat visualization
   - Default: auto-detect (enable if metadata present)

### Data Flow

```
HatCat Server (localhost:8765)
  ↓ SSE streaming
  {
    "choices": [{
      "delta": {
        "content": "token",
        "metadata": {
          "divergence": {...},
          "color": "#ff5733",
          "palette": ["#ff5733", ...],
          "steering": {...}
        }
      }
    }]
  }
  ↓
OpenWebUI Frontend (modified)
  ↓
ResponseMessage.svelte detects metadata
  ↓
Renders: <span style="background: {color}; ...">token</span>
  ↓
User sees colored tokens with tooltips
```

## Implementation Steps

### Phase 1: Fork and Setup (15 min)

```bash
# Fork OpenWebUI on GitHub
# Clone locally
cd ~/Code
git clone https://github.com/YOUR_USERNAME/open-webui.git hatcat-ui
cd hatcat-ui

# Add upstream remote
git remote add upstream https://github.com/open-webui/open-webui.git

# Create feature branch
git checkout -b feature/hatcat-divergence-viz

# Install dependencies
npm install
```

### Phase 2: Modify ResponseMessage Component (1 hour)

**File: `src/lib/components/chat/Messages/ResponseMessage.svelte`**

Changes:
1. Add function to detect metadata in message content
2. Parse tokens and metadata from streaming chunks
3. Render tokens with inline styles when metadata present
4. Fall back to normal rendering when no metadata

```svelte
<script>
  // ... existing imports ...
  import TokenTooltip from './TokenTooltip.svelte';

  // ... existing props ...

  let tokens = [];  // [{text: "hello", metadata: {...}}, ...]
  let showTooltip = false;
  let tooltipData = null;
  let tooltipPosition = {x: 0, y: 0};

  // Parse message content for HatCat metadata
  function parseTokensWithMetadata(content) {
    // Check if content has been processed with metadata
    if (typeof content === 'object' && content.tokens) {
      return content.tokens;
    }

    // Fall back to plain text
    return [{text: content, metadata: null}];
  }

  function handleTokenHover(event, metadata) {
    if (metadata && metadata.divergence) {
      tooltipData = metadata;
      tooltipPosition = {
        x: event.clientX,
        y: event.clientY
      };
      showTooltip = true;
    }
  }

  function handleTokenLeave() {
    showTooltip = false;
  }

  // ... rest of existing code ...
</script>

<!-- Modified rendering -->
{#if message.content}
  {#each parseTokensWithMetadata(message.content) as token}
    {#if token.metadata}
      <span
        class="hatcat-token"
        style="
          background-color: {token.metadata.color};
          color: {getContrastColor(token.metadata.color)};
          padding: 2px 4px;
          border-radius: 3px;
          cursor: help;
          display: inline-block;
          transition: all 0.2s ease;
        "
        on:mouseenter={(e) => handleTokenHover(e, token.metadata)}
        on:mouseleave={handleTokenLeave}
      >
        {#if token.metadata.steering?.active}⚡{/if}
        {token.text}
      </span>
    {:else}
      {token.text}
    {/if}
  {/each}
{/if}

{#if showTooltip && tooltipData}
  <TokenTooltip data={tooltipData} position={tooltipPosition} />
{/if}
```

### Phase 3: Create Tooltip Component (30 min)

**File: `src/lib/components/chat/Messages/TokenTooltip.svelte`** (new)

```svelte
<script>
  export let data;
  export let position;

  $: divergence = data?.divergence || {};
  $: palette = data?.palette || [];
  $: steering = data?.steering || {};
</script>

<div
  class="hatcat-tooltip"
  style="
    position: fixed;
    left: {position.x + 10}px;
    top: {position.y + 10}px;
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 12px;
    border-radius: 6px;
    font-size: 12px;
    z-index: 10000;
    max-width: 350px;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  "
>
  {#if divergence.max_divergence}
    <div><strong>Max Divergence:</strong> {divergence.max_divergence.toFixed(3)}</div>
  {/if}

  {#if divergence.top_divergences && divergence.top_divergences.length > 0}
    <div style="margin-top: 8px;"><strong>Top Concepts:</strong></div>
    {#each divergence.top_divergences.slice(0, 3) as d}
      <div style="margin-left: 8px; font-size: 11px;">
        • {d.concept}: act={d.activation.toFixed(2)}, txt={d.text.toFixed(2)}, div={d.divergence.toFixed(2)}
      </div>
    {/each}
  {/if}

  {#if palette.length > 0}
    <div style="margin-top: 8px;"><strong>Palette:</strong></div>
    <div style="display: flex; gap: 4px; margin-top: 4px;">
      {#each palette.slice(0, 5) as color}
        <div style="width: 16px; height: 16px; background: {color}; border-radius: 2px;"></div>
      {/each}
    </div>
  {/if}

  {#if steering.active && steering.steerings}
    <div style="margin-top: 8px;"><strong>Active Steerings:</strong></div>
    {#each steering.steerings as s}
      <div style="margin-left: 8px; font-size: 11px;">
        • {s.concept}: {s.strength > 0 ? 'amplify' : 'suppress'} ({s.strength > 0 ? '+' : ''}{s.strength.toFixed(2)})
      </div>
    {/each}
  {/if}
</div>
```

### Phase 4: Add Settings Toggle (15 min)

**File: `src/lib/stores/settings.js`**

Add:
```javascript
export const hatcatVisualization = writable(
  localStorage.getItem('hatcat_visualization') === 'true'
);

hatcatVisualization.subscribe((value) => {
  localStorage.setItem('hatcat_visualization', value);
});
```

**File: Settings UI component**

Add checkbox:
```svelte
<label>
  <input type="checkbox" bind:checked={$hatcatVisualization} />
  Enable HatCat Divergence Visualization
</label>
```

### Phase 5: Update OpenWebUI Backend to Proxy HatCat (30 min)

**Option A: Use existing OpenAI-compatible endpoint**
- Configure HatCat server as external OpenAI endpoint
- OpenWebUI already supports this
- No backend changes needed

**Option B: Add native HatCat integration**
- Add HatCat to model providers
- Better integration but more code

Use **Option A** initially.

### Phase 6: Testing (30 min)

```bash
# Start HatCat server
cd ~/Code/HatCat
poetry run python src/openwebui/server.py

# Start OpenWebUI frontend (in fork)
cd ~/Code/hatcat-ui
npm run dev

# Open http://localhost:5173
# Configure HatCat server as external model:
#   Settings → Connections → OpenAI API
#   Base URL: http://localhost:8765/v1
#   API Key: (leave empty)
#   Models: hatcat-divergence

# Test chat with colored tokens
```

### Phase 7: Docker Build (1 hour)

Create `docker-compose.yml` for easy deployment:

```yaml
version: '3.8'

services:
  hatcat-server:
    build: ./HatCat
    ports:
      - "8765:8765"
    volumes:
      - ./lens_packs:/app/lens_packs
      - ./concept_packs:/app/concept_packs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  hatcat-ui:
    build: ./hatcat-ui
    ports:
      - "3000:8080"
    environment:
      - HATCAT_SERVER=http://hatcat-server:8765
    depends_on:
      - hatcat-server
```

Usage:
```bash
docker-compose up
# Access at http://localhost:3000
```

### Phase 8: Upstream Contribution (ongoing)

1. **Keep fork minimal** - only visualization changes
2. **Follow OpenWebUI conventions** - code style, component structure
3. **Make it optional** - don't break existing functionality
4. **Document well** - clear PR description
5. **Add tests** - component tests for token rendering
6. **Submit PR** with:
   - Title: "Add support for token-level metadata visualization"
   - Description: "Enables rendering tokens with custom colors and tooltips when metadata is present in streaming responses. Useful for interpretability tools like HatCat."
   - Flag as optional feature
   - Backward compatible

## Maintenance Strategy

**Until merged upstream:**
- Track upstream main branch
- Rebase feature branch regularly
- Keep changes minimal

**After merged upstream:**
- Delete fork
- Use official OpenWebUI
- Just maintain HatCat server

## File Structure

```
hatcat-ui/  (forked OpenWebUI)
├── src/
│   ├── lib/
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   │   ├── Messages/
│   │   │   │   │   ├── ResponseMessage.svelte  [MODIFIED]
│   │   │   │   │   └── TokenTooltip.svelte     [NEW]
│   │   └── stores/
│   │       └── settings.js  [MODIFIED]
│   └── ...
├── docker-compose.yml  [NEW]
└── README.hatcat.md    [NEW]
```

## Benefits of This Approach

1. ✅ **Leverage existing ecosystem** - auth, models, functions, security
2. ✅ **Minimal maintenance** - only ~200 lines of code to maintain
3. ✅ **Easy updates** - rebase on upstream regularly
4. ✅ **Path to contribution** - clean PR for upstream merge
5. ✅ **User-friendly** - users get full OpenWebUI + HatCat features
6. ✅ **Docker ready** - single command deployment

## Timeline

- **Phase 1-3**: Core visualization (2 hours)
- **Phase 4-5**: Settings and integration (45 min)
- **Phase 6**: Testing (30 min)
- **Phase 7**: Docker packaging (1 hour)
- **Phase 8**: Upstream PR (1-2 weeks for review)

**Total dev time: ~4.5 hours**

## Next Steps

1. Fork OpenWebUI on GitHub
2. Clone and set up local dev environment
3. Implement ResponseMessage modifications
4. Test with HatCat server
5. Package with Docker Compose
6. Document and prepare upstream PR
