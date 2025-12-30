# HatCat Streamlit UI

Lightweight chat interface for HatCat with real-time AI safety concept monitoring and temporal activation visualization.

## Quick Start

```bash
# Start the Streamlit UI
streamlit run src/ui/streamlit_chat.py

# Or with custom port
streamlit run src/ui/streamlit_chat.py --server.port 8080

# Create public share link (note: not recommended for sensitive data)
streamlit run src/ui/streamlit_chat.py --server.headless=true
```

Then open http://localhost:8501 in your browser.

## Features

### Chat Interface
- Clean chat interface with message history
- Built-in example prompts for quick testing
- Real-time AI safety concept detection
- Configurable generation parameters (max tokens, target layer)

### AI Safety Concept Monitoring
- Highlights AI safety-related concepts during generation
- Shows top detected safety concepts with activation strength
- Covers concepts like: deception, manipulation, alignment, harm, transparency, etc.
- Layer-by-layer concept tracking

### Temporal Activation Visualization
- ASCII sparkline visualization of concept activations over time
- Configurable top-K concepts (1-10, default 5)
- Multiple aggregation levels:
  - **Per-timestep**: Individual activation samples
  - **Per-token**: Aggregated by token emission
  - **Per-sentence**: Aggregated by sentence boundaries
  - **Per-reply**: Overall reply summary
- Flexible timerange selection for top-K:
  - **Timestep**: Top concepts change at each step
  - **Sentence**: Top concepts per sentence
  - **Reply**: Global top concepts across entire reply
- Detailed view showing all activations per timestep
- Granularity-agnostic design (ready for intratoken/pre-token activations)

## Architecture

The Streamlit UI uses HatCat's `DynamicLensManager` for concept detection:

```
User Input → Streamlit UI → Model + DynamicLensManager → Response
                                  ↓
                     Concept Activations & Timeline Data
                                  ↓
                          Temporal Visualization
```

## Iteration Workflow

1. **Rapid prototyping**: Test UI ideas in Streamlit (Python only, fast iteration)
2. **User feedback**: Get feedback on UX patterns
3. **Backport**: Port successful patterns to OpenWebUI fork (TypeScript/Vue)

## vs OpenWebUI Fork

**Streamlit UI (This)**:
- ✅ Built-in to HatCat repo
- ✅ One-command quickstart
- ✅ Python-only (fast iteration)
- ✅ Perfect for demos and testing
- ✅ Interactive controls and visualizations
- ❌ Single-user focus
- ❌ Limited customization compared to web app

**OpenWebUI Fork**:
- ✅ Full-featured chat application
- ✅ User management, plugins, RAG
- ✅ Production-ready
- ✅ Beautiful modern UI
- ✅ Multi-user support
- ❌ Separate repository
- ❌ Heavier setup (Node.js + Python)
- ❌ Slower iteration (TypeScript compilation)

## Development

### Adding Custom Visualizations

Streamlit provides easy integration of visualizations. See `src/ui/temporal_viz.py` for the sparkline implementation.

```python
# In src/ui/streamlit_chat.py

# Add new visualization button
if st.button("Show Custom Viz"):
    viz_data = process_timeline(st.session_state.last_timeline)
    st.plotly_chart(viz_data)  # or st.pyplot(), st.altair_chart(), etc.
```

### Styling

Streamlit uses themes configured in `.streamlit/config.toml`. See [Streamlit theming guide](https://docs.streamlit.io/library/advanced-features/theming).

## TODO

- [x] Migrate from Gradio to Streamlit
- [x] Add sparkline temporal visualization
- [x] Add configurable aggregation and timerange options
- [x] Support granularity-agnostic design for future intratoken activations
- [ ] Add concept graph visualization (plotly/networkx)
- [ ] Integrate temporal viz into OpenWebUI fork
- [ ] Add steering controls (adjust concept weights)
- [ ] Add session management (save/load conversations)
- [ ] Add intratoken/pre-token activation support when available
- [ ] Export successful UX patterns to OpenWebUI
