"""
Streamlit component wrapper for HatCat timeline visualization.

Uses streamlit.components.v1.html to embed the PixiJS widget.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def _calculate_viz_height(
    reply_data: Dict[str, Any],
    zoom: str,
    max_width: int,
    top_concept_count: int
) -> int:
    """Calculate required height for visualization based on content."""
    column_width = 50
    row_height = 50
    label_column_width = 80
    padding = 20

    available_width = max_width - label_column_width - padding

    # Estimate number of wrapped lines based on zoom level
    if zoom == "chat":
        # Chat view: same as reply but no concept tracks
        tokens_per_line = max(1, available_width // column_width)
        num_tokens = len(reply_data.get("tokens", []))
        wrapped_lines = max(1, (num_tokens + tokens_per_line - 1) // tokens_per_line)
    elif zoom == "reply":
        # Reply view: all tokens in sequence
        tokens_per_line = max(1, available_width // column_width)
        num_tokens = len(reply_data.get("tokens", []))
        wrapped_lines = max(1, (num_tokens + tokens_per_line - 1) // tokens_per_line)
    elif zoom == "paragraph":
        # Paragraph view: similar to sentence but with gaps at paragraphs
        tokens_per_line = max(1, available_width // (column_width + 25))
        num_tokens = len(reply_data.get("tokens", []))
        wrapped_lines = max(1, (num_tokens + tokens_per_line - 1) // tokens_per_line)
    elif zoom == "sentence":
        # Sentence view: similar to reply but with gaps
        tokens_per_line = max(1, available_width // (column_width + 20))  # Account for gaps
        num_tokens = len(reply_data.get("tokens", []))
        wrapped_lines = max(1, (num_tokens + tokens_per_line - 1) // tokens_per_line)
    else:  # token
        # Token view: larger columns to fit concept labels
        token_column_width = column_width * 3  # Match grid_layout.ts
        tokens_per_line = max(1, int(available_width // token_column_width))
        num_tokens = len(reply_data.get("tokens", []))
        wrapped_lines = max(1, (num_tokens + tokens_per_line - 1) // tokens_per_line)

    # Each wrapped line has: 1 token row + top_concept_count concept rows (except chat view)
    if zoom == "chat":
        rows_per_line = 1  # Just token rows in chat view
    else:
        rows_per_line = 1 + top_concept_count
    total_rows = wrapped_lines * rows_per_line

    # Calculate total height with padding
    calculated_height = (total_rows * row_height) + (padding * 2) + 100  # +100 for controls

    # Clamp to reasonable range - lower minimum for chat view
    min_height = 200 if zoom == "chat" else 300
    return max(min_height, min(calculated_height, 3000))


def render_timeline_viz(
    reply_data: Dict[str, Any],
    initial_zoom: str = "reply",
    max_width: int = 800,
    height: Optional[int] = None,
    top_concept_count: int = 5
) -> None:
    """
    Render HatCat timeline visualization in Streamlit.

    Args:
        reply_data: Data in ReplyData format with tokens, sentences, and reply
        initial_zoom: Initial zoom level ("reply", "sentence", or "token")
        max_width: Maximum width of the visualization
        height: Height of the visualization iframe (if None, calculated dynamically)
        top_concept_count: Number of concept tracks to show
    """
    # Calculate dynamic height based on content if not provided
    if height is None:
        height = _calculate_viz_height(reply_data, initial_zoom, max_width, top_concept_count)
    # Get path to the bundled JavaScript
    viz_dir = Path(__file__).parent.parent.parent / "ui" / "timeline_viz"
    bundle_path = viz_dir / "dist" / "bundle.js"

    # Read the bundled JavaScript
    if not bundle_path.exists():
        st.error(f"Timeline viz bundle not found at {bundle_path}. Run `npm run build` in src/ui/timeline_viz/")
        return

    with open(bundle_path, 'r') as f:
        bundle_js = f.read()

    # Serialize data to JSON
    data_json = json.dumps(reply_data)

    # Build HTML with embedded JavaScript
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }}
        #viz-container {{
            width: 100%;
            overflow: visible;
        }}
        .controls {{
            position: sticky;
            bottom: 0;
            left: 0;
            right: 0;
            display: flex;
            justify-content: center;
            gap: 8px;
            padding: 12px;
            background: rgba(26, 26, 26, 0.95);
            border-top: 1px solid #3a3a3a;
            z-index: 100;
        }}
        .zoom-btn {{
            background: #48a4a3;
            color: #1a1a1a;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: background 0.2s;
        }}
        .zoom-btn:hover {{
            background: #5bb5b4;
        }}
        .zoom-btn.active {{
            background: #de563f;
            color: white;
        }}
        .error {{
            color: #de563f;
            padding: 20px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div id="viz-container"></div>
    <div class="controls">
        <button id="btn-chat" class="zoom-btn">Chat</button>
        <button id="btn-reply" class="zoom-btn">Reply</button>
        <button id="btn-paragraph" class="zoom-btn">Paragraph</button>
        <button id="btn-sentence" class="zoom-btn">Sentence</button>
        <button id="btn-token" class="zoom-btn">Token</button>
    </div>

    <script>
        {bundle_js}
    </script>

    <script>
        try {{
            // Initialize visualization using global HatCatViz
            const container = document.getElementById('viz-container');
            const data = {data_json};

            if (typeof HatCatViz === 'undefined' || !HatCatViz.initHatCatViz) {{
                throw new Error('HatCatViz.initHatCatViz not found');
            }}

            const viz = HatCatViz.initHatCatViz(container, {{
                text: "",
                data: data,
                options: {{
                    maxWidth: {max_width},
                    initialZoom: "{initial_zoom}",
                    topConceptCount: {top_concept_count}
                }}
            }});

            // Setup zoom controls
            const btnChat = document.getElementById('btn-chat');
            const btnReply = document.getElementById('btn-reply');
            const btnParagraph = document.getElementById('btn-paragraph');
            const btnSentence = document.getElementById('btn-sentence');
            const btnToken = document.getElementById('btn-token');

            let currentZoom = "{initial_zoom}";

            function updateActiveButton() {{
                btnChat.classList.toggle('active', currentZoom === 'chat');
                btnReply.classList.toggle('active', currentZoom === 'reply');
                btnParagraph.classList.toggle('active', currentZoom === 'paragraph');
                btnSentence.classList.toggle('active', currentZoom === 'sentence');
                btnToken.classList.toggle('active', currentZoom === 'token');
            }}

            updateActiveButton();

            // Listen for zoom changes from scroll/wheel
            window.addEventListener('hatcat:zoomchange', (e) => {{
                currentZoom = e.detail.zoom;
                updateActiveButton();
            }});

            btnChat.addEventListener('click', () => {{
                currentZoom = 'chat';
                viz.setZoom('chat');
            }});

            btnReply.addEventListener('click', () => {{
                currentZoom = 'reply';
                viz.setZoom('reply');
            }});

            btnParagraph.addEventListener('click', () => {{
                currentZoom = 'paragraph';
                viz.setZoom('paragraph');
            }});

            btnSentence.addEventListener('click', () => {{
                currentZoom = 'sentence';
                viz.setZoom('sentence');
            }});

            btnToken.addEventListener('click', () => {{
                currentZoom = 'token';
                viz.setZoom('token', {{ tokenId: 0 }});
            }});

            // Listen for canvas resize and notify Streamlit
            window.addEventListener('hatcat:zoomchange', () => {{
                // Wait for render to complete
                setTimeout(() => {{
                    const canvas = document.querySelector('canvas');
                    if (canvas) {{
                        const newHeight = canvas.height + 60; // +60 for controls
                        window.parent.postMessage({{
                            type: 'streamlit:setFrameHeight',
                            height: newHeight
                        }}, '*');
                    }}
                }}, 100);
            }});
        }} catch (error) {{
            console.error('Failed to initialize HatCat viz:', error);
            document.getElementById('viz-container').innerHTML = '<div class="error">Error loading visualization: ' + error.message + '</div>';
        }}
    </script>
</body>
</html>
    """

    # Render in Streamlit using iframe with scrolling enabled
    components.html(html_content, height=height, scrolling=True)


def convert_timeline_to_reply_data(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert HatCat timeline format to ReplyData format for the PixiJS widget.

    Args:
        timeline: Timeline data from generate_with_safety_monitoring
                 Each step has: token, token_idx, concepts

    Returns:
        ReplyData dict with tokens, sentences, and reply structure
    """
    if not timeline:
        return {"tokens": [], "sentences": [], "reply": {"concepts": []}}

    # Build tokens list
    tokens = []
    sentence_boundaries = []
    current_sentence_start = 0

    for i, step in enumerate(timeline):
        token_text = step.get('token', '')
        concepts_dict = step.get('concepts', {})

        # Convert concepts to list of {id, score, layer}
        # Use probability instead of divergence
        concepts = []
        for concept_name, data in concepts_dict.items():
            # Probability should already be in 0-1 range
            prob = data.get('probability', 0)
            layer = data.get('layer', 0)
            concepts.append({
                "id": concept_name,
                "score": prob,
                "layer": layer
            })

        # Sort by score descending
        concepts.sort(key=lambda c: c['score'], reverse=True)

        tokens.append({
            "id": i,
            "text": token_text,
            "sentenceId": len(sentence_boundaries),
            "timestep": i,
            "concepts": concepts
        })

        # Detect sentence boundaries
        if token_text.strip() in {'.', '!', '?', '。', '！', '？'}:
            sentence_boundaries.append((current_sentence_start, i))
            current_sentence_start = i + 1

    # Add final sentence if not ended
    if current_sentence_start < len(tokens):
        sentence_boundaries.append((current_sentence_start, len(tokens) - 1))

    # Build sentences list
    sentences = []
    for sent_id, (start, end) in enumerate(sentence_boundaries):
        # Aggregate concepts for this sentence (use max probability, keep layer info)
        sentence_concepts = {}  # {concept_name: (prob, layer)}
        for i in range(start, end + 1):
            if i < len(timeline):
                for concept_name, data in timeline[i].get('concepts', {}).items():
                    prob = data.get('probability', 0)
                    layer = data.get('layer', 0)
                    if concept_name not in sentence_concepts:
                        sentence_concepts[concept_name] = (prob, layer)
                    else:
                        # Keep the higher probability occurrence
                        if prob > sentence_concepts[concept_name][0]:
                            sentence_concepts[concept_name] = (prob, layer)

        # Convert to list
        concepts = [{"id": name, "score": score, "layer": layer} for name, (score, layer) in sentence_concepts.items()]
        concepts.sort(key=lambda c: c['score'], reverse=True)

        sentences.append({
            "id": sent_id,
            "tokenStart": start,
            "tokenEnd": end,
            "concepts": concepts[:20]  # Top 20 per sentence
        })

    # Build paragraphs (detect by double newline or group ~3-5 sentences)
    paragraphs = []
    current_para_start = 0
    sentences_per_para = 4  # Roughly 4 sentences per paragraph

    for para_id in range(0, len(sentences), sentences_per_para):
        para_end = min(para_id + sentences_per_para - 1, len(sentences) - 1)

        if para_id > len(sentences) - 1:
            break

        token_start = sentences[para_id]["tokenStart"]
        token_end = sentences[para_end]["tokenEnd"]

        # Aggregate concepts for this paragraph (keep layer info)
        para_concepts = {}  # {concept_name: (prob, layer)}
        for i in range(token_start, token_end + 1):
            if i < len(timeline):
                for concept_name, data in timeline[i].get('concepts', {}).items():
                    prob = data.get('probability', 0)
                    layer = data.get('layer', 0)
                    if concept_name not in para_concepts:
                        para_concepts[concept_name] = (prob, layer)
                    else:
                        # Keep the higher probability occurrence
                        if prob > para_concepts[concept_name][0]:
                            para_concepts[concept_name] = (prob, layer)

        concepts = [{"id": name, "score": score, "layer": layer} for name, (score, layer) in para_concepts.items()]
        concepts.sort(key=lambda c: c['score'], reverse=True)

        paragraphs.append({
            "id": len(paragraphs),
            "sentenceStart": para_id,
            "sentenceEnd": para_end,
            "tokenStart": token_start,
            "tokenEnd": token_end,
            "concepts": concepts[:25]  # Top 25 per paragraph
        })

    # Build reply-level aggregation (use max probability, keep layer info)
    reply_concepts = {}  # {concept_name: (prob, layer)}
    for step in timeline:
        for concept_name, data in step.get('concepts', {}).items():
            prob = data.get('probability', 0)
            layer = data.get('layer', 0)
            if concept_name not in reply_concepts:
                reply_concepts[concept_name] = (prob, layer)
            else:
                # Keep the higher probability occurrence
                if prob > reply_concepts[concept_name][0]:
                    reply_concepts[concept_name] = (prob, layer)

    # Convert to list
    reply_concepts_list = [{"id": name, "score": score, "layer": layer} for name, (score, layer) in reply_concepts.items()]
    reply_concepts_list.sort(key=lambda c: c['score'], reverse=True)

    return {
        "tokens": tokens,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "reply": {"concepts": reply_concepts_list[:30]}  # Top 30 for reply
    }
