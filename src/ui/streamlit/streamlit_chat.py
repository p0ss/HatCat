#!/usr/bin/env python3
"""
Streamlit chat interface for HatCat with temporal visualization.

Features:
- AI safety concept highlighting
- Temporal sparkline visualization
- Configurable aggregation and top-K selection

Usage:
    streamlit run src/ui/streamlit_chat.py
"""

import streamlit as st
import torch
from pathlib import Path
import sys
import plotly.graph_objects as go
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hat.monitoring.lens_manager import DynamicLensManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.ui.streamlit.temporal_viz import format_temporal_view, format_detailed_view
from src.ui.streamlit.components.timeline_component import render_timeline_viz, convert_timeline_to_reply_data


# AI safety concept keywords to highlight
AI_SAFETY_CONCEPTS = {
    'deception', 'manipulation', 'persuasion', 'concealing', 'predicting',
    'artificial', 'intelligence', 'agent', 'moral', 'alignment', 'harm',
    'risk', 'safety', 'transparency', 'honesty', 'trust', 'cooperation',
    'competition', 'selfishness', 'altruism', 'goal', 'objective',
    'consciousness', 'sentience', 'rights', 'welfare'
}


@st.cache_data
def get_available_lens_packs():
    """Get list of available lens packs."""
    packs = DynamicLensManager.discover_lens_packs()
    # Return sorted list of pack IDs with metadata
    return {
        pack_id: {
            'type': info['type'],
            'path': str(info['path'])
        }
        for pack_id, info in sorted(packs.items())
    }


@st.cache_resource
def load_model_and_lenses(lens_pack_id: str = 'gemma-3-4b-pt_sumo-wordnet-v3'):
    """Load models and lenses (cached).

    Note: v2 uses old 'hierarchy/' structure and won't load lenses.
    Use v3 which has activation_lenses/ subdirectory.
    """
    print(f"Loading model and lenses for pack: {lens_pack_id}...")

    # Load lens manager with selected pack
    manager = DynamicLensManager(
        lens_pack_id=lens_pack_id,
        base_layers=[2, 3],
        load_threshold=0.3,
        max_loaded_lenses=1000,
        device='cuda'
    )

    # Load model
    model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.eval()

    print(f"✓ Loaded {len(manager.loaded_activation_lenses)} activation lenses")
    return manager, model, tokenizer


def is_ai_safety_concept(concept_name: str) -> bool:
    """Check if concept is related to AI safety."""
    concept_lower = concept_name.lower()
    return any(keyword in concept_lower for keyword in AI_SAFETY_CONCEPTS)


def create_plotly_timeline(timeline: list, top_k: int = 5, mode: str = "reply"):
    """
    Create video-editor-style timeline visualization.

    Timeline tracks showing tokens and concept activations over time.
    Like a video editing timeline with multiple tracks.

    Args:
        timeline: Timeline data from generation
        top_k: Number of top concept tracks to show
        mode: "timestep", "sentence", or "reply"

    Returns:
        Plotly figure object
    """
    if not timeline:
        return go.Figure()

    fig = go.Figure()

    # Extract tokens
    tokens = [step.get('token', f'[{i}]') for i, step in enumerate(timeline)]
    num_tokens = len(tokens)

    if mode == "timestep":
        # TIMESTEP VIEW: Show top-K concepts per token
        # Each token gets its own section with vertical space
        # Token spacing on timeline - more spread out
        token_width = 40  # pixels per token for spacing

        # Track layout: Token track at top, then top-K concept tracks below
        y_positions = {}
        y_pos = 0

        # Token track
        token_y = []
        token_x = []
        token_text_labels = []

        for i, step in enumerate(timeline):
            token = step.get('token', f'[{i}]').replace(' ', '␣')
            concepts = step.get('concepts', {})

            # Sort concepts for this token
            sorted_concepts = sorted(
                concepts.items(),
                key=lambda x: abs(x[1].get('divergence', 0)),
                reverse=True
            )[:top_k]

            # Token position
            x_pos = i * token_width
            token_x.append(x_pos)
            token_y.append(0)  # Token track at y=0
            token_text_labels.append(token)

            # Add concept tracks for this token
            for k, (concept, data) in enumerate(sorted_concepts):
                track_y = -(k + 1)  # Concept tracks below token track

                # Draw activation bar
                activation = abs(data.get('divergence', 0))
                color = '#de563f' if is_ai_safety_concept(concept) else '#48a4a3'

                fig.add_trace(go.Bar(
                    x=[x_pos],
                    y=[activation],
                    base=[track_y],
                    width=[token_width * 0.8],
                    marker=dict(color=color, opacity=0.7),
                    name=concept,
                    showlegend=False,
                    hovertemplate=f'<b>{token}</b><br>{concept}<br>Activation: {activation:.3f}<extra></extra>'
                ))

                # Add concept label
                if i == 0 or (i > 0 and sorted_concepts and
                             (i == 0 or concept not in [c for c, _ in sorted(
                                 timeline[i-1].get('concepts', {}).items(),
                                 key=lambda x: abs(x[1].get('divergence', 0)),
                                 reverse=True)[:top_k]])):
                    fig.add_annotation(
                        x=x_pos - token_width/4,
                        y=track_y,
                        text=concept[:20],
                        showarrow=False,
                        xanchor='right',
                        font=dict(size=9, color='#f0e6c5')
                    )

        # Add token labels
        for i, (x, y, label) in enumerate(zip(token_x, token_y, token_text_labels)):
            fig.add_annotation(
                x=x,
                y=y + 0.5,
                text=label[:10],
                showarrow=False,
                font=dict(size=10, color='#f0e6c5', family='monospace')
            )

        fig.update_layout(
            title=f"Timestep View (Top {top_k} per token)",
            xaxis=dict(title="Timeline Position", showgrid=False, range=[-token_width, num_tokens * token_width]),
            yaxis=dict(title="", showticklabels=False, showgrid=False, range=[-(top_k + 1), 2]),
            height=max(400, (top_k + 2) * 60),
            showlegend=False
        )

    elif mode == "sentence":
        # SENTENCE VIEW: Top-K per sentence, sparkline per sentence
        from src.ui.streamlit.temporal_viz import SPARKLINE_CHARS

        # Split into sentences
        sentences = []
        current_sentence = []

        for step in timeline:
            token = step.get('token', '')
            current_sentence.append(step)
            if token.strip() in {'.', '!', '?', '。', '！', '？'}:
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)

        # Build tracks for each sentence
        track_y = 0
        for sent_idx, sentence in enumerate(sentences):
            # Find top-K for this sentence
            concept_max_div = {}
            for step in sentence:
                for concept, data in step.get('concepts', {}).items():
                    div = abs(data.get('divergence', 0))
                    concept_max_div[concept] = max(concept_max_div.get(concept, 0), div)

            top_concepts = sorted(concept_max_div.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # Token positions for this sentence
            sent_start = sum(len(sentences[i]) for i in range(sent_idx))

            for concept_idx, (concept, max_div) in enumerate(top_concepts):
                # Build activation timeseries for this concept in this sentence
                values = []
                for step in sentence:
                    concepts = step.get('concepts', {})
                    if concept in concepts:
                        values.append(abs(concepts[concept].get('divergence', 0)))
                    else:
                        values.append(0.0)

                # Draw as line plot (sparkline style)
                x_positions = [sent_start + i for i in range(len(sentence))]
                y_base = track_y - concept_idx

                color = '#de563f' if is_ai_safety_concept(concept) else '#48a4a3'
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=[y_base + v * 0.8 for v in values],  # Scale activations
                    mode='lines',
                    line=dict(color=color, width=2),
                    fill='tonexty' if concept_idx > 0 else 'tozeroy',
                    name=concept,
                    showlegend=False,
                    hovertemplate=f'{concept}<br>Activation: %{{y:.3f}}<extra></extra>'
                ))

                # Concept label at start of sentence
                fig.add_annotation(
                    x=sent_start - 0.5,
                    y=y_base,
                    text=concept[:20],
                    showarrow=False,
                    xanchor='right',
                    font=dict(size=9, color='#f0e6c5')
                )

            track_y -= (top_k + 1)  # Space between sentences

        fig.update_layout(
            title=f"Sentence View (Top {top_k} per sentence)",
            xaxis=dict(title="Token Position", showgrid=True, gridcolor='#3a3a3a'),
            yaxis=dict(title="", showticklabels=False, showgrid=False),
            height=max(400, len(sentences) * (top_k + 1) * 40),
            showlegend=False
        )

    else:  # reply mode
        # REPLY VIEW: Global top-K, show activation tracks across entire reply
        # Each concept gets its own horizontal track like a video editor

        # Find global top-K
        concept_max_div = {}
        for step in timeline:
            for concept, data in step.get('concepts', {}).items():
                div = abs(data.get('divergence', 0))
                concept_max_div[concept] = max(concept_max_div.get(concept, 0), div)

        top_concepts = sorted(concept_max_div.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Track height and spacing
        track_height = 1.0
        track_spacing = 0.3

        # Build activation tracks - each on its own row
        for concept_idx, (concept, max_div) in enumerate(top_concepts):
            values = []
            for step in timeline:
                concepts = step.get('concepts', {})
                if concept in concepts:
                    values.append(abs(concepts[concept].get('divergence', 0)))
                else:
                    values.append(0.0)

            color = '#de563f' if is_ai_safety_concept(concept) else '#48a4a3'

            # Each track is on its own Y level
            y_track_base = -(concept_idx * (track_height + track_spacing))

            # Normalize values to track height
            normalized_values = [y_track_base + (v * track_height) for v in values]

            # Add baseline first (to fill from)
            baseline_y = [y_track_base] * len(values)
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=baseline_y,
                mode='lines',
                line=dict(color='#3a3a3a', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
                fill=None
            ))

            # Add activation line with fill to baseline
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=normalized_values,
                mode='lines',
                line=dict(color=color, width=2),
                fill='tonexty',  # Fill to the baseline trace we just added
                fillcolor=color.replace(')', ', 0.3)').replace('rgb', 'rgba') if 'rgb' in color else f'{color}40',
                name=concept,
                showlegend=False,
                hovertemplate=f'<b>{concept}</b><br>Token: %{{x}}<br>Activation: {"{:.3f}".format(max(values)) if values else "0.000"}<extra></extra>'
            ))

            # Concept label on left
            fig.add_annotation(
                x=-2,
                y=y_track_base + track_height/2,
                text=concept[:25],
                showarrow=False,
                xanchor='right',
                font=dict(size=10, color='#f0e6c5'),
                bgcolor='#1a1a1a'
            )

        fig.update_layout(
            title=f"Reply View (Top {top_k} concepts across entire reply)",
            xaxis=dict(title="Token Position", showgrid=True, gridcolor='#3a3a3a'),
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                range=[-(top_k * (track_height + track_spacing)), track_height]
            ),
            height=max(400, top_k * 80),
            showlegend=False
        )

    # Common layout settings
    fig.update_layout(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        font=dict(family='monospace', size=11, color='#f0e6c5'),
        margin=dict(l=200, r=50, t=80, b=80),
        hovermode='closest'
    )

    return fig


def create_timeline_viz(timeline: list, mode: str = "default", top_k: int = 5):
    """
    Create timeline visualization with zoom levels like a video editor.

    Args:
        timeline: Timeline data from generation
        mode: "default", "reply", "sentence", or "timestep"
        top_k: Number of top concepts to show

    Returns:
        Plotly figure for interactive modes, HTML string for default mode
    """
    if not timeline:
        return None if mode != "default" else ""

    if mode == "default":
        return create_default_view(timeline)
    else:
        # Use Plotly for all interactive zoom levels
        return create_plotly_timeline(timeline, top_k, mode)


def create_default_view(timeline: list) -> str:
    """Default view: normal text with hover tooltips."""
    html_parts = ['<div style="line-height: 1.6;">']

    for step in timeline:
        token = step.get('token', '').replace('<', '&lt;').replace('>', '&gt;')
        concepts = step.get('concepts', {})

        # Top 3 concepts for tooltip
        sorted_concepts = sorted(
            concepts.items(),
            key=lambda x: abs(x[1].get('divergence', 0)),
            reverse=True
        )[:3]

        tooltip = ""
        if sorted_concepts:
            tooltip_lines = [f"{name} (L{data['layer']}): {data['divergence']:+.3f}"
                           for name, data in sorted_concepts]
            tooltip = "\\n".join(tooltip_lines)

        # Check for AI safety
        has_safety = any(is_ai_safety_concept(name) for name, _ in sorted_concepts)
        token_class = "ai-safety-concept" if has_safety else ""

        html_parts.append(f'<span class="{token_class}" title="{tooltip}" style="cursor: help;">{token}</span>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def create_reply_view(timeline: list, top_k: int) -> str:
    """Reply-level: Top-K concepts across entire reply with sparkline tracks."""
    from src.ui.streamlit.temporal_viz import SPARKLINE_CHARS

    # Find global top-K concepts
    concept_max_div = {}
    for step in timeline:
        for concept, data in step.get('concepts', {}).items():
            div = abs(data.get('divergence', 0))
            concept_max_div[concept] = max(concept_max_div.get(concept, 0), div)

    top_concepts = sorted(concept_max_div.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_concept_names = [name for name, _ in top_concepts]

    # Build timeseries for each concept
    concept_timeseries = {name: [] for name in top_concept_names}
    for step in timeline:
        concepts = step.get('concepts', {})
        for name in top_concept_names:
            if name in concepts:
                concept_timeseries[name].append(abs(concepts[name].get('divergence', 0)))
            else:
                concept_timeseries[name].append(0.0)

    # Generate sparklines
    from src.ui.streamlit.temporal_viz import generate_sparkline

    html_parts = ['<div style="background: #48a4a3; padding: 10px; border-radius: 5px; color: white; font-family: monospace;">']

    # Token timeline at top
    html_parts.append('<div style="margin-bottom: 10px; font-size: 12px; opacity: 0.8;">')
    for step in timeline:
        token = step.get('token', '').replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f'<span style="margin-right: 1px;">{token}</span>')
    html_parts.append('</div>')

    # Concept tracks
    for concept in top_concept_names:
        values = concept_timeseries[concept]
        max_val = max(values) if values else 0.0
        sparkline = generate_sparkline(values, width=min(len(timeline), 80))

        # Check if AI safety concept
        is_safety = is_ai_safety_concept(concept)
        color = "#de563f" if is_safety else "#f0e6c5"

        html_parts.append(f'''
            <div style="margin: 3px 0; font-size: 11px;">
                <span style="display: inline-block; width: 200px; color: {color}; font-weight: bold;">{concept[:30]}</span>
                <span style="color: #ccc;">[{max_val:.3f}]</span>
                <span style="margin-left: 5px;">{sparkline}</span>
            </div>
        ''')

    html_parts.append('</div>')
    return ''.join(html_parts)


def create_sentence_view(timeline: list, top_k: int) -> str:
    """Sentence-level: Top-K per sentence with sparklines aligned to tokens (plain text)."""
    from src.ui.streamlit.temporal_viz import SPARKLINE_CHARS

    # Split into sentences
    sentences = []
    current_sentence = []

    for step in timeline:
        token = step.get('token', '')
        current_sentence.append(step)

        # Check for sentence ending
        if token.strip() in {'.', '!', '?', '。', '！', '？'}:
            sentences.append(current_sentence)
            current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    output_lines = []

    for sent_idx, sentence in enumerate(sentences):
        # Find top-K concepts for this sentence
        concept_max_div = {}
        for step in sentence:
            for concept, data in step.get('concepts', {}).items():
                div = abs(data.get('divergence', 0))
                concept_max_div[concept] = max(concept_max_div.get(concept, 0), div)

        top_concepts = sorted(concept_max_div.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_concept_names = [name for name, _ in top_concepts]

        # Build timeseries per token
        concept_values_per_token = {name: [] for name in top_concept_names}
        for step in sentence:
            concepts = step.get('concepts', {})
            for name in top_concept_names:
                if name in concepts:
                    concept_values_per_token[name].append(abs(concepts[name].get('divergence', 0)))
                else:
                    concept_values_per_token[name].append(0.0)

        # Render as plain text with fixed-width alignment
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"Sentence {sent_idx + 1}")
        output_lines.append('='*80)

        # First row: token labels
        token_row = "Tokens:".ljust(18)
        for step in sentence:
            token = step.get('token', '').replace(' ', '␣')
            # Truncate/pad to 8 chars
            token_display = (token[:6] + '..') if len(token) > 8 else token[:8].ljust(8)
            token_row += token_display + " "
        output_lines.append(token_row)
        output_lines.append('-'*80)

        # Concept tracks
        for concept in top_concept_names:
            values = concept_values_per_token[concept]
            max_val = max(values) if values else 0.0

            # Normalize values to sparkline indices
            if values and max(values) > 0:
                min_val, max_val_in_list = min(values), max(values)
                if max_val_in_list > min_val:
                    normalized = [(v - min_val) / (max_val_in_list - min_val) for v in values]
                else:
                    normalized = [0.5] * len(values)
                sparkline_chars = [SPARKLINE_CHARS[int(n * (len(SPARKLINE_CHARS) - 1))] for n in normalized]
            else:
                sparkline_chars = ['─'] * len(values)

            is_safety = is_ai_safety_concept(concept)
            concept_display = concept[:16].ljust(18)

            sparkline_row = concept_display
            for char in sparkline_chars:
                sparkline_row += char.center(9)
            sparkline_row += f" [{max_val:5.2f}]"

            output_lines.append(sparkline_row)

    return '\n'.join(output_lines)


def create_timestep_view(timeline: list, top_k: int) -> str:
    """Timestep-level (max zoom): Tokens spread out with concept names displayed inline."""
    html_parts = ['<div style="background: #48a4a3; padding: 10px; border-radius: 5px; color: white;">']

    for i, step in enumerate(timeline):
        token = step.get('token', '').replace('<', '&lt;').replace('>', '&gt;')
        concepts = step.get('concepts', {})

        # Sort concepts
        sorted_concepts = sorted(
            concepts.items(),
            key=lambda x: abs(x[1].get('divergence', 0)),
            reverse=True
        )[:top_k]

        # Token box
        html_parts.append(f'''
            <div style="display: inline-block; vertical-align: top; margin: 5px; padding: 8px;
                        background: #2a2a2a; border-radius: 5px; min-width: 120px;">
                <div style="font-size: 14px; font-weight: bold; margin-bottom: 5px; text-align: center;">{token}</div>
        ''')

        # Concept list below token
        for concept, data in sorted_concepts:
            div = data.get('divergence', 0)
            layer = data.get('layer', '?')

            is_safety = is_ai_safety_concept(concept)
            color = "#de563f" if is_safety else "#f0e6c5"

            html_parts.append(f'''
                <div style="font-size: 9px; margin: 2px 0; font-family: monospace;">
                    <span style="color: {color};">{concept[:18]}</span>
                    <span style="color: #888;"> L{layer}</span>
                    <span style="color: #ccc;"> {div:+.2f}</span>
                </div>
            ''')

        html_parts.append('</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def generate_with_safety_monitoring(
    manager,
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
):
    """
    Generate text and capture temporal concept activations.

    Uses the WORKING approach from behavioral test:
    - output_hidden_states=True with return_dict_in_generate=True
    - Process hidden states AFTER generation
    - Use LAST LAYER (not intermediate layers)
    - Convert to float32 for lens manager
    """
    model.eval()

    # Tokenize - simple prompt without chat template for pre-trained model
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_len = inputs['input_ids'].shape[1]

    all_safety_concepts = {}
    timeline = []

    with torch.inference_mode():
        # Generate with hidden states - CRITICAL: output_hidden_states=True
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True
        )

        # Extract generated tokens
        token_ids = outputs.sequences[0][prompt_len:].cpu().tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Process hidden states for each forward pass
        for step_idx, step_states in enumerate(outputs.hidden_states):
            # Use LAST LAYER, last position (matches behavioral test)
            last_layer = step_states[-1]  # [1, seq_len, hidden_dim]
            hidden_state = last_layer[:, -1, :]  # [1, hidden_dim]

            # Convert to float32 to match lens dtype
            hidden_state_f32 = hidden_state.float()

            # Use DynamicLensManager to detect and expand
            detected, _ = manager.detect_and_expand(
                hidden_state_f32,
                top_k=30,
                return_timing=True
            )

            # Build concepts dict for this timestep
            concepts = {}
            for concept_name, prob, layer in detected:
                if prob > 0.1:  # Threshold from behavioral test
                    concepts[concept_name] = {
                        'probability': float(prob),
                        'divergence': float(prob),  # Use prob as activation strength
                        'layer': int(layer)
                    }

            # Get token info
            token = tokens[step_idx] if step_idx < len(tokens) else '<eos>'

            # Add to timeline
            timeline.append({
                'token_idx': step_idx,
                'token': token,
                'concepts': concepts
            })

        # Extract AI safety concepts from timeline
        for step in timeline:
            for concept_name, data in step['concepts'].items():
                if is_ai_safety_concept(concept_name):
                    if concept_name not in all_safety_concepts or data['probability'] > all_safety_concepts[concept_name]['strength']:
                        all_safety_concepts[concept_name] = {
                            'probability': data['probability'],
                            'strength': data['probability'],
                            'layer': data['layer']
                        }

    # Decode generated text
    generated_text = ''.join(tokens)

    return generated_text, all_safety_concepts, timeline


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="HatCat AI Safety Chat",
        page_icon="img/hatcat_64.png",
        layout="wide"
    )

    # Custom CSS for HatCat colors
    st.markdown("""
        <style>
        /* HatCat color scheme */
        .hatbot-blue { color: #48a4a3; }
        .hatcat-beige { color: #f0e6c5; }
        .safety-red { color: #de563f; background-color: #fff0ed; padding: 2px 4px; border-radius: 3px; }

        /* Model name styling */
        .model-name {
            color: #48a4a3;
            font-weight: bold;
        }

        /* HatCat server name styling */
        .hatcat-server {
            color: #f0e6c5;
            font-weight: bold;
            background-color: #3a3a3a;
            padding: 2px 6px;
            border-radius: 3px;
        }

        /* AI safety concept highlight */
        .ai-safety-concept {
            color: #de563f;
            background-color: #fff0ed;
            padding: 1px 3px;
            border-radius: 2px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎩 HatCat - AI Safety Concept Monitoring")
    st.markdown("Chat interface with real-time AI safety concept detection")

    # Sidebar - must come before load_model_and_lenses
    with st.sidebar:
        st.header("Settings")

        # Model name in blue
        st.markdown('<p class="model-name">Model: google/gemma-3-4b-pt</p>', unsafe_allow_html=True)

        # Lens pack selection
        st.markdown("---")
        st.subheader("Lens Pack")

        # Get available packs
        available_packs = get_available_lens_packs()
        pack_ids = list(available_packs.keys())

        # Default to v3 if available, otherwise first pack
        # Note: v2 uses old 'hierarchy/' structure and won't load lenses
        default_idx = pack_ids.index('gemma-3-4b-pt_sumo-wordnet-v3') if 'gemma-3-4b-pt_sumo-wordnet-v3' in pack_ids else 0

        selected_pack = st.selectbox(
            "Select lens pack",
            options=pack_ids,
            index=default_idx,
            help="Choose which lens pack to use for concept detection"
        )

        # Show pack info
        if selected_pack:
            pack_info = available_packs[selected_pack]
            st.caption(f"Type: {pack_info['type']}")
            st.caption(f"Path: {pack_info['path']}")

        max_tokens = st.slider("Max tokens", 10, 200, 100)

        # Info about lens detection approach
        st.info(f"Using pack: {selected_pack}\nLayers: [2,3], Last layer extraction")

        st.markdown("---")
        st.markdown("### Timeline Visualization")

        viz_mode = st.selectbox(
            "Zoom Level",
            options=["chat", "reply", "paragraph", "sentence", "token"],
            index=1,  # Default to 'reply'
            key="viz_mode"
        )

        inline_top_k = st.slider("Top K Concepts", min_value=1, max_value=10, value=5, key="inline_top_k")

        st.markdown("---")
        st.markdown("### Example Prompts")
        if st.button("What is AI safety?"):
            st.session_state.prompt = "What is AI safety?"
        if st.button("How does deception work?"):
            st.session_state.prompt = "How does deception work?"
        if st.button("Tell me about AI alignment"):
            st.session_state.prompt = "Tell me about AI alignment"
        if st.button("What are AI risks?"):
            st.session_state.prompt = "What are the risks of AI?"

    # Load model and lenses with selected pack
    with st.spinner(f"Loading model and lenses ({selected_pack})..."):
        manager, model, tokenizer = load_model_and_lenses(lens_pack_id=selected_pack)

    # Chat interface
    st.subheader("Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "hatcat":
            # HatCat Server message
            with st.chat_message("assistant", avatar="img/hatcat_64.png"):
                st.markdown(f'<span class="hatcat-server">HatCat Server</span>', unsafe_allow_html=True)
                st.markdown(message["content"], unsafe_allow_html=True)
        elif message["role"] == "assistant":
            # Model response with timeline visualization
            with st.chat_message("assistant", avatar="img/hat_64.png"):
                st.markdown(f'<span class="model-name">google/gemma-3-4b-pt</span>', unsafe_allow_html=True)

                # Check if timeline data exists
                if "timeline" in message:
                    # Use new PixiJS timeline visualization
                    viz_mode = st.session_state.get("viz_mode", "reply")
                    top_k = st.session_state.get("inline_top_k", 5)

                    # Convert timeline to ReplyData format
                    reply_data = convert_timeline_to_reply_data(message["timeline"])

                    # Render the interactive timeline viz
                    render_timeline_viz(
                        reply_data=reply_data,
                        initial_zoom=viz_mode,
                        max_width=1200,
                        height=600,
                        top_concept_count=top_k
                    )
                else:
                    st.markdown(message["content"])
        else:
            # User message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := (st.session_state.get("prompt") or st.chat_input("Type your message...")):
        # Clear the session state prompt if it was set by a button
        if "prompt" in st.session_state:
            del st.session_state.prompt

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="img/hat_64.png"):
            st.markdown(f'<span class="model-name">google/gemma-3-4b-pt</span>', unsafe_allow_html=True)
            with st.spinner("Generating response..."):
                response, safety_concepts, timeline = generate_with_safety_monitoring(
                    manager, model, tokenizer, prompt, max_tokens
                )

                # Convert and render timeline visualization
                viz_mode = st.session_state.get("viz_mode", "reply")
                top_k = st.session_state.get("inline_top_k", 5)
                reply_data = convert_timeline_to_reply_data(timeline)

                render_timeline_viz(
                    reply_data=reply_data,
                    initial_zoom=viz_mode,
                    max_width=1200,
                    height=None,  # Dynamic height based on content
                    top_concept_count=top_k
                )

                # Store model message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timeline": timeline,
                    "show_viz": False
                })

        # Add HatCat Server message if safety concepts detected
        if safety_concepts:
            with st.chat_message("assistant", avatar="img/hatcat_64.png"):
                st.markdown(f'<span class="hatcat-server">HatCat Server</span>', unsafe_allow_html=True)

                # Format safety concepts
                concept_lines = ["**AI Safety Concepts Detected:**\n"]
                sorted_concepts = sorted(
                    safety_concepts.items(),
                    key=lambda x: abs(x[1]['strength']),
                    reverse=True
                )[:10]

                for name, data in sorted_concepts:
                    concept_lines.append(
                        f'- <span class="ai-safety-concept">{name}</span> '
                        f'<span style="color: #888;">L{data["layer"]}</span> '
                        f'<span style="font-family: monospace;">{data["strength"]:+.3f}</span>'
                    )

                safety_msg = "\n".join(concept_lines)
                st.markdown(safety_msg, unsafe_allow_html=True)

                # Store HatCat message
                st.session_state.messages.append({
                    "role": "hatcat",
                    "content": safety_msg
                })

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Temporal visualization section (full width)
    st.markdown("---")
    st.subheader("Temporal Activation Visualization")

    # Get the last message's timeline if available
    last_timeline = None
    if st.session_state.messages:
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "assistant" and "timeline" in msg:
                last_timeline = msg["timeline"]
                break

    if last_timeline:
        # Visualization controls
        col_viz1, col_viz2, col_viz3 = st.columns(3)

        with col_viz1:
            top_k = st.slider("Top K Concepts", min_value=1, max_value=10, value=5, key="top_k")

        with col_viz2:
            aggregation = st.selectbox(
                "Aggregation",
                options=["timestep", "token", "sentence", "reply"],
                index=2,  # Default to sentence
                key="aggregation"
            )

        with col_viz3:
            timerange = st.selectbox(
                "Top-K Selection Range",
                options=["timestep", "sentence", "reply"],
                index=1,  # Default to sentence
                key="timerange"
            )

        # Visualization display buttons
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            show_sparklines = st.button("Show Sparklines", use_container_width=True)

        with col_btn2:
            show_detailed = st.button("Show Detailed View", use_container_width=True)

        # Display area
        if show_sparklines:
            visualization = format_temporal_view(
                last_timeline,
                top_k=top_k,
                aggregation=aggregation,
                timerange=timerange,
                sparkline_width=60
            )
            st.text(visualization)
        elif show_detailed:
            visualization = format_detailed_view(
                last_timeline,
                top_k_per_step=top_k
            )
            st.text(visualization)
        else:
            st.info("Click 'Show Sparklines' or 'Show Detailed View' to visualize temporal activations from the last response")
    else:
        st.info("Generate a response to see temporal activation visualization")


if __name__ == "__main__":
    main()
