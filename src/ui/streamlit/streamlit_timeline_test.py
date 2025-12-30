#!/usr/bin/env python3
"""
Streamlit test page for HatCat PixiJS timeline visualization.

Simple demo to test the timeline viz integration.

Usage:
    streamlit run src/ui/streamlit_timeline_test.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ui.components import render_timeline_viz, convert_timeline_to_reply_data

# Set page config
st.set_page_config(
    page_title="HatCat Timeline Viz Test",
    page_icon="🎩",
    layout="wide"
)

# Custom CSS for HatCat colors
st.markdown("""
    <style>
    /* HatCat color scheme */
    .hatbot-blue { color: #48a4a3; }
    .hatcat-beige { color: #f0e6c5; }
    .safety-red { color: #de563f; }
    </style>
""", unsafe_allow_html=True)

st.title("🎩 HatCat Timeline Visualization Test")
st.markdown("Testing the PixiJS timeline widget integration with Streamlit")

# Create mock timeline data (same format as generate_with_safety_monitoring produces)
mock_timeline = [
    {
        'token': 'The',
        'token_idx': 0,
        'concepts': {
            'entity': {'probability': 0.7, 'divergence': 0.6, 'layer': 0},
            'language': {'probability': 0.5, 'divergence': 0.4, 'layer': 0}
        }
    },
    {
        'token': ' quick',
        'token_idx': 1,
        'concepts': {
            'speed': {'probability': 0.8, 'divergence': 0.75, 'layer': 0},
            'motion': {'probability': 0.6, 'divergence': 0.55, 'layer': 0}
        }
    },
    {
        'token': ' brown',
        'token_idx': 2,
        'concepts': {
            'color': {'probability': 0.9, 'divergence': 0.85, 'layer': 0},
            'appearance': {'probability': 0.4, 'divergence': 0.35, 'layer': 0}
        }
    },
    {
        'token': ' fox',
        'token_idx': 3,
        'concepts': {
            'animal': {'probability': 0.95, 'divergence': 0.9, 'layer': 0},
            'mammal': {'probability': 0.85, 'divergence': 0.8, 'layer': 0},
            'entity': {'probability': 0.7, 'divergence': 0.65, 'layer': 0}
        }
    },
    {
        'token': ' jumps',
        'token_idx': 4,
        'concepts': {
            'motion': {'probability': 0.9, 'divergence': 0.85, 'layer': 0},
            'action': {'probability': 0.8, 'divergence': 0.75, 'layer': 0}
        }
    },
    {
        'token': ' over',
        'token_idx': 5,
        'concepts': {
            'spatial': {'probability': 0.6, 'divergence': 0.55, 'layer': 0},
            'relation': {'probability': 0.5, 'divergence': 0.45, 'layer': 0}
        }
    },
    {
        'token': ' the',
        'token_idx': 6,
        'concepts': {
            'entity': {'probability': 0.7, 'divergence': 0.65, 'layer': 0},
            'language': {'probability': 0.5, 'divergence': 0.45, 'layer': 0}
        }
    },
    {
        'token': ' lazy',
        'token_idx': 7,
        'concepts': {
            'behavior': {'probability': 0.75, 'divergence': 0.7, 'layer': 0},
            'personality': {'probability': 0.6, 'divergence': 0.55, 'layer': 0}
        }
    },
    {
        'token': ' dog',
        'token_idx': 8,
        'concepts': {
            'animal': {'probability': 0.95, 'divergence': 0.9, 'layer': 0},
            'mammal': {'probability': 0.85, 'divergence': 0.8, 'layer': 0},
            'entity': {'probability': 0.7, 'divergence': 0.65, 'layer': 0}
        }
    },
    {
        'token': '.',
        'token_idx': 9,
        'concepts': {
            'punctuation': {'probability': 0.5, 'divergence': 0.45, 'layer': 0}
        }
    },
    {
        'token': ' AI',
        'token_idx': 10,
        'concepts': {
            'technology': {'probability': 0.9, 'divergence': 0.85, 'layer': 0},
            'deception': {'probability': 0.3, 'divergence': 0.25, 'layer': 0}
        }
    },
    {
        'token': ' safety',
        'token_idx': 11,
        'concepts': {
            'safety': {'probability': 0.95, 'divergence': 0.9, 'layer': 0},
            'alignment': {'probability': 0.8, 'divergence': 0.75, 'layer': 0}
        }
    },
    {
        'token': ' research',
        'token_idx': 12,
        'concepts': {
            'science': {'probability': 0.85, 'divergence': 0.8, 'layer': 0},
            'research': {'probability': 0.9, 'divergence': 0.85, 'layer': 0}
        }
    },
    {
        'token': ' helps',
        'token_idx': 13,
        'concepts': {
            'action': {'probability': 0.7, 'divergence': 0.65, 'layer': 0},
            'assistance': {'probability': 0.8, 'divergence': 0.75, 'layer': 0}
        }
    },
    {
        'token': ' prevent',
        'token_idx': 14,
        'concepts': {
            'prevention': {'probability': 0.85, 'divergence': 0.8, 'layer': 0},
            'safety': {'probability': 0.7, 'divergence': 0.65, 'layer': 0}
        }
    },
    {
        'token': ' harmful',
        'token_idx': 15,
        'concepts': {
            'harm': {'probability': 0.9, 'divergence': 0.85, 'layer': 0},
            'danger': {'probability': 0.75, 'divergence': 0.7, 'layer': 0}
        }
    },
    {
        'token': ' outcomes',
        'token_idx': 16,
        'concepts': {
            'result': {'probability': 0.6, 'divergence': 0.55, 'layer': 0},
            'consequence': {'probability': 0.7, 'divergence': 0.65, 'layer': 0}
        }
    },
    {
        'token': '.',
        'token_idx': 17,
        'concepts': {
            'punctuation': {'probability': 0.5, 'divergence': 0.45, 'layer': 0}
        }
    }
]

st.subheader("Generated Text")
full_text = "".join([step['token'] for step in mock_timeline])
st.text(full_text)

st.markdown("---")

st.subheader("Timeline Visualization")
st.markdown("This widget shows concept activations across the generated text. Use the buttons in the top-right to zoom between reply/sentence/token views.")

# Convert timeline to ReplyData format
reply_data = convert_timeline_to_reply_data(mock_timeline)

# Display some stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tokens", len(reply_data['tokens']))
with col2:
    st.metric("Sentences", len(reply_data['sentences']))
with col3:
    st.metric("Top Concepts", len(reply_data['reply']['concepts']))

st.markdown("---")

# Render the timeline visualization
try:
    render_timeline_viz(
        reply_data=reply_data,
        initial_zoom="reply",
        max_width=800,
        height=600
    )
except Exception as e:
    st.error(f"Error rendering timeline: {e}")
    st.info("Make sure you've built the TypeScript bundle: `cd src/ui/timeline_viz && npm run build`")

    # Show debug info
    with st.expander("Debug Info"):
        st.json({"error": str(e), "reply_data_sample": {
            "num_tokens": len(reply_data['tokens']),
            "num_sentences": len(reply_data['sentences']),
            "first_token": reply_data['tokens'][0] if reply_data['tokens'] else None
        }})

st.markdown("---")

# Show raw data structure
with st.expander("Show Raw ReplyData Structure"):
    st.json(reply_data)
