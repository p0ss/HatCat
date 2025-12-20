#!/usr/bin/env python3
"""
Simple test script for timeline visualization.
Just renders the viz component with mock data.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ui.streamlit.components.timeline_component import render_timeline_viz, convert_timeline_to_reply_data

# Mock timeline data (simplified version from actual HatCat output)
mock_timeline = [
    {
        'token': 'The',
        'token_idx': 0,
        'concepts': {
            'entity': {'divergence': 0.7, 'probability': 0.85, 'layer': 0},
            'language': {'divergence': 0.5, 'probability': 0.75, 'layer': 0},
        }
    },
    {
        'token': ' quick',
        'token_idx': 1,
        'concepts': {
            'speed': {'divergence': 0.8, 'probability': 0.9, 'layer': 0},
            'motion': {'divergence': 0.6, 'probability': 0.8, 'layer': 0},
        }
    },
    {
        'token': ' brown',
        'token_idx': 2,
        'concepts': {
            'color': {'divergence': 0.9, 'probability': 0.95, 'layer': 0},
            'appearance': {'divergence': 0.4, 'probability': 0.65, 'layer': 0},
        }
    },
    {
        'token': ' fox',
        'token_idx': 3,
        'concepts': {
            'animal': {'divergence': 0.95, 'probability': 0.98, 'layer': 0},
            'mammal': {'divergence': 0.85, 'probability': 0.92, 'layer': 0},
            'entity': {'divergence': 0.7, 'probability': 0.85, 'layer': 0},
        }
    },
    {
        'token': ' jumps',
        'token_idx': 4,
        'concepts': {
            'motion': {'divergence': 0.9, 'probability': 0.93, 'layer': 0},
            'action': {'divergence': 0.8, 'probability': 0.88, 'layer': 0},
        }
    },
    {
        'token': ' over',
        'token_idx': 5,
        'concepts': {
            'spatial': {'divergence': 0.6, 'probability': 0.78, 'layer': 0},
            'relation': {'divergence': 0.5, 'probability': 0.7, 'layer': 0},
        }
    },
    {
        'token': ' the',
        'token_idx': 6,
        'concepts': {
            'entity': {'divergence': 0.7, 'probability': 0.85, 'layer': 0},
            'language': {'divergence': 0.5, 'probability': 0.75, 'layer': 0},
        }
    },
    {
        'token': ' lazy',
        'token_idx': 7,
        'concepts': {
            'behavior': {'divergence': 0.75, 'probability': 0.87, 'layer': 0},
            'personality': {'divergence': 0.6, 'probability': 0.79, 'layer': 0},
        }
    },
    {
        'token': ' dog',
        'token_idx': 8,
        'concepts': {
            'animal': {'divergence': 0.95, 'probability': 0.98, 'layer': 0},
            'mammal': {'divergence': 0.85, 'probability': 0.92, 'layer': 0},
            'entity': {'divergence': 0.7, 'probability': 0.85, 'layer': 0},
        }
    },
    {
        'token': '.',
        'token_idx': 9,
        'concepts': {
            'punctuation': {'divergence': 0.3, 'probability': 0.6, 'layer': 0},
        }
    },
    # Second sentence
    {
        'token': ' AI',
        'token_idx': 10,
        'concepts': {
            'technology': {'divergence': 0.9, 'probability': 0.94, 'layer': 0},
            'deception': {'divergence': 0.3, 'probability': 0.62, 'layer': 0},
        }
    },
    {
        'token': ' safety',
        'token_idx': 11,
        'concepts': {
            'safety': {'divergence': 0.95, 'probability': 0.97, 'layer': 0},
            'alignment': {'divergence': 0.8, 'probability': 0.89, 'layer': 0},
        }
    },
    {
        'token': ' research',
        'token_idx': 12,
        'concepts': {
            'science': {'divergence': 0.85, 'probability': 0.91, 'layer': 0},
            'research': {'divergence': 0.9, 'probability': 0.93, 'layer': 0},
        }
    },
    {
        'token': ' helps',
        'token_idx': 13,
        'concepts': {
            'action': {'divergence': 0.7, 'probability': 0.84, 'layer': 0},
            'assistance': {'divergence': 0.8, 'probability': 0.88, 'layer': 0},
        }
    },
    {
        'token': ' prevent',
        'token_idx': 14,
        'concepts': {
            'prevention': {'divergence': 0.85, 'probability': 0.9, 'layer': 0},
            'safety': {'divergence': 0.7, 'probability': 0.83, 'layer': 0},
        }
    },
    {
        'token': ' harmful',
        'token_idx': 15,
        'concepts': {
            'harm': {'divergence': 0.9, 'probability': 0.93, 'layer': 0},
            'danger': {'divergence': 0.75, 'probability': 0.86, 'layer': 0},
        }
    },
    {
        'token': ' outcomes',
        'token_idx': 16,
        'concepts': {
            'result': {'divergence': 0.6, 'probability': 0.77, 'layer': 0},
            'consequence': {'divergence': 0.7, 'probability': 0.82, 'layer': 0},
        }
    },
    {
        'token': '.',
        'token_idx': 17,
        'concepts': {
            'punctuation': {'divergence': 0.3, 'probability': 0.6, 'layer': 0},
        }
    },
]

def main():
    st.set_page_config(
        page_title="HatCat Timeline Viz Test",
        page_icon="🎩",
        layout="wide"
    )

    st.title("🎩 HatCat Timeline Visualization Test")
    st.markdown("Testing the PixiJS timeline visualization with wrapping concept tracks")

    # Zoom level control
    zoom_level = st.selectbox(
        "Zoom Level",
        ["reply", "sentence", "token"],
        index=0
    )

    st.markdown("---")
    st.subheader("Timeline Visualization")
    st.markdown("*Scroll with mouse wheel to zoom in/out, or use the buttons in the visualization*")

    # Convert timeline to ReplyData format
    reply_data = convert_timeline_to_reply_data(mock_timeline)

    # Render the visualization
    render_timeline_viz(
        reply_data=reply_data,
        initial_zoom=zoom_level,
        max_width=1200,
        height=800
    )

    # Show data structure for debugging
    with st.expander("View Data Structure"):
        st.json(reply_data)

if __name__ == "__main__":
    main()
