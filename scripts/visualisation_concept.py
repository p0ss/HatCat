import networkx as nx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ConceptVisualizer:
    def __init__(self, concept_graph, image_size=1024):
        self.concept_graph = concept_graph
        self.image_size = image_size
        
        # Compute layout once (expensive, cache it)
        self.concept_positions = self._compute_layout()
        
        # Create base image (black background, white concept map)
        self.base_image = self._create_base_image()
    
    def _compute_layout(self):
        """Compute 2D positions for all concepts using force-directed layout."""
        
        G = nx.Graph()
        for concept, data in self.concept_graph.items():
            for related, weight in data['related_weighted'].items():
                G.add_edge(concept, related, weight=weight)
        
        # Force-directed layout (spring layout)
        # High-degree nodes naturally cluster at center
        pos = nx.spring_layout(
            G, 
            k=1/np.sqrt(len(G.nodes())),  # Optimal distance
            iterations=50,
            weight='weight'
        )
        
        # Scale to image coordinates
        positions = {}
        for concept, (x, y) in pos.items():
            # Map from [-1, 1] to [0, image_size]
            px = int((x + 1) * self.image_size / 2)
            py = int((y + 1) * self.image_size / 2)
            positions[concept] = (px, py)
        
        return positions
    
    def _create_base_image(self):
        """Create base image with white concept pixels on black background."""
        
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Draw white pixel for each concept
        for concept, (x, y) in self.concept_positions.items():
            img[y, x] = [255, 255, 255]  # White
        
        return img
    
    def render_activation(self, concept_activations):
        """Render activation overlay for current temporal window.
        
        Args:
            concept_activations: Dict[str, float] - {concept: probability}
        
        Returns:
            PIL Image with activation overlay
        """
        
        # Start with base image
        img = self.base_image.copy()
        
        # Overlay green activations
        for concept, activation in concept_activations.items():
            if concept not in self.concept_positions:
                continue
            
            x, y = self.concept_positions[concept]
            
            # Green intensity based on activation strength
            green_value = int(255 * activation)
            
            # Blend with base (white becomes green)
            img[y, x] = [0, green_value, 0]
        
        return Image.fromarray(img)
    
    def render_temporal_sequence(self, timeline):
        """Render sequence of activations over time.
        
        Args:
            timeline: List[Dict[str, float]] - activation per window
        
        Returns:
            List of PIL Images (one per window)
        """
        
        frames = []
        for window_activations in timeline:
            frame = self.render_activation(window_activations)
            frames.append(frame)
        
        return frames
    
    def save_animation(self, timeline, output_path, fps=5):
        """Save as animated GIF."""
        
        frames = self.render_temporal_sequence(timeline)
        
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )
        
        print(f"Saved animation: {output_path}")

# Initialize visualizer (do once)
# viz = ConceptVisualizer(concept_graph, image_size=1024)

# During generation
# timeline = detect_concepts_sliding_window(activations, classifiers)

# Render each window
# for i, window in enumerate(timeline):
#     frame = viz.render_activation(window['concepts'])
#     frame.save(f"frame_{i:04d}.png")

# Or save as animation
# viz.save_animation(timeline, "semantic_fmri.gif", fps=5)


# pythondef compute_coverage(all_observed_activations):
#     """What portion of model did we observe through concepts?"""
#     
#     # Total activation volume across all concepts
#     total_concept_activation = sum(all_observed_activations.values())
#     
#     # Theoretical maximum (all weights fully activated)
#     total_model_weights = model.num_parameters()
#     
#     # Observed coverage
#     coverage = total_concept_activation / total_model_weights
#     
#     return coverage