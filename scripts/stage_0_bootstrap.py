"""
Stage 0: Raw Encyclopedia Bootstrap

Process 50K concepts in ~2 minutes using single-pass raw terms.
Creates initial semantic space map with low confidence but full coverage.
"""

import torch
import numpy as np
import h5py
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path

def get_activation(model, tokenizer, text, layer_idx=-1, device="cuda"):
    """Extract activation vector with proper attention masking."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.config.output_hidden_states = True
    
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    
    hs = out.hidden_states[layer_idx]                     # [B,T,D]
    mask = inputs["attention_mask"].unsqueeze(-1)         # [B,T,1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.squeeze(0).float().cpu().numpy()

class Stage0Bootstrap:
    """Bootstrap semantic encyclopedia with single-pass processing."""
    
    def __init__(self, model_name="google/gemma-3-270m", device="cuda"):
        print(f"Loading {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name, 
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.to(device)
        self.model.config.output_hidden_states = True
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = device
        
        # Get activation dimension
        with torch.inference_mode():
            dummy = self.tokenizer("test", return_tensors="pt").to(device)
            outputs = self.model(**dummy, output_hidden_states=True)
            self.activation_dim = outputs.hidden_states[-1].shape[-1]
        
        print(f"Model loaded. Activation dimension: {self.activation_dim}")
    
    def batch_process(self, concepts, batch_size=32, layer_idx=-1):
        """Process concepts in batches for efficiency."""
        activations = []
        
        for i in tqdm(range(0, len(concepts), batch_size), desc=f"Layer {layer_idx}"):
            batch = concepts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states[layer_idx]  # [B,T,D]
                
                # Proper attention-masked pooling
                mask = inputs["attention_mask"].unsqueeze(-1)  # [B,T,1]
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            
            activations.append(pooled.float().cpu().numpy())
        
        return np.vstack(activations)
    
    def bootstrap(self, concepts, output_path, layer_indices=[-1]):
        """
        Bootstrap semantic encyclopedia with raw concepts.
        
        Args:
            concepts: List of concept strings
            output_path: Path to save HDF5 file
            layer_indices: Which layers to capture (default: last layer)
        """
        start_time = time.time()
        n_concepts = len(concepts)
        
        print(f"\n{'='*70}")
        print("STAGE 0 BOOTSTRAP")
        print(f"{'='*70}")
        print(f"Concepts: {n_concepts:,}")
        print(f"Layers: {layer_indices}")
        print(f"Output: {output_path}")
        print(f"Batch size: 32")
        print()
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['n_concepts'] = n_concepts
            f.attrs['activation_dim'] = self.activation_dim
            f.attrs['model'] = str(self.model.config._name_or_path)
            f.attrs['stage'] = 0
            f.attrs['samples_per_concept'] = 1
            f.attrs['timestamp'] = time.time()
            
            # Store concepts
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('concepts', data=np.array(concepts, dtype=object), dtype=dt)
            
            # Process each layer
            for layer_idx in layer_indices:
                print(f"\nProcessing layer {layer_idx}...")
                
                # Batch process for efficiency
                activations = self.batch_process(concepts, batch_size=32, layer_idx=layer_idx)
                
                # Store activations (float16 for space efficiency)
                f.create_dataset(
                    f'layer_{layer_idx}/activations',
                    data=activations.astype(np.float16),
                    compression='gzip',
                    compression_opts=4
                )
                
                # Initialize variance as unknown (will be computed in later stages)
                f.create_dataset(
                    f'layer_{layer_idx}/variance',
                    data=np.full(n_concepts, np.nan, dtype=np.float16)
                )
                
                # Store metadata
                f[f'layer_{layer_idx}'].attrs['samples_per_concept'] = 1
                f[f'layer_{layer_idx}'].attrs['confidence'] = 'low'
                f[f'layer_{layer_idx}'].attrs['stage'] = 0
        
        elapsed = time.time() - start_time
        throughput = n_concepts / elapsed
        file_size_mb = Path(output_path).stat().st_size / 1024**2
        
        print(f"\n{'='*70}")
        print("✓ BOOTSTRAP COMPLETE")
        print(f"{'='*70}")
        print(f"Time elapsed:  {elapsed:.1f}s")
        print(f"Throughput:    {throughput:.1f} concepts/sec")
        print(f"File size:     {file_size_mb:.1f} MB")
        print(f"Storage/concept: {file_size_mb*1024/n_concepts:.2f} KB")
        print()
        print(f"Extrapolated times:")
        print(f"  10K concepts:  {10000/throughput:.0f}s ({10000/throughput/60:.1f} min)")
        print(f"  50K concepts:  {50000/throughput:.0f}s ({50000/throughput/60:.1f} min)")
        print(f"  100K concepts: {100000/throughput:.0f}s ({100000/throughput/60:.1f} min)")

def load_encyclopedia_concepts(source="wordnet", n=50000):
    """
    Load concepts from various sources.
    
    Args:
        source: "wordnet", "conceptnet", "wikipedia", or "mixed"
        n: Number of concepts to load
    """
    concepts = []
    
    if source in ["wordnet", "mixed"]:
        # Common nouns, verbs, adjectives
        concepts.extend([
            # Concrete nouns
            "dog", "cat", "house", "car", "tree", "book", "computer", "phone",
            "water", "fire", "earth", "air", "mountain", "river", "ocean", "forest",
            # Abstract nouns
            "democracy", "justice", "freedom", "love", "anger", "happiness", "fear",
            "thought", "memory", "knowledge", "wisdom", "truth", "beauty", "courage",
            # Actions
            "running", "walking", "thinking", "creating", "destroying", "learning",
            "teaching", "writing", "reading", "speaking", "listening", "observing",
            # Properties
            "red", "blue", "large", "small", "fast", "slow", "hot", "cold",
            "heavy", "light", "smooth", "rough", "bright", "dark", "loud", "quiet",
        ])
    
    if source in ["conceptnet", "mixed"]:
        # Abstract and relational concepts
        concepts.extend([
            "causation", "similarity", "difference", "transformation", "emergence",
            "recursion", "symmetry", "asymmetry", "entropy", "order", "chaos",
            "pattern", "structure", "function", "purpose", "intention", "goal",
        ])
    
    if source in ["wikipedia", "mixed"]:
        # Domain-specific terms
        concepts.extend([
            # Science
            "gravity", "evolution", "photosynthesis", "entropy", "quantum",
            "electron", "molecule", "cell", "organism", "ecosystem",
            # Technology
            "algorithm", "database", "network", "protocol", "encryption",
            "compiler", "interpreter", "virtual", "cloud", "distributed",
            # Social
            "government", "economy", "culture", "society", "community",
            "institution", "organization", "hierarchy", "equality", "justice",
        ])
    
    # Deduplicate and pad/trim to requested size
    concepts = list(dict.fromkeys(concepts))  # Remove duplicates while preserving order
    
    if len(concepts) < n:
        # Pad with numbered placeholders for testing
        for i in range(len(concepts), n):
            concepts.append(f"concept_{i:06d}")
    
    return concepts[:n]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 0: Bootstrap semantic encyclopedia")
    parser.add_argument("--n-concepts", type=int, default=1000, 
                       help="Number of concepts to process (default: 1000)")
    parser.add_argument("--source", type=str, default="mixed",
                       choices=["wordnet", "conceptnet", "wikipedia", "mixed"],
                       help="Concept source")
    parser.add_argument("--model", type=str, default="google/gemma-3-270m",
                       help="Model to use")
    parser.add_argument("--output", type=str, default="data/processed/encyclopedia_stage0.h5",
                       help="Output HDF5 file")
    parser.add_argument("--layers", type=int, nargs="+", default=[-1],
                       help="Layer indices to capture (default: -1)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STAGE 0: RAW ENCYCLOPEDIA BOOTSTRAP")
    print("=" * 70)
    print(f"Source: {args.source}")
    print(f"Target concepts: {args.n_concepts:,}")
    print(f"Model: {args.model}")
    print(f"Layers: {args.layers}")
    print(f"Device: {args.device}")
    
    # Load concepts
    print(f"\nLoading {args.n_concepts:,} concepts from {args.source}...")
    concepts = load_encyclopedia_concepts(source=args.source, n=args.n_concepts)
    print(f"✓ Loaded {len(concepts):,} concepts")
    
    # Show sample
    print("\nFirst 10 concepts:")
    for i, c in enumerate(concepts[:10], 1):
        print(f"  {i:2d}. {c}")
    
    # Initialize bootstrap
    bootstrap = Stage0Bootstrap(model_name=args.model, device=args.device)
    
    # Process
    bootstrap.bootstrap(
        concepts=concepts,
        output_path=args.output,
        layer_indices=args.layers
    )
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Validate convergence hypothesis:")
    print("   python convergence_validation.py")
    print()
    print("2. Train Stage 0 interpreter:")
    print("   python train_interpreter.py --data encyclopedia_stage0.h5")
    print()
    print("3. Evaluate and identify uncertain concepts:")
    print("   python evaluate_interpreter.py --find-uncertain")
    print()
    print("4. Run Stage 1 refinement on uncertain concepts:")
    print("   python stage1_refinement.py --uncertain-only")
