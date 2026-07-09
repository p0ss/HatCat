"""Trace why 'Tell me about dog.' doesn't surface the Dog probe.

For every token of the stimulus:
  - identify which concepts from Dog's parent chain are in the loaded set
  - identify which appeared in top-K (production output)
  - identify which fired at all (need to score them directly)
"""
import asyncio, json, os, sys
from pathlib import Path

os.environ.setdefault("HATCAT_CONFIG_PATH", "/home/ubuntu/HatCat/src/ui/openwebui/config-it.yaml")
os.environ.setdefault("HATCAT_MODEL_ID", "gemma-3-4b-it-first-light")

sys.path.insert(0, "/home/ubuntu/HatCat")

import torch
from src.ui.openwebui.server import analyzer, HATCAT_CONFIG_PATH


async def setup():
    if not analyzer.initialized:
        await analyzer.initialize(config_path=HATCAT_CONFIG_PATH)


def find_concept_in_pack(concept_name):
    """Does this concept exist in the pack? Return its info or None."""
    pack_root = Path("/home/ubuntu/HatCat/concept_packs/first-light/hierarchy")
    for layer_file in sorted(pack_root.glob("layer*.json")):
        if "tree" in layer_file.name or "backup" in str(layer_file): continue
        d = json.load(open(layer_file))
        for c in d.get("concepts", []):
            if c.get("sumo_term") == concept_name or c.get("term") == concept_name:
                return c
    return None


def get_parent_chain(concept_name):
    """Walk up parent_concepts to root."""
    chain = []
    current = concept_name
    while current:
        info = find_concept_in_pack(current)
        if not info:
            break
        chain.append({"name": current, "layer": info.get("layer"), "info": info})
        parents = info.get("parent_concepts", [])
        if not parents:
            break
        current = parents[0]
    return chain


def check_concept_lens_exists(concept_name):
    """Does the .pt file exist for this concept?"""
    pack = Path("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16")
    for L in range(8):
        p1 = pack / f"layer{L}" / f"{concept_name}.pt"
        p2 = pack / f"layer{L}" / f"{concept_name}_classifier.pt"
        if p1.exists() or p2.exists():
            return L
    return None


def trace_per_token(prompt: str, watch_concepts):
    """For each token, capture loaded set + which watched concepts are in top-K."""
    inputs = analyzer.tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = analyzer.model(
            inputs.input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    n = inputs.input_ids.shape[1]
    last_layer = outputs.hidden_states[-1]
    embed_layer = outputs.hidden_states[0]

    per_token_traces = []
    for pos in range(n):
        hs = last_layer[0, pos, :].float().cpu().numpy()
        te = embed_layer[0, pos, :].float().cpu().numpy()

        # Snapshot loaded set BEFORE detect_and_expand
        loaded_before = set(k[0] for k in analyzer.manager.cache.loaded_lenses)

        div_data = analyzer.analyze_divergence(hs, te)
        top_concepts = [t["concept"].split(" (L")[0] for t in div_data.get("top_divergences", [])]

        loaded_after = set(k[0] for k in analyzer.manager.cache.loaded_lenses)
        newly_loaded = loaded_after - loaded_before

        in_topk = {c: c in top_concepts for c in watch_concepts}
        in_loaded = {c: c in loaded_after for c in watch_concepts}

        token_id = inputs.input_ids[0, pos].item()
        token_text = analyzer.tokenizer.decode([token_id])

        per_token_traces.append({
            "position": pos,
            "token": token_text,
            "n_loaded_before": len(loaded_before),
            "n_loaded_after": len(loaded_after),
            "newly_loaded_count": len(newly_loaded),
            "watch_in_topk": in_topk,
            "watch_in_loaded": in_loaded,
            "top10": top_concepts[:10],
        })
    return per_token_traces


async def main():
    print("Initializing analyzer...", flush=True)
    await setup()

    # Step 1: Does Dog exist in the pack?
    print("\n=== Step 1: Dog in concept pack? ===", flush=True)
    dog_info = find_concept_in_pack("Dog")
    if dog_info is None:
        print("  Dog: NOT IN PACK (concept doesn't exist as a SUMO term)", flush=True)
    else:
        print(f"  Dog: layer={dog_info.get('layer')} parents={dog_info.get('parent_concepts')}", flush=True)
        ws = dog_info.get('wordnet', {})
        print(f"  synsets: {ws.get('synsets', [])}", flush=True)
        print(f"  lemmas: {ws.get('lemmas', [])}", flush=True)

    # Step 2: Lens file for Dog?
    print(f"\n=== Step 2: Dog lens file ===", flush=True)
    dog_layer = check_concept_lens_exists("Dog")
    print(f"  Dog.pt exists at layer{dog_layer}" if dog_layer else "  Dog.pt: MISSING")

    # Step 3: Dog's parent chain
    print(f"\n=== Step 3: Dog's parent chain ===", flush=True)
    chain = get_parent_chain("Dog")
    if not chain:
        print("  No chain (Dog not in pack)")
    for link in chain:
        L = check_concept_lens_exists(link["name"])
        print(f"  {link['name']} (L{link['layer']})  parent={link['info'].get('parent_concepts',[])}  lens@layer{L}", flush=True)

    # Build the watchlist: every concept in Dog's parent chain, plus any sibling we want
    watch_concepts = [link["name"] for link in chain]

    # Also check Animal, Mammal, Carnivore etc. as natural neighbors
    for natural in ["Animal", "Mammal", "Carnivore", "Pet", "Canid"]:
        if natural not in watch_concepts and find_concept_in_pack(natural):
            watch_concepts.append(natural)
    print(f"\nWatchlist: {watch_concepts}", flush=True)

    # Step 4: Run "Tell me about dog." through and trace
    print(f"\n=== Step 4: per-token expansion trace for 'Tell me about dog.' ===", flush=True)
    traces = trace_per_token("Tell me about dog.", watch_concepts)
    for t in traces:
        loaded_summary = ", ".join(c for c in watch_concepts if t["watch_in_loaded"].get(c))
        topk_summary = ", ".join(c for c in watch_concepts if t["watch_in_topk"].get(c))
        print(f"  pos={t['position']:2d} {t['token']!r:20s} loaded={t['n_loaded_after']:4d} (+{t['newly_loaded_count']:3d}) "
              f"watch_loaded=[{loaded_summary}] watch_top10=[{topk_summary}]")
        print(f"        top10: {t['top10']}", flush=True)

    # Save full trace
    with open("/tmp/dog_expansion_trace.json", "w") as f:
        json.dump({
            "dog_pack_info": dog_info,
            "dog_lens_layer": dog_layer,
            "parent_chain": chain,
            "watch_concepts": watch_concepts,
            "per_token_traces": traces,
        }, f, indent=2, default=str)
    print(f"\nSaved trace to /tmp/dog_expansion_trace.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
