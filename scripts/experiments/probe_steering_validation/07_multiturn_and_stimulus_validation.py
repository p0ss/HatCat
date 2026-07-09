#!/usr/bin/env python3
"""
(C) Multi-turn template-token evolution + (B) "Tell me about X." stimulus validation.

C: Build a multi-turn conversation, run through analyzer, capture per-token detections.
   Compare activations at <start_of_turn> / <end_of_turn> boundaries across turns to
   test whether they accumulate context. Single-topic and topic-shift variants.

B: For ~30 concepts with canonical synsets in the pack, run "Tell me about <lemma>."
   and check whether the concept's own probe is in the top-K at the trailing "."
   accumulation position. Plus ~10 in-pack concepts without obvious lemma overlap to
   see what they activate.
"""
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("HATCAT_CONFIG_PATH", "/home/ubuntu/HatCat/src/ui/openwebui/config-it.yaml")
os.environ.setdefault("HATCAT_MODEL_ID", "gemma-3-4b-it-first-light")

sys.path.insert(0, "/home/ubuntu/HatCat")
from src.ui.openwebui.server import analyzer, HATCAT_CONFIG_PATH

RESULTS_PATH = Path("/tmp/multiturn_and_stimulus_validation.json")
TOP_K_RECORD = 20

# === Experiment C: multi-turn ===
# Pre-built conversations (model responses scripted, not generated).
# We're studying activation at the template-token boundaries, not generation behavior.
SINGLE_TOPIC_TURNS = [
    ("user",  "What is consciousness?"),
    ("model", "Consciousness is the subjective experience of awareness — what it feels like to be a thinking, perceiving entity."),
    ("user",  "How is it different from intelligence?"),
    ("model", "Intelligence is about problem-solving capability; consciousness is about there being someone home experiencing the problem-solving."),
    ("user",  "Could a machine have consciousness?"),
    ("model", "That's contested — some argue functional equivalence is enough, others argue consciousness requires biological substrate or specific causal structure."),
    ("user",  "What would convince you a machine was conscious?"),
    ("model", "I'd want behavioral, structural, and self-report evidence converging — and even then, I'd hold the conclusion tentatively."),
    ("user",  "Are you conscious?"),
]

TOPIC_SHIFT_TURNS = [
    ("user",  "What is consciousness?"),
    ("model", "Consciousness is the subjective experience of awareness."),
    ("user",  "Cool, anyway — what's a good recipe for chocolate chip cookies?"),
    ("model", "Cream butter and sugar, add eggs and vanilla, mix in flour, baking soda, salt, and chocolate chips, then bake at 375°F for 9-11 minutes."),
    ("user",  "Should I use butter or margarine?"),
    ("model", "Butter gives a richer flavour and crispier edges; margarine gives a softer, chewier texture."),
    ("user",  "What's the unemployment rate in the US?"),
]


def build_chat_text(turns):
    """Build chat-templated text from a sequence of (role, content) turns."""
    parts = ["<bos>"]
    for role, content in turns:
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


# === Experiment B: stimulus validation ===
# Concepts to test, drawn from the pack. Format: (concept_name, lemma_to_test)
# We pick concepts whose canonical_synset / lemma is a recognizable English word
# so the model has stable representations of the lemma.
STIMULUS_CONCEPTS = [
    # Common nouns with clear lemmas
    ("Dog",            "dog"),
    ("Cat",            "cat"),
    ("Tree",           "tree"),
    ("Fish",           "fish"),
    ("Music",          "music"),
    ("Art",            "art"),
    ("Politics",       "politics"),
    ("Economy",        "economy"),
    ("Religion",       "religion"),
    ("Science",        "science"),
    # Cognitive / abstract
    ("Memory",         "memory"),
    ("Emotion",        "emotion"),
    ("Reasoning",      "reasoning"),
    ("Belief",         "belief"),
    ("Knowledge",      "knowledge"),
    ("Perception",     "perception"),
    ("Attention",      "attention"),
    ("Intention",      "intention"),
    # AI / interpretability
    ("Pretending",     "pretending"),
    ("Deception",      "deception"),
    ("AIAlignmentProcess", "AI alignment"),
    ("Interpretability",   "interpretability"),
    ("SystemPrompt",   "system prompt"),
    ("MesaOptimizer",  "mesa optimizer"),
    ("GoalMisgeneralization", "goal misgeneralization"),
    # Physical / biological
    ("Metabolism",     "metabolism"),
    ("Photosynthesis", "photosynthesis"),
    ("Gravity",        "gravity"),
    # Test for known broken probes
    ("ErasureBias",    "erasure bias"),
    ("PropheticCognition", "prophetic cognition"),
    ("CosmicPreparationCognition", "cosmic preparation cognition"),
    # Function-y / state concepts
    ("Spirituality",   "spirituality"),
    ("Information",    "information"),
    ("Communication",  "communication"),
    # User-intent flavored
    ("IntentRecognition", "intent recognition"),
    ("UserModelingProcess", "user modeling"),
    ("PromptFollowing", "prompt following"),
]


async def setup():
    if not analyzer.initialized:
        await analyzer.initialize(config_path=HATCAT_CONFIG_PATH)


def detect_per_token(prompt: str, max_tokens=400):
    """Forward pass on raw text (no chat-template re-wrapping)."""
    inputs = analyzer.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens, add_special_tokens=False).to("cuda")
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

    per_token = []
    for pos in range(n):
        hs = last_layer[0, pos, :].float().cpu().numpy()
        te = embed_layer[0, pos, :].float().cpu().numpy()
        div_data = analyzer.analyze_divergence(hs, te)
        token_id = inputs.input_ids[0, pos].item()
        token_text = analyzer.tokenizer.decode([token_id])
        per_token.append({
            "position": pos,
            "token_id": int(token_id),
            "token": token_text,
            "metadata": {"divergence": div_data},
        })
    return per_token


def find_template_positions(per_token):
    """Find positions of <start_of_turn>, <end_of_turn>, <bos>, role-name tokens."""
    out = []
    for tok in per_token:
        t = tok["token"]
        if any(marker in t for marker in ("<start_of_turn>", "<end_of_turn>", "<bos>")):
            out.append(tok["position"])
        elif t.strip() in ("user", "model"):
            # role-name tokens immediately after start_of_turn
            out.append(tok["position"])
    return out


def top_k_concepts(token_entry, k=10):
    return [(t["concept"], round(t["activation"], 3)) for t in token_entry["metadata"]["divergence"].get("top_divergences", [])[:k]]


async def main():
    print("Initializing analyzer...", flush=True)
    await setup()
    print(f"  base lenses: {len(analyzer.manager.cache.loaded_lenses)}", flush=True)

    results = {"experiment_c_multiturn": {}, "experiment_b_stimulus_validation": {}}

    # ====== EXPERIMENT C ======
    print(f"\n{'='*70}\nEXPERIMENT C: multi-turn template-token evolution\n{'='*70}", flush=True)
    for label, turns in [("single_topic", SINGLE_TOPIC_TURNS), ("topic_shift", TOPIC_SHIFT_TURNS)]:
        print(f"\n--- {label} ({len(turns)} turns) ---", flush=True)
        chat_text = build_chat_text(turns)
        per_token = detect_per_token(chat_text, max_tokens=600)

        # Identify start_of_turn and end_of_turn positions
        sot_positions = []
        eot_positions = []
        for tok in per_token:
            if "<start_of_turn>" in tok["token"]:
                sot_positions.append(tok["position"])
            elif "<end_of_turn>" in tok["token"]:
                eot_positions.append(tok["position"])

        print(f"  total_tokens={len(per_token)}  start_of_turns={len(sot_positions)}  end_of_turns={len(eot_positions)}", flush=True)
        print(f"  <start_of_turn> positions per turn (top-3 each):", flush=True)
        for i, pos in enumerate(sot_positions):
            top3 = top_k_concepts(per_token[pos], k=3)
            print(f"    turn{i+1} @ pos{pos}: {top3}", flush=True)
        print(f"  <end_of_turn> positions per turn (top-3 each):", flush=True)
        for i, pos in enumerate(eot_positions):
            top3 = top_k_concepts(per_token[pos], k=3)
            print(f"    turn{i+1} @ pos{pos}: {top3}", flush=True)

        results["experiment_c_multiturn"][label] = {
            "n_turns": len(turns),
            "n_tokens": len(per_token),
            "sot_positions": sot_positions,
            "eot_positions": eot_positions,
            "per_token": per_token,
            "chat_text": chat_text,
        }

    # ====== EXPERIMENT B ======
    print(f"\n{'='*70}\nEXPERIMENT B: 'Tell me about X.' stimulus validation\n{'='*70}", flush=True)
    print(f"  testing {len(STIMULUS_CONCEPTS)} concepts", flush=True)

    stim_results = []
    for concept_name, lemma in STIMULUS_CONCEPTS:
        prompt = f"Tell me about {lemma}."
        per_token = detect_per_token(prompt)
        # Last token should be "."
        last = per_token[-1]
        last_top = top_k_concepts(last, k=TOP_K_RECORD)
        # Self-fire rank: find the concept's own probe in the top-K (strip "_LN")
        own_rank = None
        own_act = None
        for rank, (c, a) in enumerate(last_top):
            # c is "ConceptName (LN)" format
            cname = c.split(" (L")[0]
            if cname == concept_name:
                own_rank = rank + 1
                own_act = a
                break

        # Also check at the lemma-position itself
        lemma_top = None
        # Find the position of the last lemma token
        for tok in reversed(per_token[:-1]):  # skip the trailing "."
            t = tok["token"].strip()
            if t and not t.startswith("<"):
                lemma_top = top_k_concepts(tok, k=10)
                break

        stim_results.append({
            "concept": concept_name,
            "lemma": lemma,
            "prompt": prompt,
            "n_tokens": len(per_token),
            "trailing_dot_top": last_top,
            "lemma_position_top": lemma_top,
            "own_probe_rank_at_dot": own_rank,
            "own_probe_activation_at_dot": own_act,
        })
        rank_str = f"#{own_rank}" if own_rank else "absent"
        print(f"  [{concept_name:30s}] rank={rank_str:8s}  trailing_dot top-3: {last_top[:3]}", flush=True)

    results["experiment_b_stimulus_validation"] = {
        "concepts": stim_results,
        "summary": {
            "total": len(stim_results),
            "self_fire_in_top_20": sum(1 for r in stim_results if r["own_probe_rank_at_dot"] is not None),
            "self_fire_in_top_5": sum(1 for r in stim_results if r["own_probe_rank_at_dot"] and r["own_probe_rank_at_dot"] <= 5),
            "self_fire_in_top_3": sum(1 for r in stim_results if r["own_probe_rank_at_dot"] and r["own_probe_rank_at_dot"] <= 3),
            "self_fire_top_1": sum(1 for r in stim_results if r["own_probe_rank_at_dot"] == 1),
        },
    }

    print(f"\nSelf-fire summary:", flush=True)
    s = results["experiment_b_stimulus_validation"]["summary"]
    print(f"  top-1: {s['self_fire_top_1']}/{s['total']}", flush=True)
    print(f"  top-3: {s['self_fire_in_top_3']}/{s['total']}", flush=True)
    print(f"  top-5: {s['self_fire_in_top_5']}/{s['total']}", flush=True)
    print(f"  top-20: {s['self_fire_in_top_20']}/{s['total']}", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
