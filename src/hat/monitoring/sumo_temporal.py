"""Convenience entry points for SUMO temporal concept monitoring."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .monitor import SUMOTemporalMonitor, load_sumo_classifiers


def run_temporal_detection(
    prompt: str,
    model_name: str,
    layers: Iterable[int],
    device: str = "cuda",
    max_new_tokens: int = 50,
    top_k: int = 10,
    threshold: float = 0.3,
    show_token_details: bool = True,
    output_json: Optional[Path] = None,
    *,
    model=None,
    tokenizer=None,
    monitor: Optional[SUMOTemporalMonitor] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    print_report: bool = True,
    enable_dissonance: bool = False,
    dissonance_alpha: float = 0.5,
) -> dict:
    """Run temporal monitoring end-to-end and optionally save results."""

    generation_kwargs = generation_kwargs or {}

    loaded_model = model is None or tokenizer is None
    if loaded_model:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if monitor is None:
        classifiers, _ = load_sumo_classifiers(layers=list(layers), device=device)
        monitor = SUMOTemporalMonitor(
            classifiers,
            top_k=top_k,
            threshold=threshold,
            enable_dissonance=enable_dissonance,
            dissonance_alpha=dissonance_alpha,
        )

    result = monitor.monitor_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        **generation_kwargs,
    )

    if print_report:
        monitor.print_report(result, show_token_details=show_token_details)

    if output_json is not None:
        monitor.save_json(result, output_json)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SUMO Temporal Concept Detection - Monitor concepts during generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me about artificial intelligence and machine learning",
        help="Generation prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Model name (default: gemma-3-4b-pt)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Which classifier layers to load (default: 0 1 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show top K concepts per timestep (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Probability threshold for display (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )
    parser.add_argument(
        "--no-token-details",
        action="store_true",
        help="Skip token-by-token printout",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable sampling (use greedy decode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress textual report output",
    )
    parser.add_argument(
        "--dissonance",
        action="store_true",
        help="Enable semantic dissonance measurement (requires spacy/nltk)",
    )
    parser.add_argument(
        "--dissonance-alpha",
        type=float,
        default=0.5,
        help="Dissonance decay parameter (default: 0.5)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": not args.deterministic,
    }

    run_temporal_detection(
        prompt=args.prompt,
        model_name=args.model,
        layers=args.layers,
        device=args.device,
        max_new_tokens=args.max_tokens,
        top_k=args.top_k,
        threshold=args.threshold,
        show_token_details=not args.no_token_details,
        output_json=args.output_json,
        generation_kwargs=generation_kwargs,
        print_report=not args.quiet,
        enable_dissonance=args.dissonance,
        dissonance_alpha=args.dissonance_alpha,
    )


__all__ = ["run_temporal_detection", "build_arg_parser", "main"]
