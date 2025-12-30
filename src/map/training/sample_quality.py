"""
Sample quality checking and saving for lens training data.

Quality checks help identify:
- Collapsed/repetitive outputs
- Too short/empty responses
- Off-topic or gibberish generations
- High-quality samples for CAT fine-tuning
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class SampleQuality:
    """Quality metrics for a single generated sample."""
    length: int  # Character count
    word_count: int
    unique_words: int
    repetition_ratio: float  # 1.0 = all unique, 0.0 = all repeated
    avg_word_length: float
    has_punctuation: bool
    starts_with_number: bool  # Often indicates list/counting collapse

    # Quality flags
    is_empty: bool
    is_too_short: bool  # < 10 chars
    is_collapsed: bool  # High repetition, number sequences
    is_low_diversity: bool  # < 50% unique words

    @property
    def is_good(self) -> bool:
        """Overall quality check."""
        return not (self.is_empty or self.is_too_short or
                    self.is_collapsed or self.is_low_diversity)

    @property
    def quality_score(self) -> float:
        """0-1 quality score."""
        score = 1.0
        if self.is_empty:
            return 0.0
        if self.is_too_short:
            score -= 0.5
        if self.is_collapsed:
            score -= 0.4
        if self.is_low_diversity:
            score -= 0.3
        return max(0.0, score)


def check_sample_quality(text: str, min_length: int = 10) -> SampleQuality:
    """
    Analyze quality of a generated text sample.

    Args:
        text: Generated text to check
        min_length: Minimum acceptable character length

    Returns:
        SampleQuality with metrics and flags
    """
    text = text.strip()

    # Basic metrics
    length = len(text)
    words = text.split()
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))

    # Repetition check
    if word_count > 0:
        repetition_ratio = unique_words / word_count
        avg_word_length = sum(len(w) for w in words) / word_count
    else:
        repetition_ratio = 0.0
        avg_word_length = 0.0

    # Pattern detection
    has_punctuation = bool(re.search(r'[.!?,;:]', text))
    starts_with_number = bool(re.match(r'^\s*\d', text))

    # Collapse detection - number sequences like "1, 2, 3, 4..."
    number_sequence = bool(re.search(r'\d+[,\s]+\d+[,\s]+\d+[,\s]+\d+', text))
    repeated_pattern = False
    if word_count >= 4:
        # Check for repeated n-grams
        for n in [2, 3]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = Counter(ngrams)
            if ngram_counts and max(ngram_counts.values()) >= 3:
                repeated_pattern = True
                break

    # Quality flags
    is_empty = length == 0
    is_too_short = length < min_length
    is_collapsed = number_sequence or repeated_pattern or (repetition_ratio < 0.3 and word_count > 5)
    is_low_diversity = repetition_ratio < 0.5 and word_count > 5

    return SampleQuality(
        length=length,
        word_count=word_count,
        unique_words=unique_words,
        repetition_ratio=repetition_ratio,
        avg_word_length=avg_word_length,
        has_punctuation=has_punctuation,
        starts_with_number=starts_with_number,
        is_empty=is_empty,
        is_too_short=is_too_short,
        is_collapsed=is_collapsed,
        is_low_diversity=is_low_diversity,
    )


def check_batch_quality(
    texts: List[str],
    min_length: int = 10,
) -> Tuple[List[SampleQuality], Dict[str, int]]:
    """
    Check quality of a batch of samples.

    Returns:
        (quality_results, summary_stats)
    """
    results = [check_sample_quality(t, min_length) for t in texts]

    summary = {
        'total': len(texts),
        'good': sum(1 for r in results if r.is_good),
        'empty': sum(1 for r in results if r.is_empty),
        'too_short': sum(1 for r in results if r.is_too_short),
        'collapsed': sum(1 for r in results if r.is_collapsed),
        'low_diversity': sum(1 for r in results if r.is_low_diversity),
        'avg_quality_score': sum(r.quality_score for r in results) / len(results) if results else 0.0,
    }

    return results, summary


@dataclass
class TrainingSample:
    """A single training sample for CAT fine-tuning."""
    prompt: str
    generated_text: str
    concept: str
    layer: int
    label: int  # 1 = positive, 0 = negative
    quality: SampleQuality
    timestamp: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d['quality'] = asdict(self.quality)
        # Add human-readable label
        d['sample_type'] = 'positive' if self.label == 1 else 'negative'
        return d


class SampleSaver:
    """Save training samples for later CAT fine-tuning.

    Saves incrementally - each batch of samples is written immediately to disk
    to avoid data loss on crashes and to handle large training runs.
    """

    def __init__(self, output_dir: Path, pack_name: str):
        self.output_dir = Path(output_dir) / "training_samples" / pack_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quality_stats: Dict[str, Dict[str, int]] = {}
        self.pack_name = pack_name
        self.total_samples = 0
        self.total_good = 0

    def add_samples(
        self,
        prompts: List[str],
        generated_texts: List[str],
        concept: str,
        layer: int,
        labels: List[int],
    ):
        """Add a batch of samples with quality checking - saves immediately to disk."""
        if len(generated_texts) != len(prompts):
            # Handle combined extraction mode where we get 2x activations
            # but only 1x texts - duplicate texts for both entries
            if len(generated_texts) * 2 == len(prompts):
                generated_texts = [t for t in generated_texts for _ in range(2)]
            else:
                print(f"    Warning: prompt/text count mismatch ({len(prompts)} vs {len(generated_texts)})")
                return

        timestamp = datetime.now().isoformat()

        # Ensure layer directory exists
        layer_dir = self.output_dir / f"layer{layer}"
        layer_dir.mkdir(exist_ok=True)

        # Build samples and save immediately (append mode)
        samples = []
        for prompt, text, label in zip(prompts, generated_texts, labels):
            quality = check_sample_quality(text)
            sample = TrainingSample(
                prompt=prompt,
                generated_text=text,
                concept=concept,
                layer=layer,
                label=label,
                quality=quality,
                timestamp=timestamp,
            )
            samples.append(sample)

        # Append to JSONL file immediately
        concept_file = layer_dir / f"{concept}.jsonl"
        with open(concept_file, 'a') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + '\n')

        # Update quality stats for this concept
        if concept not in self.quality_stats:
            self.quality_stats[concept] = {
                'total': 0, 'good': 0, 'bad': 0,
                'empty': 0, 'collapsed': 0, 'low_diversity': 0
            }

        stats = self.quality_stats[concept]
        for sample in samples:
            stats['total'] += 1
            self.total_samples += 1
            if sample.quality.is_good:
                stats['good'] += 1
                self.total_good += 1
            else:
                stats['bad'] += 1
            if sample.quality.is_empty:
                stats['empty'] += 1
            if sample.quality.is_collapsed:
                stats['collapsed'] += 1
            if sample.quality.is_low_diversity:
                stats['low_diversity'] += 1

    def save_layer(self, layer: int):
        """No-op - samples are now saved incrementally in add_samples()."""
        # Samples are saved immediately when added, so this is just for compatibility
        pass

    def save_all(self):
        """Save quality report (samples already saved incrementally)."""
        # Save quality report
        report = {
            'pack_name': self.pack_name,
            'total_samples': self.total_samples,
            'good_samples': self.total_good,
            'bad_samples': self.total_samples - self.total_good,
            'by_concept': self.quality_stats,
            'timestamp': datetime.now().isoformat(),
        }

        report_file = self.output_dir / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        good_rate = self.total_good / self.total_samples * 100 if self.total_samples > 0 else 0
        print(f"\n  Sample Quality Report:")
        print(f"    Total: {self.total_samples}")
        print(f"    Good: {self.total_good} ({good_rate:.1f}%)")
        print(f"    Bad: {self.total_samples - self.total_good}")
        print(f"    Saved to: {self.output_dir}")

        # Flag problematic concepts
        bad_concepts = [
            (c, s) for c, s in self.quality_stats.items()
            if s['total'] > 0 and s['good'] / s['total'] < 0.5
        ]
        if bad_concepts:
            print(f"\n  Warning: {len(bad_concepts)} concepts with <50% good samples:")
            for concept, stats in sorted(bad_concepts, key=lambda x: x[1]['good']/x[1]['total'])[:10]:
                rate = stats['good'] / stats['total'] * 100
                print(f"    - {concept}: {rate:.0f}% good ({stats['good']}/{stats['total']})")


def load_training_samples(
    samples_dir: Path,
    layers: Optional[List[int]] = None,
    concepts: Optional[List[str]] = None,
    only_good: bool = True,
) -> List[Dict]:
    """
    Load saved training samples for CAT fine-tuning.

    Args:
        samples_dir: Directory containing training_samples/<pack>/
        layers: Filter to specific layers (None = all)
        concepts: Filter to specific concepts (None = all)
        only_good: Only return samples that passed quality checks

    Returns:
        List of sample dicts
    """
    samples = []
    samples_dir = Path(samples_dir)

    # Find all layer directories
    layer_dirs = sorted(samples_dir.glob("layer*"))

    for layer_dir in layer_dirs:
        layer = int(layer_dir.name.replace("layer", ""))
        if layers is not None and layer not in layers:
            continue

        # Load all concept files
        for concept_file in layer_dir.glob("*.jsonl"):
            concept = concept_file.stem
            if concepts is not None and concept not in concepts:
                continue

            with open(concept_file) as f:
                for line in f:
                    sample = json.loads(line)
                    if only_good and not sample['quality']['is_good']:
                        continue
                    samples.append(sample)

    return samples
