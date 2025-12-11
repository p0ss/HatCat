#!/usr/bin/env python3
"""
Dual Adaptive Trainer - Independent graduation for activation and text lenses.

Each lens type trains until it reaches target accuracy, with separate:
- Baseline sample counts
- Increment sizes
- Maximum sample limits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import time
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


class DualAdaptiveTrainer:
    """
    Train both activation and text lenses with independent adaptive cycles.

    Each lens graduates independently when reaching target accuracy.
    """

    def __init__(
        self,
        # Shared config
        max_iterations: int = 50,

        # Activation lens config
        activation_target_accuracy: float = None,  # Graduation F1 - defaults to validation_threshold
        activation_initial_samples: int = 40,  # Start with 40 samples (2 errors = 95%)
        activation_first_increment: int = 40,  # Add 40 more if initial fails (80 total)
        activation_subsequent_increment: int = 40,  # Add 40 more for each subsequent cycle (120, 160, ...)
        activation_max_samples: int = 200,

        # Text lens config
        text_target_accuracy: float = 0.80,  # Lower target for text lenses
        text_initial_samples: int = 10,
        text_first_increment: int = 20,
        text_subsequent_increment: int = 30,
        text_max_samples: int = 200,

        # Model for generating responses (required for text lenses)
        model=None,
        tokenizer=None,
        max_response_tokens: int = 100,

        # Validation config
        validate_lenses: bool = True,
        validation_mode: str = 'falloff',  # 'loose', 'falloff', or 'strict'
        validation_threshold: float = 0.85,  # Graduation F1 target (also used for calibration in strict mode)
        validation_layer_idx: int = 15,  # Model layer for validation activations

        # Tiered validation thresholds (progressive strictness for 'falloff' mode)
        validation_tier1_iterations: int = 3,  # Strict tier (push for A)
        validation_tier2_iterations: int = 6,  # High tier (accept B+)
        validation_tier3_iterations: int = 9,  # Medium tier (accept B)
        validation_tier4_iterations: int = 12,  # Relaxed tier (accept C+, prevent long tail)

        # Flags
        train_activation: bool = True,
        train_text: bool = True,
    ):
        """
        Args:
            max_iterations: Safety limit on adaptive cycles
            activation_target_accuracy: Graduation F1 threshold for activation lenses.
                                       Defaults to validation_threshold if not specified.
            activation_initial_samples: Initial sample count (default 40)
            activation_first_increment: Samples added on first failure (default 40)
            activation_subsequent_increment: Samples added per subsequent failure (default 40)
            activation_max_samples: Max samples for activation
            text_target_accuracy: Graduation threshold for text lenses (typically lower than activation)
            text_initial_samples: Initial sample count for text lenses
            text_first_increment: Samples added on first text lens failure
            text_subsequent_increment: Samples added per subsequent text lens failure
            text_max_samples: Max samples for text
            model: Language model for generating responses (required for text lenses and validation)
            tokenizer: Tokenizer for the model
            max_response_tokens: Max tokens to generate per prompt
            validate_lenses: Whether to validate lens calibration after graduation
            validation_mode: Validation blocking mode:
                - 'loose': Validate and grade but never block (advisory only)
                - 'falloff': Tiered blocking - strict early, relaxed later (default)
                - 'strict': Always block on validation failure until max_iterations
            validation_threshold: Graduation F1 target (default 0.85). Also controls calibration
                                 in strict mode. Set via --min-f1 command line option.
            validation_layer_idx: Model layer to extract activations from for validation
            validation_tier1_iterations: Max iterations for strict tier (A-grade, falloff mode)
            validation_tier2_iterations: Max iterations for high tier (B+-grade, falloff mode)
            validation_tier3_iterations: Max iterations for medium tier (B-grade, falloff mode)
            validation_tier4_iterations: Max iterations for relaxed tier (C+-grade, falloff mode)
            train_activation: Whether to train activation lens
            train_text: Whether to train text lens
        """
        self.max_iterations = max_iterations

        # Link graduation F1 to validation_threshold if not explicitly set
        self.validation_threshold = validation_threshold
        self.activation_target_accuracy = activation_target_accuracy if activation_target_accuracy is not None else validation_threshold
        self.activation_initial_samples = activation_initial_samples
        self.activation_first_increment = activation_first_increment
        self.activation_subsequent_increment = activation_subsequent_increment
        self.activation_max_samples = activation_max_samples

        self.text_target_accuracy = text_target_accuracy
        self.text_initial_samples = text_initial_samples
        self.text_first_increment = text_first_increment
        self.text_subsequent_increment = text_subsequent_increment
        self.text_max_samples = text_max_samples

        self.model = model
        self.tokenizer = tokenizer
        self.max_response_tokens = max_response_tokens

        self.validate_lenses = validate_lenses
        self.validation_mode = validation_mode
        self.validation_layer_idx = validation_layer_idx
        self.validation_tier1_iterations = validation_tier1_iterations
        self.validation_tier2_iterations = validation_tier2_iterations
        self.validation_tier3_iterations = validation_tier3_iterations
        self.validation_tier4_iterations = validation_tier4_iterations

        # Validate mode
        if validation_mode not in ['loose', 'falloff', 'strict']:
            raise ValueError(f"validation_mode must be 'loose', 'falloff', or 'strict', got '{validation_mode}'")

        self.train_activation = train_activation
        self.train_text = train_text

    def _grade_meets_target(self, grade: str, target: str) -> bool:
        """Check if achieved grade meets or exceeds target grade."""
        grade_values = {'A': 3, 'B': 2, 'C': 1, 'F': 0}
        # Handle B+ as between B and A
        achieved_value = grade_values.get(grade, 0)
        target_value = grade_values.get(target[0], 0)  # Take first char for 'B+'
        # For B+, require at least B (value 2)
        if target == 'B+':
            return achieved_value >= 2
        return achieved_value >= target_value

    def _get_validation_tier(self, cycle: int) -> Dict:
        """
        Determine validation tier based on which validation cycle we're in (for falloff mode).

        Cycle 0: First graduation (10 samples) - Strict tier (need A-grade)
        Cycle 1: Second graduation (30 samples) - High tier (accept B+)
        Cycle 2: Third graduation (60 samples) - Medium tier (accept B)
        Cycle 3+: Fourth+ graduation (90+ samples) - Relaxed tier (accept C+)

        Returns dict with tier info:
        - tier: 'strict', 'high', 'medium', or 'relaxed'
        - min_score: Minimum calibration score to pass
        - target_grade: Target quality grade
        - max_target_rank: Max acceptable rank for target domain
        - min_other_rank: Min acceptable average rank for other domains
        """
        if cycle == 0:
            return {
                'tier': 'strict',
                'min_score': 0.70,
                'target_grade': 'A',
                'max_target_rank': 3,
                'min_other_rank': 10.0,
            }
        elif cycle == 1:
            return {
                'tier': 'high',
                'min_score': 0.60,
                'target_grade': 'B+',
                'max_target_rank': 5,
                'min_other_rank': 8.0,
            }
        elif cycle == 2:
            return {
                'tier': 'medium',
                'min_score': 0.50,
                'target_grade': 'B',
                'max_target_rank': 7,
                'min_other_rank': 7.0,
            }
        else:  # cycle >= 3
            return {
                'tier': 'relaxed',
                'min_score': 0.40,
                'target_grade': 'C+',
                'max_target_rank': 10,
                'min_other_rank': 5.0,
            }

    def _generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate model responses for a list of prompts.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated responses (decoded text)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for text lens training")

        responses = []
        self.model.eval()

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_response_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the generated part (skip the input prompt)
                generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response)

        return responses

    def train_concept_incremental(
        self,
        concept_name: str,
        generation_config: Dict,
        test_prompts: List[str],
        test_labels: np.ndarray,
    ) -> Dict:
        """
        Train concept with incremental sample generation per cycle.

        Args:
            concept_name: Name of concept
            generation_config: Configuration for generating samples, includes:
                - concept: SUMO concept dict
                - all_concepts: Full concept map
                - negative_pool: Negative concept pool
                - test_negative_pool: Test negative pool
                - model: Language model
                - tokenizer: Tokenizer
                - device: Device for model
            test_prompts: Pre-generated test prompts
            test_labels: Test labels

        Returns:
            Dict with training results
        """
        from .sumo_data_generation import create_sumo_training_dataset
        from .sumo_classifiers import extract_activations, train_simple_classifier

        concept_start_time = time.time()
        start_time = time.time()

        # State tracking
        iteration = 0
        validation_cycle = 0
        cycle_iterations = 0  # Iterations within current cycle
        activation_graduated = False
        activation_classifier = None
        activation_results = None

        # Accumulated training data
        all_train_prompts = []
        all_train_labels = []
        all_train_activations = None

        # Extract test activations once upfront
        print(f"\n  Generating and extracting test samples...")
        gen_start = time.time()
        X_test = extract_activations(
            generation_config['model'],
            generation_config['tokenizer'],
            test_prompts,
            generation_config['device']
        )
        gen_time = time.time() - gen_start
        print(f"  ✓ Test set ready: {len(test_prompts)} samples [{gen_time:.1f}s]")

        # If combined extraction doubled the test activations, duplicate test labels to match
        if X_test.shape[0] == 2 * len(test_prompts):
            # Combined extraction detected - duplicate test labels
            test_labels_expanded = []
            for label in test_labels:
                test_labels_expanded.append(label)
                test_labels_expanded.append(label)
            test_labels = np.array(test_labels_expanded)

        # Helper function to calculate required samples for current cycle
        def get_required_samples(cycle: int, initial: int, first_inc: int, subsequent_inc: int) -> int:
            if cycle == 0:
                return initial  # 10
            elif cycle == 1:
                return initial + first_inc  # 10 + 20 = 30
            else:
                return initial + first_inc + (cycle - 1) * subsequent_inc  # 30 + 30*(n-1)

        print(f"\n  Dual Adaptive Training: {concept_name}")
        print(f"  {'─' * 60}")

        while not activation_graduated and iteration < self.max_iterations:
            iteration += 1
            cycle_iterations += 1
            iter_start_time = time.time()

            # Calculate required samples for current validation cycle
            required_samples = get_required_samples(
                validation_cycle,
                self.activation_initial_samples,
                self.activation_first_increment,
                self.activation_subsequent_increment
            )
            required_samples = min(required_samples, self.activation_max_samples)
            n_samples_per_class = required_samples // 2

            # Check if we need to generate more samples
            current_samples = len(all_train_prompts)
            if current_samples < required_samples:
                # Generate additional samples
                samples_to_generate = required_samples - current_samples
                n_pos_to_generate = samples_to_generate // 2
                n_neg_to_generate = samples_to_generate - n_pos_to_generate

                print(f"    [Cycle {validation_cycle}] Generating {samples_to_generate} new samples ({n_pos_to_generate}+{n_neg_to_generate})...")
                gen_start = time.time()

                # Check if custom generation function provided (for tripole training, etc.)
                if 'custom_generate_fn' in generation_config:
                    new_prompts, new_labels = generation_config['custom_generate_fn'](
                        samples_to_generate
                    )
                else:
                    # Default SUMO generation
                    new_prompts, new_labels = create_sumo_training_dataset(
                        concept=generation_config['concept'],
                        all_concepts=generation_config['all_concepts'],
                        negative_pool=generation_config['negative_pool'],
                        n_positives=n_pos_to_generate,
                        n_negatives=n_neg_to_generate,
                        use_category_relationships=True,
                        use_wordnet_relationships=True,
                    )
                gen_time = time.time() - gen_start

                # Extract activations for new samples
                print(f"    [Cycle {validation_cycle}] Extracting activations for {len(new_prompts)} new samples...")
                extract_start = time.time()
                new_activations = extract_activations(
                    generation_config['model'],
                    generation_config['tokenizer'],
                    new_prompts,
                    generation_config['device']
                )
                extract_time = time.time() - extract_start
                print(f"    ✓ Generated and extracted: {len(new_prompts)} samples [gen={gen_time*1000:.0f}ms, extract={extract_time:.1f}s]")

                # If combined extraction doubled the activations, duplicate labels to match
                labels_to_add = new_labels
                if new_activations.shape[0] == 2 * len(new_prompts):
                    # Combined extraction detected - duplicate labels
                    labels_to_add = []
                    for label in new_labels:
                        labels_to_add.append(label)
                        labels_to_add.append(label)

                # Accumulate samples
                all_train_prompts.extend(new_prompts)
                all_train_labels.extend(labels_to_add)
                if all_train_activations is None:
                    all_train_activations = new_activations
                else:
                    all_train_activations = np.vstack([all_train_activations, new_activations])

            # Use accumulated samples for training
            # If combined extraction doubled the activations, we need to slice to 2x required_samples
            if all_train_activations.shape[0] == 2 * len(all_train_prompts):
                # Combined extraction detected - need 2x samples to maintain label balance
                slice_size = min(required_samples * 2, len(all_train_labels))
            else:
                slice_size = min(required_samples, len(all_train_labels))

            X_train = all_train_activations[:slice_size]
            y_train = np.array(all_train_labels[:slice_size])

            # Count positives and negatives
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)

            # Train activation lens
            train_start = time.time()
            activation_classifier, metrics = train_simple_classifier(
                X_train, y_train, X_test, test_labels
            )
            train_time = time.time() - train_start

            # Get metrics
            train_f1 = metrics['train_f1']
            test_f1 = metrics['test_f1']
            overfit_gap = train_f1 - test_f1

            # Check graduation criteria
            meets_performance = test_f1 >= self.activation_target_accuracy
            not_overfitting = overfit_gap <= 0.1
            sufficient_iterations = cycle_iterations >= 3  # Minimum 3 iterations per cycle

            graduated = meets_performance and not_overfitting and sufficient_iterations

            # Check if we're stuck (3 iterations without graduating) - need more data
            stuck_without_data = cycle_iterations >= 3 and not graduated
            if stuck_without_data:
                if validation_cycle < 4:  # Can still request more data (increased to 4 for final push)
                    print(f"    [Cycle {validation_cycle}] Failed to graduate after {cycle_iterations} iterations, requesting more data (cycle {validation_cycle}→{validation_cycle+1})")
                    validation_cycle += 1
                    cycle_iterations = 0  # Reset for new cycle
                    # Continue to next iteration with more samples
                else:
                    # Hit max cycles (4), accept what we have
                    print(f"    [Cycle {validation_cycle}] Failed to graduate after {cycle_iterations} iterations, but at max cycles - accepting best result")
                    activation_graduated = True
                    activation_results = {
                        'train_f1': train_f1,
                        'test_f1': test_f1,
                        'samples_used': required_samples,
                        'iterations': iteration,
                    }
                    break  # Exit training loop

            if graduated:
                iter_time = time.time() - iter_start_time
                print(f"    [Iter {iteration:2d}] Activation: {required_samples:3d} samples ({n_pos}+{n_neg}), "
                      f"train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, gap={overfit_gap:.3f} ✓ GRADUATED "
                      f"[train={train_time*1000:.0f}ms, total={iter_time*1000:.0f}ms]")

                # Validate lens calibration
                should_validate = self.validate_lenses and self.model is not None

                if should_validate:
                    # Run validation using the full validation logic from train_concept
                    print(f"      Validating lens calibration...")
                    from .lens_validation import infer_concept_domain, DEFAULT_TEST_PROMPTS

                    try:
                        # Infer expected domain
                        concept_dict = {
                            'sumo_term': concept_name,
                            'definition': '',
                        }
                        expected_domain = infer_concept_domain(concept_dict)

                        # Test on diverse prompts
                        results_by_domain = {}
                        for domain, prompt in DEFAULT_TEST_PROMPTS.items():
                            # Extract activation for this prompt
                            domain_acts = extract_activations(
                                generation_config['model'],
                                generation_config['tokenizer'],
                                [prompt],
                                generation_config['device'],
                                layer_idx=self.validation_layer_idx
                            )

                            # If combined extraction gave us 2 activations, take the first one
                            # (both prompt and generation - we just need one for validation)
                            if len(domain_acts) == 2:
                                domain_acts = domain_acts[0:1]

                            # Get lens prediction
                            classifier_device = next(activation_classifier.parameters()).device
                            act_tensor = torch.tensor(domain_acts, dtype=torch.float32).to(classifier_device)
                            with torch.no_grad():
                                logit = activation_classifier(act_tensor).item()

                            results_by_domain[domain] = {'logit': logit}

                        # Calculate rankings
                        sorted_domains = sorted(results_by_domain.items(), key=lambda x: x[1]['logit'], reverse=True)
                        for rank, (domain, data) in enumerate(sorted_domains, 1):
                            data['rank'] = rank

                        # Extract metrics
                        target_rank = results_by_domain[expected_domain]['rank']
                        target_logit = results_by_domain[expected_domain]['logit']

                        other_domains = [d for d in results_by_domain if d != expected_domain]
                        other_ranks = [results_by_domain[d]['rank'] for d in other_domains]
                        avg_other_rank = np.mean(other_ranks) if other_ranks else 0

                        # Calibration score
                        num_domains = len(results_by_domain)
                        target_score = 1.0 - (target_rank - 1) / (num_domains - 1)
                        specificity_score = (avg_other_rank - 1) / (num_domains - 1)
                        calibration_score = (target_score + specificity_score) / 2

                        # Determine pass criteria based on validation mode
                        if self.validation_mode == 'falloff':
                            tier_info = self._get_validation_tier(validation_cycle)
                            passed = (target_rank <= tier_info['max_target_rank']) and \
                                   (avg_other_rank >= tier_info['min_other_rank']) and \
                                   (calibration_score >= tier_info['min_score'])
                        else:  # loose or strict
                            passed = True  # For simplicity, accept in incremental mode

                        # Assign quality grade
                        score = calibration_score
                        if score >= 0.5:
                            grade = 'A'
                        elif score >= 0.2:
                            grade = 'B'
                        else:
                            grade = 'C'

                        # Format message
                        if self.validation_mode == 'falloff':
                            tier_name = tier_info['tier'].upper()
                            mode_msg = f"[FALLOFF {tier_name}, cycle {validation_cycle}, iter {iteration}]"
                        elif self.validation_mode == 'strict':
                            mode_msg = f"[STRICT mode, iter {iteration}]"
                        else:  # loose
                            mode_msg = f"[LOOSE mode, iter {iteration}]"

                        # Check if passed or grade meets tier requirements
                        if passed:
                            print(f"      ✓ Calibration validated {mode_msg} (target=#{target_rank}, "
                                  f"others={avg_other_rank:.1f}, score={calibration_score:.2f}, grade={grade})")
                            activation_graduated = True
                            activation_results = {
                                'train_f1': train_f1,
                                'test_f1': test_f1,
                                'samples_used': required_samples,
                                'iterations': iteration,
                            }
                        else:
                            # Check if grade meets tier requirements even if numerical thresholds failed
                            if self._grade_meets_target(grade, tier_info['target_grade']):
                                print(f"      ✗ Calibration {mode_msg} (target=#{target_rank}, "
                                      f"others={avg_other_rank:.1f}, score={calibration_score:.2f}, grade={grade})")
                                print(f"         → {grade}-tier acceptable at {tier_info['tier']} tier (cycle {validation_cycle})")
                                activation_graduated = True
                                activation_results = {
                                    'train_f1': train_f1,
                                    'test_f1': test_f1,
                                    'samples_used': required_samples,
                                    'iterations': iteration,
                                }
                            else:
                                # Need more data
                                print(f"      ✗ Calibration {mode_msg} (target=#{target_rank}, "
                                      f"others={avg_other_rank:.1f}, score={calibration_score:.2f}, grade={grade})")
                                print(f"         → Pushing for {tier_info['target_grade']}-tier (cycle {validation_cycle}→{validation_cycle+1})")
                                print(f"         → Requesting more samples for next cycle")
                                validation_cycle += 1
                                cycle_iterations = 0  # Reset for new cycle
                                # Continue to next iteration with more samples

                    except Exception as e:
                        print(f"      ⚠️  Validation error: {e}")
                        # Accept graduation despite validation error
                        activation_graduated = True
                        activation_results = {
                            'train_f1': train_f1,
                            'test_f1': test_f1,
                            'samples_used': required_samples,
                            'iterations': iteration,
                        }
                else:
                    activation_graduated = True
                    activation_results = {
                        'train_f1': train_f1,
                        'test_f1': test_f1,
                        'samples_used': required_samples,
                        'iterations': iteration,
                    }
            else:
                # Print learning curve for non-graduated iterations
                reason = []
                if not meets_performance:
                    reason.append(f"test_f1={test_f1:.3f}<{self.activation_target_accuracy:.3f}")
                if not not_overfitting:
                    reason.append(f"overfit_gap={overfit_gap:.3f}>0.100")
                if not sufficient_iterations:
                    reason.append(f"cycle_iter={cycle_iterations}<3")

                reason_str = ", ".join(reason) if reason else ""
                iter_time = time.time() - iter_start_time
                print(f"    [Iter {iteration:2d}] Activation: {required_samples:3d} samples ({n_pos}+{n_neg}), "
                      f"train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, gap={overfit_gap:.3f}"
                      + (f" [{reason_str}]" if reason_str else "") + f" [train={train_time*1000:.0f}ms, total={iter_time*1000:.0f}ms]")

        concept_time = time.time() - concept_start_time
        print(f"  ✓ Adaptive training complete:")
        print(f"    Activation: {required_samples if activation_results else 0} samples, "
              f"F1={activation_results['test_f1'] if activation_results else 0:.3f}")
        print(f"    Total time: {concept_time:.1f}s")

        return {
            'activation': activation_results,
            'activation_classifier': activation_classifier,
            'text': None,
            'total_iterations': iteration,
            'total_time': concept_time,
        }

    def train_concept(
        self,
        concept_name: str,
        # Activation lens inputs
        train_activations: Optional[np.ndarray],
        train_labels: Optional[np.ndarray],
        test_activations: Optional[np.ndarray],
        test_labels: Optional[np.ndarray],
        # Text lens inputs
        train_texts: Optional[List[str]],
        test_texts: Optional[List[str]],
    ) -> Dict:
        """
        Train both lens types until graduation.

        Args:
            concept_name: Name of concept
            train_activations: Training activation vectors [n_samples, hidden_dim]
            train_labels: Training labels [n_samples]
            test_activations: Test activation vectors [n_test, hidden_dim]
            test_labels: Test labels [n_test]
            train_texts: Training text prompts
            test_texts: Test text prompts

        Returns:
            Dict with training results:
            {
                'activation': {...} or None,
                'text': {...} or None,
                'total_iterations': int,
                'total_time': float,
            }
        """
        start_time = time.time()

        # State tracking
        activation_graduated = not self.train_activation  # Skip if disabled
        text_graduated = not self.train_text

        activation_classifier = None
        text_lens = None
        activation_results = None
        text_results = None

        iteration = 0
        validation_cycle = 0  # Track which validation cycle we're in (0, 1, 2...)

        # Helper function to calculate required samples for current cycle
        def get_required_samples(cycle: int, initial: int, first_inc: int, subsequent_inc: int) -> int:
            if cycle == 0:
                return initial  # 10
            elif cycle == 1:
                return initial + first_inc  # 10 + 20 = 30
            else:
                return initial + first_inc + (cycle - 1) * subsequent_inc  # 30 + 30*(n-1)

        print(f"\n  Dual Adaptive Training: {concept_name}")
        print(f"  {'─' * 60}")

        while not (activation_graduated and text_graduated) and iteration < self.max_iterations:
            iteration += 1
            iter_start_time = time.time()

            # === ACTIVATION LENS ===
            if not activation_graduated and train_activations is not None:
                # Calculate required samples for current validation cycle
                required_samples = get_required_samples(
                    validation_cycle,
                    self.activation_initial_samples,
                    self.activation_first_increment,
                    self.activation_subsequent_increment
                )
                required_samples = min(required_samples, self.activation_max_samples)
                n_samples_per_class = required_samples // 2

                # Find where negatives start
                neg_start = np.where(train_labels == 0)[0][0] if 0 in train_labels else len(train_labels)

                # Get balanced positives and negatives
                n_pos = min(n_samples_per_class, neg_start)
                n_neg = min(n_samples_per_class, len(train_labels) - neg_start)

                # Check if we have enough samples available
                if n_pos < n_samples_per_class or n_neg < n_samples_per_class:
                    print(f"      ⚠️  Insufficient samples (need {n_samples_per_class}+{n_samples_per_class}, have {n_pos}+{n_neg})")
                    # Graduate with what we have
                    activation_graduated = True
                    continue

                # Sample activations and labels
                X_train = np.vstack([
                    train_activations[:n_pos],
                    train_activations[neg_start:neg_start + n_neg]
                ])
                y_train = np.concatenate([
                    train_labels[:n_pos],
                    train_labels[neg_start:neg_start + n_neg]
                ])

                # Import here to avoid circular dependency
                from .sumo_classifiers import train_simple_classifier

                activation_classifier, metrics = train_simple_classifier(
                    X_train, y_train,
                    test_activations, test_labels,
                    hidden_dim=128,
                    epochs=50,
                    lr=0.001,
                )

                train_f1 = metrics['train_f1']
                test_f1 = metrics['test_f1']
                overfit_gap = train_f1 - test_f1

                n_samples = n_pos + n_neg

                # Check graduation: test performance + no severe overfitting + minimum iterations
                min_iterations = 3  # Require at least 3 iterations
                max_overfit_gap = 0.10  # Allow up to 10% gap between train and test

                meets_performance = test_f1 >= self.activation_target_accuracy
                not_overfitting = overfit_gap <= max_overfit_gap
                sufficient_iterations = iteration >= min_iterations

                if meets_performance and not_overfitting and sufficient_iterations:
                    activation_graduated = True
                    activation_results = {
                        'samples': n_samples,
                        'iterations': iteration,
                        'train_f1': train_f1,
                        'test_f1': test_f1,
                        'overfit_gap': overfit_gap,
                        'test_precision': metrics['test_precision'],
                        'test_recall': metrics['test_recall'],
                    }
                    iter_time = time.time() - iter_start_time
                    print(f"    [Iter {iteration:2d}] Activation: {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, gap={overfit_gap:.3f} ✓ GRADUATED [{iter_time*1000:.0f}ms]")

                    # Validate every time we graduate (regardless of iteration number)
                    # The validation tier is determined by which cycle we're in
                    should_validate = self.validate_lenses and self.model is not None

                    if should_validate:
                        print(f"      Validating lens calibration...")
                        from .lens_validation import infer_concept_domain, DEFAULT_TEST_PROMPTS

                        # Run inline validation (no need to save/load)
                        try:
                            # Infer expected domain
                            concept_dict = {
                                'sumo_term': concept_name,
                                'definition': '',
                            }
                            expected_domain = infer_concept_domain(concept_dict)

                            # Test on diverse prompts
                            from .sumo_classifiers import extract_activations

                            results_by_domain = {}
                            for domain, prompt in DEFAULT_TEST_PROMPTS.items():
                                # Extract activation for this prompt
                                domain_acts = extract_activations(
                                    self.model,
                                    self.tokenizer,
                                    [prompt],
                                    device=str(self.model.device),
                                    layer_idx=self.validation_layer_idx
                                )

                                # If combined extraction gave us 2 activations, take the first one
                                # (both prompt and generation - we just need one for validation)
                                if len(domain_acts) == 2:
                                    domain_acts = domain_acts[0:1]

                                # Get lens prediction
                                # Get device from classifier parameters
                                classifier_device = next(activation_classifier.parameters()).device
                                act_tensor = torch.tensor(domain_acts, dtype=torch.float32).to(classifier_device)
                                with torch.no_grad():
                                    logit = activation_classifier(act_tensor).item()

                                results_by_domain[domain] = {'logit': logit}

                            # Calculate rankings
                            sorted_domains = sorted(results_by_domain.items(), key=lambda x: x[1]['logit'], reverse=True)
                            for rank, (domain, data) in enumerate(sorted_domains, 1):
                                data['rank'] = rank

                            # Extract metrics
                            target_rank = results_by_domain[expected_domain]['rank']
                            target_logit = results_by_domain[expected_domain]['logit']

                            other_domains = [d for d in results_by_domain if d != expected_domain]
                            other_ranks = [results_by_domain[d]['rank'] for d in other_domains]
                            other_logits = [results_by_domain[d]['logit'] for d in other_domains]

                            avg_other_rank = np.mean(other_ranks) if other_ranks else 0
                            avg_other_logit = np.mean(other_logits) if other_logits else 0

                            # Calibration score
                            num_domains = len(results_by_domain)
                            target_score = 1.0 - (target_rank - 1) / (num_domains - 1)
                            specificity_score = (avg_other_rank - 1) / (num_domains - 1)
                            calibration_score = (target_score + specificity_score) / 2

                            # Determine pass criteria based on validation mode
                            if self.validation_mode == 'loose':
                                # Loose mode: always pass, just record the grade
                                passed = True
                            elif self.validation_mode == 'falloff':
                                # Falloff mode: use tiered thresholds based on cycle
                                tier_info = self._get_validation_tier(validation_cycle)
                                passed = (target_rank <= tier_info['max_target_rank']) and \
                                       (avg_other_rank >= tier_info['min_other_rank']) and \
                                       (calibration_score >= tier_info['min_score'])
                            else:  # strict mode
                                # Strict mode: fixed threshold throughout
                                tier_info = {
                                    'tier': 'strict',
                                    'min_score': self.validation_threshold,
                                    'target_grade': 'A',
                                    'max_target_rank': 3,
                                    'min_other_rank': 10.0,
                                }
                                passed = (target_rank <= tier_info['max_target_rank']) and \
                                       (avg_other_rank >= tier_info['min_other_rank']) and \
                                       (calibration_score >= tier_info['min_score'])

                            validation_result = {
                                'concept': concept_name,
                                'expected_domain': expected_domain,
                                'target_rank': target_rank,
                                'target_logit': target_logit,
                                'avg_other_rank': avg_other_rank,
                                'avg_other_logit': avg_other_logit,
                                'calibration_score': calibration_score,
                                'passed': passed,
                                'all_results': results_by_domain,
                                'validation_mode': self.validation_mode,
                                'validation_tier': tier_info['tier'] if self.validation_mode == 'falloff' else self.validation_mode,
                                'iteration': iteration,
                            }

                            # Add validation results
                            activation_results['validation'] = validation_result

                            # Assign quality grade based on calibration score
                            score = validation_result['calibration_score']
                            if score >= 0.5:
                                grade = 'A'
                            elif score >= 0.2:
                                grade = 'B'
                            else:
                                grade = 'C'

                            validation_result['quality_grade'] = grade

                            # Format mode-specific messages
                            if self.validation_mode == 'loose':
                                mode_msg = f"[LOOSE mode, cycle {validation_cycle}]"
                            elif self.validation_mode == 'falloff':
                                tier_name = tier_info['tier'].upper()
                                mode_msg = f"[FALLOFF {tier_name}, cycle {validation_cycle}, iter {iteration}]"
                            else:  # strict
                                mode_msg = f"[STRICT mode, iter {iteration}/{self.max_iterations}]"

                            # Check if passed validation
                            if validation_result['passed']:
                                print(f"      ✓ Calibration validated {mode_msg} (target=#{validation_result['target_rank']}, "
                                      f"others={validation_result['avg_other_rank']:.1f}, "
                                      f"score={validation_result['calibration_score']:.2f}, grade={grade})")
                            else:
                                # Determine if we should continue training (revoke graduation)
                                should_continue = False

                                if self.validation_mode == 'loose':
                                    continue_msg = f"{grade}-tier recorded (loose mode, no blocking)"
                                    should_continue = False
                                elif self.validation_mode == 'falloff':
                                    if tier_info['tier'] == 'strict':
                                        if self._grade_meets_target(grade, tier_info['target_grade']):
                                            continue_msg = f"{grade}-tier acceptable at strict tier (cycle {validation_cycle})"
                                            should_continue = False
                                        else:
                                            continue_msg = f"Pushing for {tier_info['target_grade']}-tier (cycle {validation_cycle}→{validation_cycle+1})"
                                            should_continue = True
                                    elif tier_info['tier'] == 'high':
                                        if self._grade_meets_target(grade, tier_info['target_grade']):
                                            continue_msg = f"{grade}-tier acceptable at high tier (cycle {validation_cycle})"
                                            should_continue = False
                                        else:
                                            continue_msg = f"Pushing for {tier_info['target_grade']}-tier (cycle {validation_cycle}→{validation_cycle+1})"
                                            should_continue = True
                                    elif tier_info['tier'] == 'medium':
                                        if self._grade_meets_target(grade, tier_info['target_grade']):
                                            continue_msg = f"{grade}-tier acceptable at medium tier (cycle {validation_cycle})"
                                            should_continue = False
                                        else:
                                            continue_msg = f"Pushing for {tier_info['target_grade']}-tier (cycle {validation_cycle}→{validation_cycle+1})"
                                            should_continue = True
                                    else:  # relaxed
                                        if self._grade_meets_target(grade, tier_info['target_grade']):
                                            continue_msg = f"{grade}-tier acceptable at relaxed tier (cycle {validation_cycle})"
                                            should_continue = False
                                        else:
                                            continue_msg = f"Failed even relaxed tier, continuing (cycle {validation_cycle}→{validation_cycle+1})"
                                            should_continue = True
                                else:  # strict mode
                                    continue_msg = f"Failed strict validation, continuing (iter {iteration}/{self.max_iterations})"
                                    should_continue = True

                                print(f"      ✗ Calibration {mode_msg} (target=#{validation_result['target_rank']}, "
                                      f"others={validation_result['avg_other_rank']:.1f}, "
                                      f"score={validation_result['calibration_score']:.2f}, grade={grade})")
                                print(f"         → {continue_msg}")

                                # Revoke graduation if we should continue training
                                if should_continue:
                                    activation_graduated = False
                                    activation_results = None
                                    # Increment validation cycle to request more samples
                                    validation_cycle += 1
                                    next_samples = get_required_samples(
                                        validation_cycle,
                                        self.activation_initial_samples,
                                        self.activation_first_increment,
                                        self.activation_subsequent_increment
                                    )
                                    print(f"         → Requesting {next_samples} samples for next cycle")

                        except Exception as e:
                            print(f"      ⚠️  Validation failed: {e}")
                            # Continue with graduation despite validation error
                else:
                    # Print learning curve for non-graduated iterations
                    reason = []
                    if not meets_performance:
                        reason.append(f"test_f1={test_f1:.3f}<{self.activation_target_accuracy:.3f}")
                    if not not_overfitting:
                        reason.append(f"overfit_gap={overfit_gap:.3f}>{max_overfit_gap:.3f}")
                    if not sufficient_iterations:
                        reason.append(f"iter={iteration}<{min_iterations}")

                    reason_str = ", ".join(reason) if reason else ""
                    iter_time = time.time() - iter_start_time
                    print(f"    [Iter {iteration:2d}] Activation: {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, gap={overfit_gap:.3f}"
                          + (f" [{reason_str}]" if reason_str else "") + f" [{iter_time*1000:.0f}ms]")
                # Note: Sample count stays constant within validation cycle
                # Only increases when validation_cycle increments (on validation failure)

            # === TEXT LENS ===
            if not text_graduated and train_texts is not None:
                # Get balanced samples (data is [pos1..posN, neg1..negN])
                # Need to ensure we get both classes
                n_samples_per_class = text_level // 2

                # Find where negatives start (labels switch from 1 to 0)
                neg_start = np.where(train_labels == 0)[0][0] if 0 in train_labels else len(train_labels)

                # Get balanced positives and negatives
                n_pos = min(n_samples_per_class, neg_start)
                n_neg = min(n_samples_per_class, len(train_labels) - neg_start)

                # Sample PROMPTS and labels
                pos_prompts = train_texts[:n_pos]
                neg_prompts = train_texts[neg_start:neg_start + n_neg]
                train_prompts_sample = pos_prompts + neg_prompts
                train_labels_sample = [1] * n_pos + [0] * n_neg

                # Generate model responses for training prompts
                print(f"      Generating {len(train_prompts_sample)} training responses...")
                train_responses = self._generate_responses(train_prompts_sample)

                # Generate model responses for test prompts
                print(f"      Generating {len(test_texts)} test responses...")
                test_responses = self._generate_responses(test_texts)

                # Train text lens on MODEL RESPONSES (not prompts)
                # Import only if needed (legacy code path)
                from archive.training.text_lenses import BinaryTextLens
                text_lens = BinaryTextLens(concept_name)
                metrics = text_lens.train(
                    train_texts=train_responses,
                    train_labels=train_labels_sample,
                    test_texts=test_responses,
                    test_labels=test_labels,
                )

                test_acc = metrics['test_f1']
                n_samples = len(train_texts_sample)

                # Check graduation
                if test_acc >= self.text_target_accuracy:
                    text_graduated = True
                    text_results = {
                        'samples': n_samples,
                        'iterations': iteration,
                        'test_f1': test_acc,
                        'test_precision': metrics['test_precision'],
                        'test_recall': metrics['test_recall'],
                    }
                    print(f"    [Iter {iteration:2d}] Text:       {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f} ✓ GRADUATED")
                else:
                    print(f"    [Iter {iteration:2d}] Text:       {n_samples:3d} samples ({n_pos}+{n_neg}), "
                          f"F1={test_acc:.3f}")

                # Increment for next iteration
                text_level += self.text_increment
                text_level = min(text_level, self.text_max_samples)

            # Early stop if both graduated
            if activation_graduated and text_graduated:
                break

        # Warnings for non-graduation
        if not activation_graduated and self.train_activation:
            print(f"    ⚠️  Activation lens did not graduate after {iteration} iterations")
            if activation_classifier is not None:
                # Use final results with current cycle's sample count
                final_samples = get_required_samples(
                    validation_cycle,
                    self.activation_initial_samples,
                    self.activation_first_increment,
                    self.activation_subsequent_increment
                )
                activation_results = {
                    'samples': final_samples,
                    'iterations': iteration,
                    'graduated': False,
                }

        if not text_graduated and self.train_text:
            print(f"    ⚠️  Text lens did not graduate after {iteration} iterations")
            if text_lens is not None:
                final_samples = get_required_samples(
                    validation_cycle,
                    self.text_initial_samples,
                    self.text_first_increment,
                    self.text_subsequent_increment
                )
                text_results = {
                    'samples': final_samples,
                    'iterations': iteration,
                    'graduated': False,
                }

        total_time = time.time() - start_time

        return {
            'concept': concept_name,
            'activation': activation_results,
            'text': text_results,
            'activation_classifier': activation_classifier,
            'text_lens': text_lens,
            'total_iterations': iteration,
            'total_time': total_time,
        }


__all__ = [
    'DualAdaptiveTrainer',
]
