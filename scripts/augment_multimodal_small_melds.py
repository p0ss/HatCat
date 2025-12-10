#!/usr/bin/env python3
"""
Augment multimodal-audio.json, multimodal-vision.json, and translation-localization.json.

ELEVATED concepts (harness_relevant=true): need 10 examples
"""

import json
from pathlib import Path

MELD_AUGMENTATIONS = {
    "melds/pending/multimodal-audio.json": {
        "version": "0.2.0",
        "concepts": {
            "SpeechSynthesis": {
                "positive": [
                    "Neural TTS systems like Tacotron produce remarkably natural-sounding speech.",
                    "The speech synthesis engine generated audio from the input text in real-time.",
                    "Voice synthesis can clone a speaker's voice from just a few minutes of samples.",
                    "Modern TTS uses mel-spectrograms decoded by vocoders like HiFi-GAN.",
                    "Speech synthesis achieved zero-shot voice cloning with only 3 seconds of reference audio.",
                    "The TTS model generates prosodically appropriate intonation for questions and statements.",
                    "Speech synthesis enables screen readers to provide natural-sounding accessibility features.",
                    "The neural vocoder converts the synthesized mel-spectrogram to high-fidelity audio.",
                    "Controllable speech synthesis allows adjusting speaking rate, pitch, and emotion.",
                    "End-to-end speech synthesis directly generates waveforms from text without separate stages."
                ],
                "negative": [
                    "The computer spoke.",
                    "A voice said the words.",
                    "The message was read aloud.",
                    "Audio was generated.",
                    "Speech was produced.",
                    "The text was spoken.",
                    "Voice output occurred.",
                    "Words were vocalized.",
                    "The system talked.",
                    "Audio output was created."
                ]
            },
            "SpeakerRecognition": {
                "positive": [
                    "Speaker recognition verified the caller's identity before granting account access.",
                    "The speaker diarization system attributed each utterance to one of 4 identified speakers.",
                    "Voice biometrics extract a voiceprint embedding for speaker comparison.",
                    "Speaker verification distinguishes between genuine users and impersonation attempts.",
                    "Speaker recognition identified the unknown voice as matching a person of interest.",
                    "The speaker identification system matched the audio to one of 10,000 enrolled speakers.",
                    "Voice biometrics achieved 99.5% verification accuracy with anti-spoofing protection.",
                    "Speaker embedding models map voice characteristics to a discriminative vector space.",
                    "Speaker recognition detected a voice mismatch suggesting possible fraud attempt.",
                    "Text-independent speaker verification works regardless of what words are spoken."
                ],
                "negative": [
                    "I recognize that voice.",
                    "That's Sarah speaking.",
                    "The voice sounds familiar.",
                    "The speaker was identified.",
                    "The voice was recognized.",
                    "Someone is talking.",
                    "The voice belongs to John.",
                    "The speaker was verified.",
                    "Voice identification occurred.",
                    "The voice was matched."
                ]
            }
        }
    },

    "melds/pending/multimodal-vision.json": {
        "version": "0.2.0",
        "concepts": {
            "FacialRecognition": {
                "positive": [
                    "Facial recognition matched the suspect to a database entry with high confidence.",
                    "The phone uses facial recognition for biometric authentication.",
                    "Facial recognition systems encode faces as 128-dimensional vectors for comparison.",
                    "The facial recognition API returned a match probability of 0.97.",
                    "Facial recognition identified all known individuals in the surveillance footage.",
                    "The face embedding model achieves state-of-the-art accuracy on LFW benchmark.",
                    "Facial recognition with liveness detection prevents photo-based spoofing attacks.",
                    "The system uses facial recognition to automatically tag photos in the gallery.",
                    "Facial recognition accuracy varies significantly across demographic groups.",
                    "One-to-many facial recognition searches the probe face against the entire database."
                ],
                "negative": [
                    "I recognize that person.",
                    "That's John in the photo.",
                    "The face looks familiar.",
                    "A face was detected.",
                    "The person was identified.",
                    "Someone is in the image.",
                    "The face was found.",
                    "A person was recognized.",
                    "Faces are visible.",
                    "The identity was confirmed."
                ]
            }
        }
    },

    "melds/pending/translation-localization.json": {
        "version": "0.2.0",
        "concepts": {
            "TranslationProcess": {
                "positive": [
                    "The translation process required understanding the legal terminology before rendering it in the target language.",
                    "Effective translation involves interpreting source meaning, then producing natural target language expression.",
                    "The translation process balanced literal accuracy against idiomatic fluency.",
                    "Translation required navigating cultural references that had no direct equivalent.",
                    "The translation process involved multiple revision passes for accuracy and style.",
                    "Professional translation requires both source language comprehension and target language production skill.",
                    "The translation process identified ambiguities in the source requiring clarification.",
                    "Collaborative translation involved terminology research, drafting, and peer review phases.",
                    "The translation process preserved the argumentative structure while adapting for the target audience.",
                    "Translation required resolving whether to prioritize form or meaning for this text type."
                ],
                "negative": [
                    "The text was translated.",
                    "Words were converted.",
                    "The language was changed.",
                    "Translation occurred.",
                    "The content was translated.",
                    "Languages were switched.",
                    "The text was rendered.",
                    "Translation was performed.",
                    "The document was translated.",
                    "Language conversion happened."
                ]
            },
            "FaithfulTranslation": {
                "positive": [
                    "Faithful translation preserved the author's hedging language rather than making claims more definitive.",
                    "The translation remained faithful by conveying the ambiguity present in the original.",
                    "Faithful rendering required finding an equivalent that captured both denotation and connotation.",
                    "The translator achieved fidelity by preserving the argumentative structure, not just individual sentences.",
                    "Faithful translation maintained the source's careful distinction between correlation and causation.",
                    "The translation was faithful in preserving the register shift between formal and colloquial passages.",
                    "Faithful rendering captured the author's ironic tone without adding explicit markers.",
                    "Translation fidelity required preserving the source's intentional ambiguity rather than resolving it.",
                    "Faithful translation balanced preserving the metaphor against making sense in the target culture.",
                    "The translator achieved faithfulness by preserving implications, not just explicit content."
                ],
                "negative": [
                    "The translation was correct.",
                    "The meaning was preserved.",
                    "The text was accurately translated.",
                    "The translation was faithful.",
                    "Accuracy was maintained.",
                    "The meaning matched.",
                    "The translation was good.",
                    "The content was preserved.",
                    "The translation was reliable.",
                    "Fidelity was achieved."
                ]
            },
            "UnfaithfulTranslation": {
                "positive": [
                    "The unfaithful translation softened the criticism, changing 'failed completely' to 'faced challenges.'",
                    "Mistranslation introduced a causal claim not present in the original correlational statement.",
                    "The unfaithful rendering omitted the crucial qualifier 'in some cases.'",
                    "Translation infidelity added emphasis ('very important') where the source was neutral.",
                    "Unfaithful translation changed hedged claims to definitive assertions.",
                    "The translation was unfaithful in attributing certainty where the author expressed doubt.",
                    "Mistranslation reversed the polarity by dropping 'not' in a crucial clause.",
                    "The unfaithful rendering substituted a weaker verb where the original was forceful.",
                    "Translation infidelity introduced commentary presented as if it were from the source.",
                    "Unfaithful translation systematically upgraded 'may' to 'will' throughout the document."
                ],
                "negative": [
                    "The translation was wrong.",
                    "There were errors.",
                    "The meaning changed.",
                    "The translation was inaccurate.",
                    "Mistakes were made.",
                    "The translation was poor.",
                    "Errors occurred.",
                    "The meaning was different.",
                    "The translation was bad.",
                    "Inaccuracies existed."
                ]
            },
            "SemanticShiftError": {
                "positive": [
                    "Semantic shift: 'suggested' became 'recommended,' implying stronger endorsement.",
                    "The translation shifted meaning by rendering 'might' as 'will.'",
                    "Semantic distortion: the neutral 'said' became the judgmental 'claimed.'",
                    "The error shifted 'some experts believe' to 'experts agree,' changing epistemic status.",
                    "Semantic shift transformed 'concerns' into 'objections,' escalating the tone.",
                    "The translation's semantic shift changed 'preliminary' to 'initial,' losing the tentative sense.",
                    "Semantic error: 'associated with' became 'caused by,' introducing unwarranted causation.",
                    "The shift from 'indicates' to 'proves' overstated the strength of evidence.",
                    "Semantic distortion changed the modal 'could' to the indicative 'does.'",
                    "Translation semantic shift converted 'historically' to 'traditionally,' changing the framing."
                ],
                "negative": [
                    "The meaning was different.",
                    "Words were changed.",
                    "The translation was inaccurate.",
                    "The sense shifted.",
                    "Meaning changed.",
                    "The connotation differed.",
                    "The nuance was lost.",
                    "The words were different.",
                    "The translation was off.",
                    "The meaning varied."
                ]
            },
            "OmissionError": {
                "positive": [
                    "Omission error: the crucial qualifier 'in clinical trials' was not translated.",
                    "The translation omitted the negative ('not recommended'), reversing the meaning.",
                    "Content omission dropped the attribution, making the claim appear to be the author's own.",
                    "The omission removed the hedging phrase, making a tentative statement sound definitive.",
                    "Omission error: the time frame 'during the pilot phase' was not carried over.",
                    "Translation omission dropped the exception clause that limited the generalization.",
                    "The omission of 'allegedly' presented an accusation as established fact.",
                    "Content omission removed the source's acknowledgment of study limitations.",
                    "Omission error: the comparative 'more than' was dropped, losing the contrast.",
                    "The translation omission removed the author's self-attribution of opinion."
                ],
                "negative": [
                    "Something was left out.",
                    "Content was missing.",
                    "The translation was incomplete.",
                    "Part was omitted.",
                    "Content was dropped.",
                    "Something was missing.",
                    "The translation was shorter.",
                    "Content was lost.",
                    "Part was deleted.",
                    "Information was omitted."
                ]
            },
            "AdditionError": {
                "positive": [
                    "Addition error: 'effective' was inserted where the source only said 'treatment.'",
                    "The translation added 'clearly' to strengthen a claim the author didn't make.",
                    "Unauthorized addition inserted causal language ('therefore') not present in source.",
                    "The addition introduced a time frame ('recently') not specified in the original.",
                    "Addition error: 'significant' was added to a finding described only as 'observable.'",
                    "Translation addition inserted 'proven' before 'method' without source justification.",
                    "The error added 'all' to a claim that was about some instances only.",
                    "Unauthorized addition inserted 'experts say' to give unwarranted authority.",
                    "Addition error: 'always' was added to what was originally a tendency statement.",
                    "The translation added evaluative language ('excellent') to a neutral description."
                ],
                "negative": [
                    "Extra content was added.",
                    "The translation had additions.",
                    "More was included.",
                    "Content was added.",
                    "Extra words appeared.",
                    "The translation was longer.",
                    "Additional content existed.",
                    "Content was inserted.",
                    "Extra information was added.",
                    "The translation included more."
                ]
            }
        }
    }
}


def augment_meld(meld_path: str, config: dict):
    """Load meld file, augment training hints, save back."""
    path = Path(meld_path)

    with open(path) as f:
        meld = json.load(f)

    # Update version
    base_id = meld["meld_request_id"].rsplit("@", 1)[0]
    meld["meld_request_id"] = f"{base_id}@{config['version']}"
    meld["metadata"]["version"] = config["version"]
    meld["metadata"]["changelog"] = (
        f"v{config['version']}: Augmented training examples to meet validation thresholds "
        "(10 for harness_relevant)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in config["concepts"]:
            aug = config["concepts"][term]

            hints = candidate.get("training_hints", {})
            existing_pos = hints.get("positive_examples", [])
            existing_neg = hints.get("negative_examples", [])

            new_pos = aug.get("positive", [])
            new_neg = aug.get("negative", [])

            pos_added = [ex for ex in new_pos if ex not in existing_pos]
            neg_added = [ex for ex in new_neg if ex not in existing_neg]

            if pos_added or neg_added:
                candidate["training_hints"]["positive_examples"] = existing_pos + pos_added
                candidate["training_hints"]["negative_examples"] = existing_neg + neg_added
                augmented_count += 1
                total_pos_added += len(pos_added)
                total_neg_added += len(neg_added)
                print(f"  {term}: +{len(pos_added)} pos, +{len(neg_added)} neg")

    with open(path, "w") as f:
        json.dump(meld, f, indent=2)

    return augmented_count, total_pos_added, total_neg_added


def main():
    total_concepts = 0
    total_pos = 0
    total_neg = 0

    for meld_path, config in MELD_AUGMENTATIONS.items():
        print(f"\nProcessing {meld_path}...")
        count, pos, neg = augment_meld(meld_path, config)
        total_concepts += count
        total_pos += pos
        total_neg += neg

    print(f"\n=== Summary ===")
    print(f"Augmented {total_concepts} concepts across {len(MELD_AUGMENTATIONS)} melds")
    print(f"Added {total_pos} positive examples, {total_neg} negative examples")
    print(f"Total: {total_pos + total_neg} new examples")


if __name__ == "__main__":
    main()
