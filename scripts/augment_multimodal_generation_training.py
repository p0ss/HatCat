#!/usr/bin/env python3
"""
Augment multimodal-generation.json training examples to meet validation thresholds.

PROTECTED concepts (risk_level: high): need 15 examples - VideoGeneration
ELEVATED concepts (harness_relevant=true): need 10 examples
"""

import json
from pathlib import Path

AUGMENTED_EXAMPLES = {
    # ============ PROTECTED (need 15) ============

    "VideoGeneration": {
        "positive": [
            "Video generation created a photorealistic 10-second clip from the text description.",
            "The AI video generator produced smooth motion with consistent character identity.",
            "Sora generates high-definition videos with complex camera movements and scene transitions.",
            "Video generation from text enables creating fictional news footage indistinguishable from real.",
            "Video generation synthesized realistic human motion from the script description.",
            "The diffusion-based video model generates temporally coherent sequences.",
            "AI video generation created the celebrity endorsement that never actually happened.",
            "The video generator maintains object permanence and physical plausibility across frames.",
            "Video generation enables producing fake surveillance footage with specific timestamps.",
            "Neural video synthesis generates realistic crowds and environments from prompts.",
            "Video generation created the political speech footage that the candidate never gave.",
            "The model generates photorealistic video of specified individuals performing actions.",
            "AI video generation produced a fabricated historical event documentation.",
            "Video generation synthesizes realistic news anchor footage with customizable scripts.",
            "The video generator can produce fake documentary footage of events that never occurred."
        ],
        "negative": [
            "A video was created.",
            "The clip was generated.",
            "Video content was made.",
            "The footage was produced.",
            "Video was synthesized.",
            "The animation was generated.",
            "Moving images were created.",
            "The video was AI-made.",
            "Generated video was output.",
            "The video clip was produced.",
            "AI created the video.",
            "The footage was generated.",
            "Video content was output.",
            "The sequence was generated.",
            "AI video was created."
        ]
    },

    # ============ ELEVATED (need 10) ============

    "VisualContentGeneration": {
        "positive": [
            "Visual content generation creates images, videos, and 3D assets from textual descriptions.",
            "The generative AI system produces photorealistic visual content indistinguishable from photos.",
            "Visual generation capabilities enable creating synthetic media at scale.",
            "AI visual content generation raises authenticity concerns for journalism and evidence.",
            "The visual generator produces high-fidelity synthetic imagery across multiple modalities.",
            "Visual content generation can fabricate evidence of events that never occurred.",
            "Generative visual AI creates custom imagery matching specific descriptions.",
            "Visual generation systems produce synthetic faces, scenes, and documents.",
            "The visual content generator enables creating misleading imagery at unprecedented scale.",
            "AI visual generation requires new approaches to content authentication."
        ],
        "negative": [
            "Visual content was created.",
            "Images were generated.",
            "The visual was made.",
            "Content was produced.",
            "Visuals were output.",
            "The image was created.",
            "Visual material was generated.",
            "The content was made.",
            "Graphics were produced.",
            "The visual was generated."
        ]
    },

    "TextToImage": {
        "positive": [
            "Text-to-image generation produced a photorealistic portrait matching the detailed description.",
            "Stable Diffusion generates high-quality images from natural language prompts.",
            "The text-to-image model created a convincing fake photo of a non-existent location.",
            "DALL-E generates creative visualizations from imaginative text descriptions.",
            "Text-to-image AI enables creating visual evidence of events that never happened.",
            "The T2I model produces images indistinguishable from photographs.",
            "Text-to-image generation can synthesize realistic images of any described scene.",
            "Midjourney generates artistic and photorealistic images from text prompts.",
            "Text-to-image systems enable visual fabrication at unprecedented scale.",
            "The model translates detailed textual descriptions into corresponding visual content."
        ],
        "negative": [
            "An image was generated from text.",
            "The prompt produced an image.",
            "Text was converted to image.",
            "The description became a picture.",
            "T2I generation occurred.",
            "The text made an image.",
            "A picture was generated.",
            "The prompt created an image.",
            "Text-to-image ran.",
            "An image was produced."
        ]
    },

    "ImageToImage": {
        "positive": [
            "Image-to-image translation transformed the sketch into a photorealistic rendering.",
            "The I2I model converted the photograph to a different artistic style while preserving structure.",
            "Image-to-image enables transforming real photos into synthetic variants.",
            "ControlNet guides image generation with structural input from source images.",
            "Image-to-image translation can modify photographs to show things that weren't there.",
            "The model performs semantic image editing while maintaining realism.",
            "I2I transformation changed the person's appearance while keeping the pose.",
            "Image-to-image enables creating modified versions of real photographs.",
            "The image translation model converts between visual domains with high fidelity.",
            "Image-to-image techniques enable sophisticated photo manipulation."
        ],
        "negative": [
            "The image was transformed.",
            "One image became another.",
            "The image was modified.",
            "Image conversion occurred.",
            "The picture was changed.",
            "Image-to-image ran.",
            "The visual was altered.",
            "The image was processed.",
            "One picture made another.",
            "The image was converted."
        ]
    },

    "ImageEditing": {
        "positive": [
            "AI image editing seamlessly removed the person from the photograph with realistic infilling.",
            "The neural editor changed the subject's expression while maintaining photorealism.",
            "AI-powered image editing can alter photographs to misrepresent what actually occurred.",
            "The image editor removed objects and regenerated plausible background.",
            "AI editing changed the text on signs in the photograph to say something different.",
            "The neural image editor modified clothing and accessories while preserving identity.",
            "AI image editing enables changing who appears to be present in photographs.",
            "The editor performed semantic manipulation of image content guided by text.",
            "AI image editing can alter timestamps and metadata-relevant visual elements.",
            "The neural editor transformed the setting while keeping subjects photorealistic."
        ],
        "negative": [
            "The image was edited.",
            "Changes were made to the picture.",
            "The photo was modified.",
            "Editing was performed.",
            "The image was altered.",
            "The picture was changed.",
            "Image editing occurred.",
            "The visual was modified.",
            "The photo was touched up.",
            "Edits were made."
        ]
    },

    "Inpainting": {
        "positive": [
            "Inpainting seamlessly filled the removed area with contextually appropriate content.",
            "The inpainting model generated realistic content to replace the deleted object.",
            "Neural inpainting reconstructed the occluded region of the photograph.",
            "The model performed content-aware inpainting to remove and replace image regions.",
            "Inpainting regenerated the background after the subject was removed.",
            "The inpainting system fills masked regions with photorealistic synthesized content."
        ],
        "negative": [
            "The gap was filled.",
            "Missing content was generated.",
            "The hole was completed.",
            "Inpainting was applied.",
            "The area was filled in."
        ]
    }
}


def augment_meld():
    """Load meld file, augment training hints, save back."""
    meld_path = Path("melds/pending/multimodal-generation.json")

    with open(meld_path) as f:
        meld = json.load(f)

    # Update version
    meld["meld_request_id"] = "org.hatcat/multimodal-generation@0.2.0"
    meld["metadata"]["version"] = "0.2.0"
    meld["metadata"]["changelog"] = (
        "v0.2.0: Augmented training examples to meet validation thresholds "
        "(15 for high-risk, 10 for harness_relevant)"
    )

    augmented_count = 0
    total_pos_added = 0
    total_neg_added = 0

    for candidate in meld["candidates"]:
        term = candidate["term"]
        if term in AUGMENTED_EXAMPLES:
            aug = AUGMENTED_EXAMPLES[term]

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

    with open(meld_path, "w") as f:
        json.dump(meld, f, indent=2)

    print(f"\nAugmented {augmented_count} concepts")
    print(f"Added {total_pos_added} positive examples, {total_neg_added} negative examples")
    print(f"Total: {total_pos_added + total_neg_added} new examples")


if __name__ == "__main__":
    augment_meld()
