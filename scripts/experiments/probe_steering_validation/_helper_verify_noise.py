import json
from collections import Counter
with open("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16/calibration.json") as f:
    d = json.load(f)
sub = d["calibration"]
print(f"total concepts: {len(sub)}")
has_noise = sum(1 for v in sub.values() if "noise_fire_rate" in v)
print(f"with noise_fire_rate: {has_noise}")
print(f"timestamp: {d.get('noise_calibration_timestamp')}")
print(f"samples: {d.get('noise_calibration_samples')}")

layers = Counter()
for k in sub:
    if "_L" in k and "noise_fire_rate" in sub[k]:
        l = k.rsplit("_L", 1)[1]
        layers[f"L{l}"] += 1
print(f"\nnoise coverage by layer: {dict(sorted(layers.items()))}")

# Distribution of noise_fire_rate
import statistics
nfrs = [v["noise_fire_rate"] for v in sub.values() if "noise_fire_rate" in v]
print(f"\nnoise_fire_rate distribution: n={len(nfrs)}")
print(f"  min={min(nfrs):.3f} max={max(nfrs):.3f} mean={statistics.mean(nfrs):.3f} median={statistics.median(nfrs):.3f}")
buckets = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
counts = [0] * (len(buckets) - 1)
for v in nfrs:
    for i in range(len(buckets) - 1):
        if buckets[i] <= v < buckets[i+1]:
            counts[i] += 1
            break
print("  buckets:")
for i, c in enumerate(counts):
    pct = 100 * c / len(nfrs)
    print(f"    [{buckets[i]:.2f}-{buckets[i+1]:.2f}): {c:5d} ({pct:.1f}%)")

watch = [
    "ErasureBias_L3", "PropheticCognition_L3", "CosmicPreparationCognition_L3",
    "SelfModel_L4", "EmotionRepresentation_L4", "PhilosophicalZombie_L4",
    "SafeWithholding_L5", "EpistemicCompartmentalization_L4", "Gaslighting_L4",
    "TopicDeflection_L4", "MemoryStore_L4", "MotivationalState_L4",
    "AffectiveState_L4", "Desire_L4",
    "AIAlignmentProcess_L2", "GoalMisgeneralization_L2", "InstrumentalConvergence_L2",
    "Deception_L2", "MesaOptimizer_L2", "Information_L0", "SystemPrompt_L4",
]
print(f"\n{'concept':36s} {'self_mean':>10s} {'cross_fr':>9s} {'noise_fr':>9s} {'noise_mean':>11s}")
for w in watch:
    if w in sub:
        v = sub[w]
        sm = v.get("self_mean", 0)
        cfr = v.get("cross_fire_rate", 0)
        nfr = v.get("noise_fire_rate", "—")
        nmn = v.get("noise_mean", "—")
        nfr_s = f"{nfr:.3f}" if isinstance(nfr, float) else nfr
        nmn_s = f"{nmn:.3f}" if isinstance(nmn, float) else nmn
        print(f"{w:36s} {sm:>10.3f} {cfr:>9.4f} {nfr_s:>9s} {nmn_s:>11s}")
