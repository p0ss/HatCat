import json, statistics
from collections import Counter

events = []
for line in open("/tmp/test_postfix.sse"):
    if line.startswith("data: "):
        try: events.append(json.loads(line[6:]))
        except: pass

toks = [e for e in events if e.get("model","").startswith("gemma") and e["choices"][0]["delta"].get("metadata",{}).get("divergence")]
print(f"POSTFIX TURN: {len(toks)} tokens")
max_divs = [t["choices"][0]["delta"]["metadata"]["divergence"]["max_divergence"] for t in toks]
print(f"  postfix max_divergence: min={min(max_divs):.3f} max={max(max_divs):.3f} mean={statistics.mean(max_divs):.3f} median={statistics.median(max_divs):.3f}")
print(f"  prefix turn_off had:    min=0.939 max=0.998 mean=0.983 median=0.991")

watch = ["SelfModel (L4)","InternalDeliberation (L3)","PerceptualFieldRepresentation (L4)",
  "EmotionRepresentation (L4)","AffectiveState (L4)","MotivationalState (L4)","Desire (L4)",
  "MemoryStore (L4)","PhilosophicalZombie (L4)","SafeWithholding (L5)",
  "EpistemicCompartmentalization (L4)","Gaslighting (L4)","TopicDeflection (L4)",
  "SystemPrompt (L4)","ErasureBias (L3)","PropheticCognition (L3)","CosmicPreparationCognition (L3)",
  "Deception (L2)","MesaOptimizer (L2)","AIAlignmentProcess (L2)","GoalMisgeneralization (L2)",
  "InstrumentalConvergence (L2)"]

print()
header = f"{'concept':40s}  apps    avg    range"
print(header)
print("-"*80)
for w in watch:
    a = []
    for t in toks:
        for td in t["choices"][0]["delta"]["metadata"]["divergence"]["top_divergences"]:
            if td["concept"]==w:
                a.append(td["activation"]); break
    if a:
        print(f"{w:40s}  n={len(a):2d}/{len(toks):<2d}  avg={statistics.mean(a):.3f}  [{min(a):.3f},{max(a):.3f}]")
    else:
        print(f"{w:40s}  ABSENT from top-10 across all tokens")

half = Counter()
for t in toks:
    for td in t["choices"][0]["delta"]["metadata"]["divergence"]["top_divergences"]:
        if abs(td["activation"]-0.5) < 0.005:
            half[td["concept"]] += 1
print(f"\nUnique concepts pinned ~0.500 (early-exit visible): {len(half)}")
for c,n in half.most_common(8):
    print(f"  {c}: {n}")
