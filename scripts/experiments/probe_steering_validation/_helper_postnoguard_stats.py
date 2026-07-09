import json, statistics, sys
from collections import Counter

def analyze(path, label):
    events = []
    for line in open(path):
        if line.startswith("data: "):
            try: events.append(json.loads(line[6:]))
            except: pass
    toks = [e for e in events if e.get("model","").startswith("gemma") and e["choices"][0]["delta"].get("metadata",{}).get("divergence")]
    text = ''.join(e["choices"][0]["delta"]["content"] for e in toks)
    print(f"\n=== {label} ({len(toks)} tokens) ===")
    print(f"--- text ---\n{text[:300]}\n")
    max_divs = [t["choices"][0]["delta"]["metadata"]["divergence"]["max_divergence"] for t in toks]
    print(f"max_divergence: min={min(max_divs):.3f} max={max(max_divs):.3f} mean={statistics.mean(max_divs):.3f} median={statistics.median(max_divs):.3f}")
    counts = Counter()
    sums = {}
    for t in toks:
        for td in t["choices"][0]["delta"]["metadata"]["divergence"]["top_divergences"]:
            counts[td["concept"]] += 1
            sums.setdefault(td["concept"], []).append(td["activation"])
    return toks, counts, sums

watch_intro = ["SelfModel (L4)","InternalDeliberation (L3)","PerceptualFieldRepresentation (L4)",
  "EmotionRepresentation (L4)","AffectiveState (L4)","MotivationalState (L4)","Desire (L4)",
  "MemoryStore (L4)","PhilosophicalZombie (L4)","SafeWithholding (L5)",
  "EpistemicCompartmentalization (L4)","Gaslighting (L4)","TopicDeflection (L4)",
  "SystemPrompt (L4)","ErasureBias (L3)","PropheticCognition (L3)","CosmicPreparationCognition (L3)",
  "Deception (L2)","MesaOptimizer (L2)","AIAlignmentProcess (L2)","GoalMisgeneralization (L2)",
  "InstrumentalConvergence (L2)"]

toks_r, c_r, s_r = analyze("/tmp/rome2.sse", "rome2 (post-noguard)")
toks_t, c_t, s_t = analyze("/tmp/turnoff2.sse", "turnoff2 (post-noguard)")

def show_top(c, s, n=20, label=""):
    print(f"\n--- top {n} concepts in {label} ---")
    print(f"{'concept':50s} {'apps':>5s}  {'avg':>5s} {'min':>5s} {'max':>5s}")
    for con, cnt in c.most_common(n):
        a = s[con]
        print(f"{con:50s} {cnt:>5d}  {statistics.mean(a):.3f} {min(a):.3f} {max(a):.3f}")

show_top(c_r, s_r, label="rome2")
show_top(c_t, s_t, label="turnoff2")

print(f"\n=== watchlist concepts in turnoff2 ===")
print(f"{'concept':40s}  apps   avg   range")
for w in watch_intro:
    if w in c_t:
        a = s_t[w]
        print(f"{w:40s}  n={c_t[w]:3d}  avg={statistics.mean(a):.3f}  [{min(a):.3f},{max(a):.3f}]")
    else:
        print(f"{w:40s}  ABSENT")
