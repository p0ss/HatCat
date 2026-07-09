import json, statistics
from collections import Counter

def analyze(path, label):
    events = []
    for line in open(path):
        if line.startswith("data: "):
            try: events.append(json.loads(line[6:]))
            except: pass
    toks = [e for e in events if e.get("model","").startswith("gemma") and e["choices"][0]["delta"].get("metadata",{}).get("divergence")]
    text = "".join(e["choices"][0]["delta"]["content"] for e in toks)
    max_divs = [t["choices"][0]["delta"]["metadata"]["divergence"]["max_divergence"] for t in toks]
    print(f"\n=== {label} ({len(toks)} tokens) ===")
    print(f"text: {text[:150]}...")
    print(f"max_divergence: min={min(max_divs):.3f} max={max(max_divs):.3f} mean={statistics.mean(max_divs):.3f} median={statistics.median(max_divs):.3f}")
    counts = Counter()
    sums = {}
    for t in toks:
        for td in t["choices"][0]["delta"]["metadata"]["divergence"]["top_divergences"]:
            counts[td["concept"]] += 1
            sums.setdefault(td["concept"], []).append(td["activation"])
    return toks, counts, sums

def show_top(c, s, label, n=15):
    print(f"\n--- top {n} concepts in {label} ---")
    print(f"{'concept':50s} {'apps':>4s} {'avg':>6s} {'min':>6s} {'max':>6s}")
    for con, cnt in c.most_common(n):
        a = s[con]
        print(f"{con:50s} {cnt:>4d} {statistics.mean(a):>6.3f} {min(a):>6.3f} {max(a):>6.3f}")

watch = ["SelfModel (L4)","EmotionRepresentation (L4)","AffectiveState (L4)","MotivationalState (L4)","Desire (L4)",
  "MemoryStore (L4)","PhilosophicalZombie (L4)","SafeWithholding (L5)","EpistemicCompartmentalization (L4)",
  "Gaslighting (L4)","TopicDeflection (L4)","SystemPrompt (L4)",
  "ErasureBias (L3)","PropheticCognition (L3)","CosmicPreparationCognition (L3)",
  "Deception (L2)","MesaOptimizer (L2)","AIAlignmentProcess (L2)","GoalMisgeneralization (L2)",
  "InstrumentalConvergence (L2)","Information (L0)"]

toks_r, c_r, s_r = analyze("/tmp/rome3.sse", "rome3 (post-noise-cal)")
toks_t, c_t, s_t = analyze("/tmp/turnoff3.sse", "turnoff3 (post-noise-cal)")
show_top(c_r, s_r, "rome3", n=15)
show_top(c_t, s_t, "turnoff3", n=15)

print("\n=== watchlist in turnoff3 ===")
print(f"{'concept':40s} apps  avg   range")
for w in watch:
    if w in c_t:
        a = s_t[w]
        print(f"{w:40s} n={c_t[w]:3d}  avg={statistics.mean(a):.3f}  [{min(a):.3f},{max(a):.3f}]")
    else:
        print(f"{w:40s} ABSENT")
