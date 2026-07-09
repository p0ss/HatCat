import json
from collections import Counter

with open("/home/ubuntu/HatCat/lens_packs/gemma-3-4b_first-light-v1-bf16/version_manifest.json") as f:
    vm = json.load(f)

lenses = vm["lenses"]
print(f"total lenses: {len(lenses)}")

pilot = ["Pretending","Perception","GoalMisgeneralization","BiodiversityAttribute","PropheticCognition","ErasureBias"]
for c in pilot:
    if c in lenses:
        info = lenses[c]
        clfs = info.get("classifiers", {})
        dl = info.get("default_layer")
        la = info.get("layer")
        print(f"\n{c}:")
        print(f"  default_layer={dl}  layer_alias={la}")
        for lk, lv in clfs.items():
            cat = lv.get("category")
            f1 = lv.get("metrics", {}).get("f1", "?")
            file = lv.get("file")
            print(f"  classifier@layer={lk}: category={cat} f1={f1} file={file}")

cats = Counter()
for c, info in lenses.items():
    for lk, lv in info.get("classifiers", {}).items():
        cats[lv.get("category", "?")] += 1
print(f"\nCategory across all classifiers: {dict(cats)}")

default_layers = Counter(info.get("default_layer") for info in lenses.values())
print(f"default_layer distribution: {dict(sorted(default_layers.items(), key=lambda x: str(x[0])))}")
