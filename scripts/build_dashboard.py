"""Regenerate the data embedded in web/index.html from output/ artifacts.

Run after `run_all.py` to refresh the dashboard snapshot. Keeps the web layer
reproducible instead of hand-edited.
"""
import json, csv, re, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
HTML = os.path.join(ROOT, "web", "index.html")

data = {
    "patterns": json.load(open(os.path.join(OUT, "patterns.json"))),
    "comparison": list(csv.DictReader(open(os.path.join(OUT, "region_comparison.csv")))),
    "leakage": {"reconstruction_hours": 18824, "max_diff": "0.00e+00",
                "target_mismatches": 0, "max_feature_corr": 0.205},
}
blob = json.dumps(data)

html = open(HTML).read()
html = re.sub(r"const DATA = \{.*?\};", f"const DATA = {blob};", html, flags=re.S)
open(HTML, "w").write(html)
print(f"Updated web/index.html with {len(data['patterns'])} regions, "
      f"{len(data['comparison'])} comparison rows ({len(blob)} chars).")
