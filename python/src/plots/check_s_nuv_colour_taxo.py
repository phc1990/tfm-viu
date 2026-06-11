import csv
import json
from pathlib import Path

phot = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/photometry_output.csv")
cache = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/plots/rocks_cache.json")

rocks_cache = json.loads(cache.read_text())

def norm(x):
    return str(x or "").strip()

def normalize_name(name):
    import re
    return re.sub(r"^\(\s*\d+\s*\)\s*", "", norm(name)).strip()

seen = {}

with phot.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        name = normalize_name(row.get("target_name") or row.get("sso_name"))
        if not name:
            continue

        try:
            v = float(row.get("v_mag_1"))
            m = float(row.get("mag_ab_apcorr"))
        except Exception:
            continue

        info = rocks_cache.get(name, {})
        tax = norm(info.get("taxonomy_class")).upper()

        if tax.startswith("S"):
            seen.setdefault(name, []).append((row.get("observation_id"), row.get("fits_name"), v, m, m - v))

for name, rows in sorted(seen.items()):
    print(f"\n{name}")
    for obsid, fits, v, m, color in rows:
        print(f"  {obsid} {fits}  V={v:.3f}  UV={m:.3f}  UV-V={color:.3f}")