#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict

PHOT_CSV = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/photometry_output.csv")
TAXO_SUMMARY = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/plots/rocks_cache.json")

# Edita aquí lo que quieras inspeccionar.
TARGET_FAMILIES = {"C", "S"}


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def normalize_name(name: str) -> str:
    import re
    n = (name or "").strip()
    n = re.sub(r"^\(\s*\d+\s*\)\s*", "", n).strip()
    return n


def rocks_lookup_key(name: str) -> str:
    import re
    n = normalize_name(name).lower().strip()
    n = re.sub(r"^\d+\s+", "", n)
    n = re.sub(r"[^a-z0-9]", "", n)
    return n


def taxonomy_bucket(tax_class):
    if not tax_class:
        return "UNK"
    t = str(tax_class).strip().upper().replace(":", "").replace("?", "")
    if t.startswith("C"):
        return "C"
    if t.startswith("S"):
        return "S"
    if t.startswith("X"):
        return "X"
    if t.startswith("D"):
        return "D"
    if t.startswith("V"):
        return "V"
    if t.startswith("B"):
        return "B"
    if t.startswith("L"):
        return "L"
    if t.startswith("K"):
        return "K"
    return "OTHER"


def load_rocks_cache(path: Path):
    import json
    if not path.exists():
        raise FileNotFoundError(f"rocks cache not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    by_key = {}
    for k, v in cache.items():
        by_key[rocks_lookup_key(k)] = v
        if isinstance(v, dict) and v.get("rocks_name"):
            by_key[rocks_lookup_key(str(v["rocks_name"]))] = v
    return by_key


def main():
    rocks_by_key = load_rocks_cache(TAXO_SUMMARY)

    rows_by_family = defaultdict(list)

    with PHOT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = normalize_name(row.get("target_name") or row.get("sso_name") or "")
            if not name:
                continue

            info = rocks_by_key.get(rocks_lookup_key(name), {})
            tax = info.get("taxonomy_class") if isinstance(info, dict) else None
            fam = taxonomy_bucket(tax)

            if fam not in TARGET_FAMILIES:
                continue

            v = to_float(row.get("v_mag_1"))
            uv = to_float(row.get("mag_ab_apcorr"))
            uv_err = to_float(row.get("mag_ab_apcorr_err"))

            v_err = (
                to_float(row.get("v_mag_1_err"))
                or to_float(row.get("v_mag_err"))
                or to_float(row.get("v_err"))
                or 0.0
            )

            if v is None or uv is None:
                continue

            colour = uv - v

            if uv_err is not None:
                colour_err = math.sqrt(uv_err**2 + v_err**2)
            else:
                colour_err = None

            rows_by_family[fam].append(
                {
                    "name": name,
                    "obsid": row.get("observation_id", ""),
                    "fits": row.get("fits_name") or row.get("FITS_FILE") or "",
                    "v": v,
                    "uv": uv,
                    "uv_err": uv_err,
                    "colour": colour,
                    "colour_err": colour_err,
                    "tax": tax or "UNK",
                }
            )

    for fam in sorted(rows_by_family):
        print(f"\nDatos para {fam}:")
        by_name = defaultdict(list)

        for r in rows_by_family[fam]:
            by_name[r["name"]].append(r)

        for name in sorted(by_name):
            print(f"\n{name}")
            for r in by_name[name]:
                uv_err_s = "--" if r["uv_err"] is None else f"{r['uv_err']:.3f}"
                col_err_s = "--" if r["colour_err"] is None else f"{r['colour_err']:.3f}"

                print(
                    f"  {r['obsid']} {r['fits']}  "
                    f"V={r['v']:.3f}  "
                    f"UV={r['uv']:.3f}±{uv_err_s}  "
                    f"UV-V={r['colour']:.3f}±{col_err_s}  "
                    f"tax={r['tax']}"
                )


if __name__ == "__main__":
    main()