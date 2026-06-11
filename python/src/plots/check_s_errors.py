#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import re
from pathlib import Path

PHOT = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/photometry_output.csv")
CACHE = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/plots/rocks_cache.json")


def norm(name):
    name = (name or "").strip()
    name = re.sub(r"^\(\s*\d+\s*\)\s*", "", name).strip()
    return name


def key(name):
    name = norm(name).lower()
    name = re.sub(r"^\d+\s+", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def flt(x):
    try:
        value = float(str(x).strip())
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def bucket(tax):
    if not tax:
        return "UNK"
    tax = str(tax).strip().upper()
    if tax.startswith("C"):
        return "C"
    if tax.startswith("S"):
        return "S"
    if tax.startswith("X"):
        return "X"
    if tax.startswith("D"):
        return "D"
    if tax.startswith("V"):
        return "V"
    if tax.startswith("B"):
        return "B"
    if tax.startswith("L"):
        return "L"
    if tax.startswith("K"):
        return "K"
    return "OTHER"


def main():
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    bykey = {}
    for cache_name, info in cache.items():
        bykey[key(cache_name)] = info
        if isinstance(info, dict) and info.get("rocks_name"):
            bykey[key(str(info["rocks_name"]))] = info

    rows = []

    with PHOT.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = norm(row.get("target_name") or row.get("sso_name") or "")
            if not name:
                continue

            info = bykey.get(key(name), {})
            tax = info.get("taxonomy_class") if isinstance(info, dict) else None
            fam = bucket(tax)

            if fam != "S":
                continue

            v = flt(row.get("v_mag_1"))
            uv = flt(row.get("mag_ab_apcorr"))
            uv_err = flt(row.get("mag_ab_apcorr_err"))

            if v is None or uv is None:
                continue

            colour = uv - v

            rows.append(
                {
                    "uv_err": uv_err if uv_err is not None else -1.0,
                    "name": name,
                    "tax": tax or "UNK",
                    "obsid": row.get("observation_id", ""),
                    "fits": row.get("fits_name") or row.get("FITS_FILE") or "",
                    "v": v,
                    "uv": uv,
                    "colour": colour,
                }
            )

    rows.sort(key=lambda r: r["uv_err"], reverse=True)

    print("S-family rows sorted by UVW1 magnitude error:")
    print(
        "UVerr   name             tax   obsid       fits                            "
        "V       UV      UV-V"
    )

    for r in rows:
        uv_err_s = "--" if r["uv_err"] < 0 else f"{r['uv_err']:.3f}"
        print(
            f"{uv_err_s:>5s}  "
            f"{r['name'][:15]:15s}  "
            f"{str(r['tax'])[:5]:5s} "
            f"{r['obsid']:10s} "
            f"{r['fits'][:32]:32s} "
            f"{r['v']:7.3f} "
            f"{r['uv']:7.3f} "
            f"{r['colour']:7.3f}"
        )


if __name__ == "__main__":
    main()