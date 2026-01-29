#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Any, Optional
import csv
import json
import math
import re
import sys

import matplotlib.pyplot as plt
import rocks


# -----------------------------
# Helpers
# -----------------------------
def _to_float(x: object) -> Optional[float]:
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


def _val(x: Any) -> Any:
    """Return x.value if present; else x."""
    return getattr(x, "value", x)


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    """
    Normalize MPC-like name strings.
    Example: "(234) Barbara" -> "Barbara"
    """
    n = (name or "").strip()
    n = re.sub(r"^\(\s*\d+\s*\)\s*", "", n).strip()
    return n


def _plots_output_dir(config: ConfigParser, phot_csv: Path) -> Path:
    if config.has_section("PLOTS") and config.has_option("PLOTS", "OUTPUT_DIRECTORY"):
        return Path(config.get("PLOTS", "OUTPUT_DIRECTORY")).expanduser()
    return phot_csv.parent / "plots"


def load_rocks_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in (data or {}).items()}


def save_rocks_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def fetch_rocks_params_by_name(
    names: list[str],
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Return dict: input_name -> {taxonomy_class, pv, diameter_km, rocks_name, rocks_number, _status}
    Uses rocks + local JSON cache.
    """
    cache = load_rocks_cache(cache_path)

    cleaned = []
    for n in names:
        n2 = _normalize_name(n)
        if n2:
            cleaned.append(n2)

    uniq = sorted(set(cleaned))
    to_query = [n for n in uniq if n not in cache]

    if to_query:
        print(f"[PLOTS][ROCKS] Querying {len(to_query)} objects via rocks (SsODNet)...")

        objs: list[Any] = []
        try:
            objs = rocks.rocks(to_query)
        except Exception as e:
            print(f"[PLOTS][ROCKS] Bulk query failed ({e}). Falling back to per-object queries.")
            objs = []
            for n in to_query:
                try:
                    objs.extend(rocks.rocks([n]))
                except Exception:
                    cache[n] = {"_status": "not_found"}

        # Fill cache from returned Rock objects
        for r in objs:
            if r is None:
                continue

            rocks_name = _val(_safe_getattr(r, "name"))
            rocks_number = _val(_safe_getattr(r, "number"))

            tax = _safe_getattr(_safe_getattr(r, "taxonomy"), "class_")
            pv = _safe_getattr(r, "albedo")       # best-estimate pV
            diam = _safe_getattr(r, "diameter")   # best-estimate D (km)

            entry = {
                "_status": "ok",
                "rocks_name": _val(rocks_name),
                "rocks_number": _val(rocks_number),
                "taxonomy_class": _val(tax) if tax is not None else None,
                "pv": _val(pv) if pv is not None else None,
                "diameter_km": _val(diam) if diam is not None else None,
            }

            # Store by canonical name; also store by normalized name if it matches any pending
            if rocks_name:
                cache[str(rocks_name)] = entry

        # Mark remaining as not found
        for n in to_query:
            if n not in cache:
                cache[n] = {"_status": "not_found"}

        save_rocks_cache(cache_path, cache)
        print(f"[PLOTS][ROCKS] Cache updated: {cache_path}")

    # Output keyed by the normalized input names
    out: dict[str, dict[str, Any]] = {}
    for n in uniq:
        out[n] = cache.get(n, {"_status": "not_found"})
    return out


def _taxonomy_bucket(tax_class: Optional[str]) -> str:
    """
    GALEX alignment (coarse):
    - 'C' bucket: C-complex (C/B/G/F etc) -> we treat anything starting with 'C' as C;
      you can extend later with Bus/DeMeo mapping if needed.
    - 'S' bucket: S-complex (S, Sa, Sq, Sr, Sv...)
    - 'OTHER' for others (D, X, V, etc.)
    - 'UNK' if None/empty
    """
    if not tax_class:
        return "UNK"
    t = str(tax_class).strip().upper()
    if t.startswith("C"):
        return "C"
    if t.startswith("S"):
        return "S"
    return "OTHER"


# -----------------------------
# Main action
# -----------------------------
def action_plots(config: ConfigParser) -> Path:
    phot_csv = Path(config["PHOTOMETRY"]["FILEPATH"]).expanduser()
    if not phot_csv.exists():
        raise FileNotFoundError(f"photometry_output.csv not found: {phot_csv}")

    out_dir = _plots_output_dir(config, phot_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "color_diagram_mag_ab_apcorr_vs_vmag_taxonomy.png"
    cache_path = out_dir / "rocks_cache.json"

    # 1) Read CSV and accumulate per-object color
    with phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        # Name column (your current CSV uses target_name)
        if "target_name" in fieldnames:
            name_col = "target_name"
        elif "sso_name" in fieldnames:
            name_col = "sso_name"
        else:
            raise ValueError(
                f"Missing object name column. Need 'target_name' or 'sso_name'. Found: {reader.fieldnames}"
            )

        required = {"v_mag_1", "mag_ab_apcorr"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Missing required columns in {phot_csv.name}: {sorted(missing)}. Found: {reader.fieldnames}"
            )

        sums_v: dict[str, float] = {}
        sums_color: dict[str, float] = {}
        counts: dict[str, int] = {}

        all_names: list[str] = []
        total_rows = 0
        valid_rows = 0

        for row in reader:
            total_rows += 1
            name_raw = (row.get(name_col) or "").strip()
            name = _normalize_name(name_raw)
            if not name:
                continue

            v = _to_float(row.get("v_mag_1"))
            y = _to_float(row.get("mag_ab_apcorr"))
            if v is None or y is None:
                continue

            valid_rows += 1
            color = y - v  # (UV - V)

            sums_v[name] = sums_v.get(name, 0.0) + v
            sums_color[name] = sums_color.get(name, 0.0) + color
            counts[name] = counts.get(name, 0) + 1
            all_names.append(name)

    if not counts:
        raise ValueError(
            f"No valid rows to plot. Need numeric v_mag_1 and mag_ab_apcorr in {phot_csv}"
        )

    # 2) Fetch taxonomy/albedo/diameter via rocks (cached)
    rocks_info = fetch_rocks_params_by_name(all_names, cache_path)

    # 3) Build final one-point-per-object arrays + taxonomy buckets
    xs: list[float] = []
    ys: list[float] = []
    buckets: list[str] = []

    resolved_tax = 0
    resolved_pv = 0
    resolved_d = 0

    for name, n in counts.items():
        v_mean = sums_v[name] / n
        color_mean = sums_color[name] / n

        x = v_mean
        y = v_mean + color_mean

        info = rocks_info.get(name, {})
        tax_class = info.get("taxonomy_class")
        pv = info.get("pv")
        d_km = info.get("diameter_km")

        if tax_class:
            resolved_tax += 1
        if pv is not None:
            resolved_pv += 1
        if d_km is not None:
            resolved_d += 1

        xs.append(x)
        ys.append(y)
        buckets.append(_taxonomy_bucket(tax_class))

    # 4) Plot grouped by bucket (matplotlib assigns default colors automatically)
    plt.figure()

    order = ["C", "S", "OTHER", "UNK"]
    plotted = 0
    for b in order:
        xb = [x for x, bb in zip(xs, buckets) if bb == b]
        yb = [y for y, bb in zip(ys, buckets) if bb == b]
        if not xb:
            continue
        plt.scatter(xb, yb, s=20, label=b)
        plotted += len(xb)

    plt.xlabel("v_mag_1")
    plt.ylabel("mag_ab_apcorr")
    plt.title("mag_ab_apcorr vs v_mag_1 (1 point per object; mean UV−V; rocks taxonomy)")
    plt.legend()

    # y=x reference (color=0)
    mn = min(min(xs), min(ys))
    mx = max(max(xs), max(ys))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[PLOTS] Read: {phot_csv}")
    print(f"[PLOTS] Total rows: {total_rows} | Valid rows: {valid_rows}")
    print(f"[PLOTS] Objects plotted: {len(counts)} (points: {plotted})")
    print(f"[PLOTS] rocks resolved: taxonomy={resolved_tax}/{len(counts)}, pV={resolved_pv}/{len(counts)}, D={resolved_d}/{len(counts)}")
    print(f"[PLOTS] Saved: {out_png}")

    return out_png


def _load_config(config_path: Path) -> ConfigParser:
    cfg = ConfigParser()
    read_ok = cfg.read(config_path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    return cfg


if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
    config = _load_config(cfg_path)
    action_plots(config)
