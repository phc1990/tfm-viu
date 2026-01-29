#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Optional
import csv
import math
import sys

import matplotlib.pyplot as plt

from typing import Any, Optional
import json

import rocks


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


def _plots_output_dir(config: ConfigParser, phot_csv: Path) -> Path:
    # Prefer [PLOTS] OUTPUT_DIRECTORY, else sibling 'plots'
    if config.has_section("PLOTS") and config.has_option("PLOTS", "OUTPUT_DIRECTORY"):
        return Path(config.get("PLOTS", "OUTPUT_DIRECTORY")).expanduser()
    return phot_csv.parent / "plots"


def action_plots(config: ConfigParser) -> Path:
    """
    Plot y=mag_ab_apcorr vs x=v_mag_1.

    If multiple rows for the same sso_name, average the color:
        color = (mag_ab_apcorr - v_mag_1)
    Then plot one point per sso_name:
        x = mean(v_mag_1)
        y = x + mean(color)
    """
    phot_csv = Path(config["PHOTOMETRY"]["FILEPATH"]).expanduser()
    if not phot_csv.exists():
        raise FileNotFoundError(f"photometry_output.csv not found: {phot_csv}")

    out_dir = _plots_output_dir(config, phot_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "color_diagram_mag_ab_apcorr_vs_vmag.png"

    cache_path = out_dir / "rocks_cache.json"

    # ... mientras parseas el CSV, guarda una lista con los nombres únicos:
    all_names = []  # llena con target_name

    # Una vez los tienes:
    rocks_info = fetch_rocks_params_by_name(all_names, cache_path)

    # Ejemplo de uso por nombre:
    info = rocks_info.get(name, {})
    tax_class = info.get("taxonomy_class")   # 'C', 'S', etc o None
    pv = info.get("pv")                      # float o None
    d_km = info.get("diameter_km")           # float o None


    sums_v: dict[str, float] = {}
    sums_color: dict[str, float] = {}
    counts: dict[str, int] = {}

    with phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"target_name", "v_mag_1", "mag_ab_apcorr"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {phot_csv.name}: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            name = (row.get("target_name") or "").strip()
            if not name:
                continue

            v = _to_float(row.get("v_mag_1"))
            y = _to_float(row.get("mag_ab_apcorr"))
            if v is None or y is None:
                continue

            color = y - v  # (UV - V)

            sums_v[name] = sums_v.get(name, 0.0) + v
            sums_color[name] = sums_color.get(name, 0.0) + color
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        raise ValueError(
            f"No valid rows to plot. Need numeric v_mag_1 and mag_ab_apcorr in {phot_csv}"
        )

    xs: list[float] = []
    ys: list[float] = []
    for name, n in counts.items():
        v_mean = sums_v[name] / n
        color_mean = sums_color[name] / n
        xs.append(v_mean)
        ys.append(v_mean + color_mean)

    plt.figure()
    plt.scatter(xs, ys, s=20)
    plt.xlabel("v_mag_1")
    plt.ylabel("mag_ab_apcorr")
    plt.title("mag_ab_apcorr vs v_mag_1 (1 point per target; mean color)")

    # y=x reference line (color=0)
    mn = min(min(xs), min(ys))
    mx = max(max(xs), max(ys))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[PLOTS] Read: {phot_csv}")
    print(f"[PLOTS] Objects plotted: {len(xs)}")
    print(f"[PLOTS] Saved: {out_png}")
    return out_png


def _load_config(config_path: Path) -> ConfigParser:
    cfg = ConfigParser()
    read_ok = cfg.read(config_path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    return cfg



def _val(x: Any) -> Any:
    """Devuelve x.value si existe; si no, x."""
    return getattr(x, "value", x)


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def load_rocks_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normaliza claves a str
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
    Devuelve dict: input_name -> {taxonomy_class, pv, diameter_km, rocks_name, rocks_number}
    usando rocks + cache local (json).
    """
    cache = load_rocks_cache(cache_path)

    # Normaliza nombres de entrada (mantén el original para mapear, pero usa cleaned para query)
    cleaned: list[str] = []
    for n in names:
        n2 = (n or "").strip()
        if n2:
            cleaned.append(n2)

    # Sólo consulta lo que no esté en cache
    to_query = [n for n in sorted(set(cleaned)) if n not in cache]

    if to_query:
        print(f"[PLOTS][ROCKS] Querying {len(to_query)} objects from SsODNet via rocks...")
        # rocks.rocks() hace bulk query; algunos pueden fallar -> manejamos exceptions por item
        # Nota: rocks puede lanzar si algún nombre no existe; lo encapsulamos.
        try:
            objs = rocks.rocks(to_query)
        except Exception as e:
            # fallback: intentar uno-a-uno para no perder todo el batch
            print(f"[PLOTS][ROCKS] Bulk query failed ({e}). Falling back to per-object queries.")
            objs = []
            for n in to_query:
                try:
                    objs.extend(rocks.rocks([n]))
                except Exception:
                    # marca como no resuelto
                    cache[n] = {"_status": "not_found"}
            objs = [o for o in objs if o is not None]

        # Rellena cache con lo que venga resuelto
        for r in objs:
            # r.name suele ser el nombre “canónico” en SsODNet (puede diferir de input)
            rocks_name = _val(_safe_getattr(r, "name"))
            rocks_number = _val(_safe_getattr(r, "number"))

            tax = _safe_getattr(_safe_getattr(r, "taxonomy"), "class_")
            pv = _safe_getattr(r, "albedo")        # best-estimate pV (si existe)
            diam = _safe_getattr(r, "diameter")    # best-estimate D (km si existe)

            entry = {
                "_status": "ok",
                "rocks_name": _val(rocks_name),
                "rocks_number": _val(rocks_number),
                "taxonomy_class": _val(tax) if tax is not None else None,
                "pv": _val(pv) if pv is not None else None,
                "diameter_km": _val(diam) if diam is not None else None,
            }

            # ¿Cómo mapeamos al input?
            # rocks no siempre conserva el input; así que guardamos por rocks_name y por rocks_number si hay.
            # Pero como tu llave ahora mismo es sso_name, guardamos por rocks_name.
            if rocks_name:
                cache[str(rocks_name)] = entry

        # Marca explícitamente “not found” los que sigan sin estar
        for n in to_query:
            if n not in cache:
                cache[n] = {"_status": "not_found"}

        save_rocks_cache(cache_path, cache)
        print(f"[PLOTS][ROCKS] Cache updated: {cache_path}")

    # Construye salida con las claves exactas de entrada
    out: dict[str, dict[str, Any]] = {}
    for n in cleaned:
        out[n] = cache.get(n, {"_status": "not_found"})
    return out



if __name__ == "__main__":
    # Standalone usage:
    #   python action_plots.py /path/to/config.ini
    # If omitted, default to ./config.ini
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
    config = _load_config(cfg_path)
    action_plots(config)
