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
import numpy as np



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

def _rocks_lookup_key(name: str) -> str:
    """
    Aggressive key for matching photometry names against rocks/cache names.
    Removes MPC number, spaces, punctuation and case.
    """
    n = _normalize_name(name)
    n = n.lower().strip()
    n = re.sub(r"^\d+\s+", "", n)
    n = re.sub(r"[^a-z0-9]", "", n)
    return n

def _normalize_name(name: str) -> str:
    """
    Normalize MPC-like name strings.
    Example: "(234) Barbara" -> "Barbara"
    """
    n = (name or "").strip()
    n = re.sub(r"^\(\s*\d+\s*\)\s*", "", n).strip()
    return n


# def _latex_header_name(c: str) -> str:
#     return {
#         "target_name": "Asteroid",
#         "observation_id": "Obs. ID",
#         "fits_name": "FITS file",
#         "count_rate": r"$c$",
#         "count_rate_err": r"$\sigma_c$",
#         "trail_height_pix": r"$h_{\rm trail}$",
#         "mag_ab_apcorr": r"$m_{\rm AB}$",
#         "mag_ab_apcorr_err": r"$\sigma_m$",
#         "v_mag_1": r"$V_{\rm pred}$",
#         "mlim_obs": r"$m_{\rm lim}$",
#     }.get(c, c)
def _latex_header_name(c: str) -> str:
    return {
        "target_name": "Asteroid",
        "observation_id": "Obs. ID",
        "fits_name": "Frame",
        "count_rate": r"$c$ (ct s$^{-1}$)",
        "mag_ab": r"$m_{\rm AB}$",
        "mag_ab_apcorr": r"$m_{\rm AB,apcorr}$",
        "v_mag_1": r"$V_{\rm pred}$",
        "filter": "Filter",
    }.get(c, c)

def _plots_output_dir(config: ConfigParser, phot_csv: Path) -> Path:
    if config.has_section("PLOTS") and config.has_option("PLOTS", "OUTPUT_DIRECTORY"):
        return Path(config.get("PLOTS", "OUTPUT_DIRECTORY")).expanduser()
    return phot_csv.parent / "plots"


def _iter_noncomment_lines(f):
    """Yield CSV lines, ignoring rows disabled with a leading '#'."""
    for line in f:
        if line.lstrip().startswith("#"):
            continue
        yield line


def _reflectance_settings(config: ConfigParser) -> tuple[float, float]:
    """Return (solar UVW1_AB-V colour, empirical Vpred scatter in mag)."""
    solar_color = config.getfloat(
        "REFLECTANCE",
        "SOLAR_UVW1_AB_MINUS_V",
        fallback=2.8466,
    )
    vpred_scatter = config.getfloat(
        "REFLECTANCE",
        "VPRED_VALIDATION_SCATTER_MAG",
        fallback=0.367,
    )
    return float(solar_color), float(vpred_scatter)


def compute_uv_reflectance(
    m_uv: float,
    m_uv_err: Optional[float],
    v_pred: float,
    solar_color: float,
    vpred_scatter_mag: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """
    Compute solar-normalised OM/UVW1 reflectance relative to V.

    R_UVW1/V = 10**[-0.4 * ((m_UVW1,AB - Vpred) - solar_color)]

    ``reflectance_err_phot`` propagates only the formal UVW1 photometric
    uncertainty (and a formal V uncertainty if it has already been folded into
    ``m_uv_err`` by the caller).  The Vpred validation scatter is kept separate
    and returned as asymmetric errors because the magnitude-to-reflectance
    transformation is exponential.
    """
    colour = float(m_uv) - float(v_pred)
    delta_solar = colour - float(solar_color)
    reflectance = 10.0 ** (-0.4 * delta_solar)

    k = 0.4 * math.log(10.0)
    rerr_phot: Optional[float] = None
    if m_uv_err is not None and math.isfinite(m_uv_err) and m_uv_err >= 0:
        rerr_phot = k * reflectance * float(m_uv_err)

    rerr_ext_lo: Optional[float] = None
    rerr_ext_hi: Optional[float] = None
    if vpred_scatter_mag is not None and math.isfinite(vpred_scatter_mag) and vpred_scatter_mag >= 0:
        sigma_mag = math.sqrt((float(m_uv_err or 0.0)) ** 2 + float(vpred_scatter_mag) ** 2)
        r_low = reflectance * 10.0 ** (-0.4 * sigma_mag)
        r_high = reflectance * 10.0 ** (+0.4 * sigma_mag)
        rerr_ext_lo = reflectance - r_low
        rerr_ext_hi = r_high - reflectance

    return {
        "uv_minus_v": colour,
        "delta_uv_solar": delta_solar,
        "reflectance_uv_v": reflectance,
        "reflectance_err_phot": rerr_phot,
        "reflectance_err_external_lo": rerr_ext_lo,
        "reflectance_err_external_hi": rerr_ext_hi,
    }


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


# def fetch_rocks_params_by_name(
#     names: list[str],
#     cache_path: Path,
# ) -> dict[str, dict[str, Any]]:
#     """
#     Return dict: input_name -> {taxonomy_class, pv, diameter_km, rocks_name, rocks_number, _status}
#     Uses rocks + local JSON cache.
#     """
#     cache = load_rocks_cache(cache_path)

#     cleaned = []
#     for n in names:
#         n2 = _normalize_name(n)
#         if n2:
#             cleaned.append(n2)

#     uniq = sorted(set(cleaned))
#     to_query = [n for n in uniq if n not in cache]

#     if to_query:
#         print(f"[PLOTS][ROCKS] Querying {len(to_query)} objects via rocks (SsODNet)...")

#         objs: list[Any] = []
#         try:
#             objs = rocks.rocks(to_query)
#         except Exception as e:
#             print(f"[PLOTS][ROCKS] Bulk query failed ({e}). Falling back to per-object queries.")
#             objs = []
#             for n in to_query:
#                 try:
#                     objs.extend(rocks.rocks([n]))
#                 except Exception:
#                     cache[n] = {"_status": "not_found"}

#         # Fill cache from returned Rock objects
#         for r in objs:
#             if r is None:
#                 continue

#             rocks_name = _val(_safe_getattr(r, "name"))
#             rocks_number = _val(_safe_getattr(r, "number"))

#             tax = _safe_getattr(_safe_getattr(r, "taxonomy"), "class_")
#             pv = _safe_getattr(r, "albedo")       # best-estimate pV
#             diam = _safe_getattr(r, "diameter")   # best-estimate D (km)

#             entry = {
#                 "_status": "ok",
#                 "rocks_name": _val(rocks_name),
#                 "rocks_number": _val(rocks_number),
#                 "taxonomy_class": _val(tax) if tax is not None else None,
#                 "pv": _val(pv) if pv is not None else None,
#                 "diameter_km": _val(diam) if diam is not None else None,
#             }

#             # Store by canonical name; also store by normalized name if it matches any pending
#             if rocks_name:
#                 cache[str(rocks_name)] = entry

#                 # Also store by normalized returned name.
#                 cache[_normalize_name(str(rocks_name))] = entry

#         # Mark remaining as not found
#         for n in to_query:
#             if n not in cache:
#                 print(f"[PLOTS][ROCKS] WARN: not resolved by rocks: {n}")
#                 cache[n] = {"_status": "not_found"}

#         save_rocks_cache(cache_path, cache)
#         print(f"[PLOTS][ROCKS] Cache updated: {cache_path}")

#     # Output keyed by the normalized input names
#     out: dict[str, dict[str, Any]] = {}
#     for n in uniq:
#         out[n] = cache.get(n, {"_status": "not_found"})
#     return out

def fetch_rocks_params_by_name(
    names: list[str],
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Return dict keyed by normalized input name.

    Robust against cache keys such as:
    - 'Aaltje'
    - '(677) Aaltje'
    - '677 Aaltje'
    - different case / punctuation
    """
    cache = load_rocks_cache(cache_path)

    cleaned = []
    for n in names:
        n2 = _normalize_name(n)
        if n2:
            cleaned.append(n2)

    uniq = sorted(set(cleaned))

    print(f"[PLOTS][ROCKS][DBG] Requested names: {len(uniq)}")
    print(f"[PLOTS][ROCKS][DBG] First requested names: {uniq[:15]}")

    # Build aggressive lookup from whatever is already in the cache.
    cache_by_key: dict[str, dict[str, Any]] = {}
    for k, v in cache.items():
        kk = _rocks_lookup_key(k)
        if kk:
            cache_by_key[kk] = v

        rocks_name = v.get("rocks_name") if isinstance(v, dict) else None
        if rocks_name:
            cache_by_key[_rocks_lookup_key(str(rocks_name))] = v

    def _has_good_cache_entry(n: str) -> bool:
        entry = cache.get(n) or cache_by_key.get(_rocks_lookup_key(n))
        if not isinstance(entry, dict):
            return False

        # Re-query not_found entries. They may come from a previous bad run.
        if entry.get("_status") == "not_found":
            return False

        return True

    to_query = [n for n in uniq if not _has_good_cache_entry(n)]

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
                    rr = rocks.rocks([n])
                    objs.extend(rr)
                except Exception:
                    cache[n] = {"_status": "not_found"}

        # Store returned objects both by rocks canonical name and by requested name
        for r in objs:
            if r is None:
                continue

            rocks_name = _val(_safe_getattr(r, "name"))
            rocks_number = _val(_safe_getattr(r, "number"))

            tax = _safe_getattr(_safe_getattr(r, "taxonomy"), "class_")
            pv = _safe_getattr(r, "albedo")
            diam = _safe_getattr(r, "diameter")

            entry = {
                "_status": "ok",
                "rocks_name": _val(rocks_name),
                "rocks_number": _val(rocks_number),
                "taxonomy_class": _val(tax) if tax is not None else None,
                "pv": _val(pv) if pv is not None else None,
                "diameter_km": _val(diam) if diam is not None else None,
            }

            if rocks_name:
                cache[str(rocks_name)] = entry
                cache_by_key[_rocks_lookup_key(str(rocks_name))] = entry

            # Attach this result to any pending requested name with same lookup key.
            rkey = _rocks_lookup_key(str(rocks_name or ""))
            for n in to_query:
                if _rocks_lookup_key(n) == rkey:
                    cache[n] = entry
                    cache_by_key[_rocks_lookup_key(n)] = entry

        # Mark only genuinely unresolved names
        for n in to_query:
            if n not in cache and _rocks_lookup_key(n) not in cache_by_key:
                cache[n] = {"_status": "not_found"}

        save_rocks_cache(cache_path, cache)
        print(f"[PLOTS][ROCKS] Cache updated: {cache_path}")

    # Output keyed exactly by the normalized input names
    out: dict[str, dict[str, Any]] = {}
    for n in uniq:
        entry = cache.get(n)
        if not entry:
            entry = cache_by_key.get(_rocks_lookup_key(n))
        if not entry:
            entry = {"_status": "not_found"}
        out[n] = entry

    return out
def _get_rocks_info(
    rocks_info: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    """
    Robust lookup in rocks_info.
    """
    if name in rocks_info:
        return rocks_info[name]

    key = _rocks_lookup_key(name)

    for k, v in rocks_info.items():
        if _rocks_lookup_key(k) == key:
            return v

        rocks_name = v.get("rocks_name") if isinstance(v, dict) else None
        if rocks_name and _rocks_lookup_key(str(rocks_name)) == key:
            return v

    return {}

# def _taxonomy_bucket(tax_class: Optional[str]) -> str:
#     """
#     GALEX alignment (coarse):
#     - 'C' bucket: C-complex (C/B/G/F etc) -> we treat anything starting with 'C' as C;
#       you can extend later with Bus/DeMeo mapping if needed.
#     - 'S' bucket: S-complex (S, Sa, Sq, Sr, Sv...)
#     - 'OTHER' for others (D, X, V, etc.)
#     - 'UNK' if None/empty
#     """
#     if not tax_class:
#         return "UNK"
#     t = str(tax_class).strip().upper()
#     if t.startswith("C"):
#         return "C"
#     if t.startswith("S"):
#         return "S"
#     return "OTHER"
def _taxonomy_bucket(tax_class: Optional[str]) -> str:
    """
    Coarse taxonomic family bucket.

    Main families are kept separated instead of collapsing everything
    into C/S/OTHER.
    """
    if not tax_class:
        return "UNK"

    t = str(tax_class).strip().upper()
    if not t:
        return "UNK"

    # Remove common separators / uncertainty markers
    t = t.replace(":", "").replace("?", "").strip()

    # Large taxonomic families.
    # Order matters for subclasses such as SQ, SA, XC, etc.
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
    if t.startswith("A"):
        return "A"
    if t.startswith("L"):
        return "L"
    if t.startswith("K"):
        return "K"
    if t.startswith("Q"):
        return "Q"

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
    out_png = out_dir / "color_diagram_mag_ab_vs_vmag_taxonomy.png"
    cache_path = out_dir / "rocks_cache.json"


    latex_columns = [
        "target_name",
        "observation_id",
        "fits_name",
        "count_rate",
        "mag_ab",
        "mag_ab_apcorr",
        "v_mag_1",
    ]

    out_tex = out_dir / "photometry_output_table.tex"

        # 1) Read CSV and accumulate per-object colour
    # Main science magnitude: mag_ab.
    # mag_ab_apcorr remains available for comparison/table, but is not required for the plot.
    mag_col = "mag_ab"
    mag_err_col = "mag_err"

    with phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = set(reader.fieldnames or [])

        # Name column
        if "target_name" in fieldnames:
            name_col = "target_name"
        elif "sso_name" in fieldnames:
            name_col = "sso_name"
        else:
            raise ValueError(
                f"Missing object name column. Need 'target_name' or 'sso_name'. "
                f"Found: {reader.fieldnames}"
            )

        required = {"v_mag_1", mag_col}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Missing required columns in {phot_csv.name}: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        measurements: dict[str, list[dict[str, Optional[float]]]] = {}
        all_table_names: list[str] = []

        total_rows = 0
        valid_rows = 0

        for row in reader:
            total_rows += 1

            name_raw = (row.get(name_col) or "").strip()
            name = _normalize_name(name_raw)
            if not name:
                continue

            # Keep every object for table/name diagnostics, even if not plottable.
            all_table_names.append(name)

            v = _to_float(row.get("v_mag_1"))
            y = _to_float(row.get(mag_col))
            yerr = _to_float(row.get(mag_err_col))
            count_rate = _to_float(row.get("count_rate"))
            count_rate_err = _to_float(row.get("count_rate_err"))
            snr_raw = None
            if (
                count_rate is not None
                and count_rate_err is not None
                and count_rate_err > 0
            ):
                snr_raw = count_rate / count_rate_err

            # Optional V uncertainty, if added later.
            xerr = (
                _to_float(row.get("v_mag_1_err"))
                or _to_float(row.get("v_mag_err"))
                or _to_float(row.get("v_err"))
            )

            if v is None or y is None:
                print(
                    f"[PLOTS][SKIP] {name}: "
                    f"v_mag_1={row.get('v_mag_1')!r}, "
                    f"{mag_col}={row.get(mag_col)!r}"
                )
                continue

            valid_rows += 1

            measurements.setdefault(name, []).append(
                {
                    "v": v,
                    "y": y,
                    "yerr": yerr,
                    "xerr": xerr,
                    "color": y - v,
                    "count_rate": count_rate,
                    "count_rate_err": count_rate_err,
                    "snr_raw": snr_raw,
                }
            )

        if not measurements:
            raise ValueError(
                f"No valid rows to plot. Need numeric v_mag_1 and {mag_col} in {phot_csv}"
            )

    print(f"[PLOTS][ROCKS][DBG] names for table lookup: {len(set(all_table_names))}")
    
    for test_name in ["Astrometria", "Potomac", "Sarpedon"]:
        print(f"[PLOTS][ROCKS][DBG] {test_name} in all_table_names? {test_name in set(all_table_names)}")
    
    # 2) Fetch taxonomy/albedo/diameter via rocks (cached)
    all_names = sorted(measurements.keys())

    print(f"[PLOTS][DBG] Names sent to rocks: {len(all_names)}")
    print("[PLOTS][DBG] First names:", all_names[:10])
    
    # Use exactly the object names that survived the photometry filters.
    all_names = sorted(measurements.keys())

    print(f"[PLOTS][DBG] Unique valid object names: {len(all_names)}")
    print(f"[PLOTS][DBG] First valid object names: {all_names[:15]}")

    rocks_info = fetch_rocks_params_by_name(
        names=all_names,
        cache_path=cache_path,
    )

    print(f"[PLOTS][DBG] rocks_info entries: {len(rocks_info)}")
    # rocks_info = fetch_rocks_params_by_name(all_table_names, cache_path)

    solar_uv_v, vpred_scatter_mag = _reflectance_settings(config)
    print(
        f"[PLOTS][REFLECTANCE] solar(UVW1_AB-V)={solar_uv_v:.4f} mag; "
        f"Vpred scatter={vpred_scatter_mag:.3f} mag"
    )


    export_photometry_csv_to_latex(
        phot_csv=phot_csv,
        out_tex=out_tex,
        columns=latex_columns,
        caption="Photometry output exported from the pipeline.",
        label="tab:photometry_output",
        max_rows=None,
        rocks_cache_path=cache_path,
        landscape=False,
    )

    export_l_reference_multifilter_table_from_config(
        config=config,
        default_l_csv=phot_csv,
        default_output_dir=out_dir,
        rocks_cache_path=cache_path,
    )

    # 3) Build final one-point-per-object arrays + taxonomy buckets
    # 3) Build final one-point-per-object arrays + taxonomy buckets
    xs: list[float] = []
    ys: list[float] = []
    xerrs: list[Optional[float]] = []
    yerrs: list[Optional[float]] = []
    uv_minus_v: list[float] = []
    buckets: list[str] = []
    tax_labels: list[str] = []

    resolved_tax = 0
    resolved_pv = 0
    resolved_d = 0

    reflectance_objects: list[dict[str, Any]] = []
    reflectance_frames: list[dict[str, Any]] = []


    def _mean(vals: list[float]) -> float:
        return float(np.mean(np.array(vals, dtype=float)))


    def _stderr(vals: list[float]) -> Optional[float]:
        if len(vals) < 2:
            return None
        arr = np.array(vals, dtype=float)
        return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


    def _weighted_mean_and_err(
        vals: list[float],
        errs: list[Optional[float]],
    ) -> tuple[float, Optional[float]]:
        """
        Weighted mean using sigma if available.
        If no valid uncertainties exist, use simple mean and standard error.
        """
        good_vals = []
        good_w = []

        for v, e in zip(vals, errs):
            if e is not None and math.isfinite(e) and e > 0:
                good_vals.append(v)
                good_w.append(1.0 / (e * e))

        if good_vals:
            v_arr = np.array(good_vals, dtype=float)
            w_arr = np.array(good_w, dtype=float)
            mean = float(np.sum(w_arr * v_arr) / np.sum(w_arr))
            err = float(math.sqrt(1.0 / np.sum(w_arr)))
            return mean, err

        mean = _mean(vals)
        err = _stderr(vals)
        return mean, err


    for name, rows_for_obj in measurements.items():
        v_vals = [r["v"] for r in rows_for_obj if r["v"] is not None]
        y_vals = [r["y"] for r in rows_for_obj if r["y"] is not None]
        yerr_vals = [r["yerr"] for r in rows_for_obj if r["y"] is not None]
        xerr_vals = [r["xerr"] for r in rows_for_obj if r["v"] is not None]

        if not v_vals or not y_vals:
            continue

        # Existing magnitude-vs-V diagnostic point.
        x = _mean(v_vals)
        xerr_from_scatter = _stderr(v_vals)

        valid_xerrs = [
            e for e in xerr_vals
            if e is not None and math.isfinite(e) and e > 0
        ]
        if valid_xerrs:
            xerr = float(math.sqrt(np.sum(np.array(valid_xerrs) ** 2)) / len(valid_xerrs))
        else:
            xerr = xerr_from_scatter

        y, yerr = _weighted_mean_and_err(y_vals, yerr_vals)

        # IMPORTANT: combine UVW1-V per frame, rather than subtracting two
        # independently averaged magnitudes. This is the object-level science colour.
        colour_vals: list[float] = []
        colour_errs: list[Optional[float]] = []
        for r in rows_for_obj:
            if r["color"] is None:
                continue
            colour_vals.append(float(r["color"]))
            ye = r["yerr"]
            xe = r["xerr"]
            if ye is None and xe is None:
                colour_errs.append(None)
            else:
                colour_errs.append(
                    math.sqrt(float(ye or 0.0) ** 2 + float(xe or 0.0) ** 2)
                )

        if not colour_vals:
            continue

        color_mean, color_err = _weighted_mean_and_err(colour_vals, colour_errs)

        # Build an equivalent (m_UVW1, Vpred) pair that preserves the combined
        # colour exactly. Only the colour is used in the R transformation.
        v_for_r = x
        m_for_r = v_for_r + color_mean
        refl = compute_uv_reflectance(
            m_uv=m_for_r,
            m_uv_err=color_err,
            v_pred=v_for_r,
            solar_color=solar_uv_v,
            vpred_scatter_mag=vpred_scatter_mag,
        )

        info = _get_rocks_info(rocks_info, name)
        tax_class = info.get("taxonomy_class")
        pv = _to_float(info.get("pv"))
        d_km = _to_float(info.get("diameter_km"))
        rocks_number = info.get("rocks_number")
        family = _taxonomy_bucket(tax_class)

        if tax_class:
            resolved_tax += 1
        if pv is not None:
            resolved_pv += 1
        if d_km is not None:
            resolved_d += 1

        xs.append(x)
        ys.append(y)
        xerrs.append(xerr)
        yerrs.append(yerr)
        uv_minus_v.append(color_mean)
        buckets.append(family)
        tax_labels.append(str(tax_class or "UNK"))

        r_value = float(refl["reflectance_uv_v"])
        r_err_phot = refl["reflectance_err_phot"]
        p_uv = pv * r_value if pv is not None else None
        p_uv_err = pv * float(r_err_phot) if pv is not None and r_err_phot is not None else None

        snrs = [
            float(r["snr_raw"])
            for r in rows_for_obj
            if r.get("snr_raw") is not None
            and math.isfinite(float(r["snr_raw"]))
            and float(r["snr_raw"]) > 0
        ]

        reflectance_objects.append(
            {
                "target_name": name,
                "rocks_number": rocks_number,
                "taxonomy_class": tax_class,
                "taxonomy_family": family,
                "n_frames": len(colour_vals),
                "uvw1_minus_vpred": color_mean,
                "uvw1_minus_vpred_err_phot": color_err,
                "solar_uvw1_ab_minus_v": solar_uv_v,
                "delta_uv_solar": refl["delta_uv_solar"],
                "reflectance_uv_v": r_value,
                "reflectance_err_phot": r_err_phot,
                "reflectance_err_external_lo": refl["reflectance_err_external_lo"],
                "reflectance_err_external_hi": refl["reflectance_err_external_hi"],
                "pv": pv,
                "puvw1_proxy": p_uv,
                "puvw1_proxy_err_phot": p_uv_err,
                "diameter_km": d_km,
                "snr_raw_median": float(np.median(snrs)) if snrs else None,
                "snr_raw_min": min(snrs) if snrs else None,
                "snr_raw_max": max(snrs) if snrs else None,
            }
        )

        # Frame-level R-S/N QC points. This deliberately stays frame-level:
        # it is meant to reveal whether high R values accumulate near the
        # detection threshold.
        for r in rows_for_obj:
            if r.get("snr_raw") is None or r.get("color") is None:
                continue
            frame_refl = compute_uv_reflectance(
                m_uv=float(r["y"]),
                m_uv_err=r.get("yerr"),
                v_pred=float(r["v"]),
                solar_color=solar_uv_v,
                vpred_scatter_mag=None,
            )
            reflectance_frames.append(
                {
                    "target_name": name,
                    "taxonomy_family": family,
                    "snr_raw": float(r["snr_raw"]),
                    "reflectance_uv_v": frame_refl["reflectance_uv_v"],
                    "reflectance_err_phot": frame_refl["reflectance_err_phot"],
                }
            )

    export_taxonomy_colour_summary_tables(
        output_dir=out_dir,
        xs=xs,
        ys=ys,
        yerrs=yerrs,
        buckets=buckets,
    )

    export_reflectance_products(
        output_dir=out_dir,
        object_rows=reflectance_objects,
        frame_rows=reflectance_frames,
    )
    # # 4) Plot grouped by taxonomic family, with error bars
    # plt.figure(figsize=(7.0, 5.5))

    # order = ["C", "S", "X", "D", "V", "B", "A", "L", "K", "Q", "OTHER", "UNK"]
    # plotted = 0

    # for b in order:
    #     idx = [i for i, bb in enumerate(buckets) if bb == b]
    #     if not idx:
    #         continue

    #     xb = [xs[i] for i in idx]
    #     yb = [ys[i] for i in idx]

    #     xeb = [xerrs[i] if xerrs[i] is not None else 0.0 for i in idx]
    #     yeb = [yerrs[i] if yerrs[i] is not None else 0.0 for i in idx]

    #     # Error bars + markers. Matplotlib assigns default colors.
    #     container = plt.errorbar(
    #         xb,
    #         yb,
    #         xerr=xeb if any(e > 0 for e in xeb) else None,
    #         yerr=yeb if any(e > 0 for e in yeb) else None,
    #         fmt="o",
    #         markersize=4,
    #         capsize=2,
    #         elinewidth=0.8,
    #         linewidth=0.8,
    #         label=b,
    #         alpha=0.85,
    #     )

    #     plotted += len(xb)

    #     fit = fit_line(xb, yb, yeb)

    #     if fit is not None:
    #         m, c, m_err, c_err = fit

    #         # Extend family slopes across the full x-axis range for readability.
    #         xx = np.linspace(min(xs), max(xs), 100)
    #         yy = m * xx + c

    #         # Use the same color as the plotted family.
    #         try:
    #             fit_color = container.lines[0].get_color()
    #         except Exception:
    #             fit_color = None

    #         plt.plot(xx, yy, linestyle="-", linewidth=1.0, color=fit_color)

    #         if math.isfinite(m_err) and math.isfinite(c_err):
    #             print(
    #                 f"[PLOTS][FIT] {b}: "
    #                 f"slope={m:.4f}±{m_err:.4f}, "
    #                 f"intercept={c:.4f}±{c_err:.4f}, "
    #                 f"N={len(xb)}"
    #             )
    #         else:
    #             print(
    #                 f"[PLOTS][FIT] {b}: "
    #                 f"slope={m:.4f}, intercept={c:.4f}, N={len(xb)}"
    #             )
    #     else:
    #         print(f"[PLOTS][FIT] {b}: skipped (N={len(xb)} or insufficient valid errors)")

    # 4) Plot grouped by taxonomic family, with error bars
    plt.figure(figsize=(7.0, 5.5))

    order = ["C", "S", "X", "D", "V", "B", "A", "L", "K", "Q", "OTHER", "UNK"]
    plotted = 0

    for b in order:
        idx = [i for i, bb in enumerate(buckets) if bb == b]
        if not idx:
            continue

        xb = [xs[i] for i in idx]
        yb = [ys[i] for i in idx]

        xeb = [xerrs[i] if xerrs[i] is not None else 0.0 for i in idx]
        yeb = [yerrs[i] if yerrs[i] is not None else 0.0 for i in idx]

        has_xerr = any(e > 0 for e in xeb)
        has_yerr = any(e > 0 for e in yeb)

        # ------------------------------------------------------------
        # Plot points
        # ------------------------------------------------------------
        if b == "UNK":
            # Unknown taxonomies: grey diagnostic points only.
            container = plt.errorbar(
                xb,
                yb,
                xerr=xeb if has_xerr else None,
                yerr=yeb if has_yerr else None,
                fmt="o",
                markersize=4,
                capsize=2,
                elinewidth=0.8,
                linewidth=0.8,
                label="UNK",
                alpha=0.55,
                color="0.65",
                ecolor="0.65",
                markerfacecolor="0.65",
                markeredgecolor="0.65",
            )

            plotted += len(xb)
            print(f"[PLOTS][FIT] UNK: skipped; unknown taxonomy")
            continue

        # Known taxonomic families: default matplotlib colours.
        container = plt.errorbar(
            xb,
            yb,
            xerr=xeb if has_xerr else None,
            yerr=yeb if has_yerr else None,
            fmt="o",
            markersize=4,
            capsize=2,
            elinewidth=0.8,
            linewidth=0.8,
            label=b,
            alpha=0.85,
        )

        plotted += len(xb)

        # ------------------------------------------------------------
        # Weighted family fit
        # ------------------------------------------------------------
        fit = fit_line(xb, yb, yeb)

        if fit is None:
            print(f"[PLOTS][FIT] {b}: skipped (N={len(xb)} or insufficient valid errors)")
            continue

        m, c, m_err, c_err = fit

        xx = np.linspace(min(xs), max(xs), 100)
        yy = m * xx + c

        try:
            fit_color = container.lines[0].get_color()
        except Exception:
            fit_color = None

        plt.plot(
            xx,
            yy,
            linestyle="-",
            linewidth=1.0,
            color=fit_color,
        )

        if math.isfinite(m_err) and math.isfinite(c_err):
            print(
                f"[PLOTS][FIT] {b}: "
                f"slope={m:.4f}±{m_err:.4f}, "
                f"intercept={c:.4f}±{c_err:.4f}, "
                f"N={len(xb)}"
            )
        else:
            print(
                f"[PLOTS][FIT] {b}: "
                f"slope={m:.4f}, intercept={c:.4f}, N={len(xb)}"
            )

    # Optional C/S UV−V threshold estimate using object-level colours.
    c_colors = [col for col, bb in zip(uv_minus_v, buckets) if bb == "C"]
    s_colors = [col for col, bb in zip(uv_minus_v, buckets) if bb == "S"]

    if len(c_colors) >= 2 and len(s_colors) >= 2:
        muC = float(np.mean(c_colors))
        sC = float(np.std(c_colors, ddof=1))
        muS = float(np.mean(s_colors))
        sS = float(np.std(s_colors, ddof=1))

        if sC > 0 and sS > 0:
            thr = gaussian_intersection(muC, sC, muS, sS)
            if thr is not None:
                print(
                    f"[PLOTS][THR] UV−V: "
                    f"muC={muC:.3f}±{sC:.3f}, "
                    f"muS={muS:.3f}±{sS:.3f}, "
                    f"threshold≈{thr:.3f}"
                )

                # xx = np.linspace(min(xs), max(xs), 100)
                # yy = xx + thr
                # plt.plot(
                #     xx,
                #     yy,
                #     linestyle="--",
                #     linewidth=0.9,
                #     label="C/S UV−V threshold",
                # )
            else:
                print(
                    f"[PLOTS][THR] UV−V: "
                    f"muC={muC:.3f}±{sC:.3f}, "
                    f"muS={muS:.3f}±{sS:.3f}, "
                    f"threshold skipped; Gaussian intersection returned None."
                )
        else:
            print("[PLOTS][THR] C/S scatter is zero; threshold skipped.")
    else:
        print("[PLOTS][THR] Not enough C/S points to estimate threshold robustly.")
        
    # plt.xlabel("v_mag_1")
    # plt.ylabel("mag_ab_apcorr")
    # plt.title("mag_ab_apcorr vs v_mag_1 (1 point per object; mean UV−V; rocks taxonomy)")
    plt.xlabel(r"$V_{\rm pred}$")
    plt.ylabel(r"$m_{\rm UVW1,AB}$")
    plt.title(r"UVW1 AB magnitude vs predicted $V$ by taxonomic family")
    plt.legend()

    # y=x reference (color=0)
    # mn = min(min(xs), min(ys))
    # mx = max(max(xs), max(ys))
    # plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[PLOTS] Read: {phot_csv}")
    print(f"[PLOTS] Total rows: {total_rows} | Valid rows: {valid_rows}")
    print(f"[PLOTS] Objects plotted: {len(measurements)} (points: {plotted})")
    print(
        f"[PLOTS] rocks resolved: "
        f"taxonomy={resolved_tax}/{len(measurements)}, "
        f"pV={resolved_pv}/{len(measurements)}, "
        f"D={resolved_d}/{len(measurements)}"
    )
    print(f"[PLOTS] Saved: {out_png}")

    return out_png


def _save_plot_pair(fig: Any, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save an article-ready PDF plus a PNG preview."""
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTS][REFLECTANCE] Saved: {pdf_path}")
    print(f"[PLOTS][REFLECTANCE] Saved: {png_path}")
    return pdf_path, png_path


def _finite_float(x: Any) -> Optional[float]:
    v = _to_float(x)
    return v if v is not None and math.isfinite(v) else None


def export_reflectance_products(
    output_dir: Path,
    object_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> None:
    """
    Export the dynamic solar-normalised reflectance products used by the paper.

    Products:
      - reflectance_by_object.csv
      - reflectance_by_taxonomy.{pdf,png}
      - puvw1_proxy_vs_pv.{pdf,png}
      - reflectance_vs_diameter.{pdf,png}
      - reflectance_vs_snr.{pdf,png}

    ``p_UVW1`` is explicitly a proxy: p_UVW1 = p_V * R_UVW1/V.  Interpreting
    it as an absolute UV geometric/bond albedo requires the usual assumption
    that the UV and visible phase functions are not strongly different.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not object_rows:
        print("[PLOTS][REFLECTANCE] No object-level reflectances; products skipped.")
        return

    # ------------------------------------------------------------
    # CSV: one row per asteroid
    # ------------------------------------------------------------
    csv_path = output_dir / "reflectance_by_object.csv"
    fieldnames = [
        "target_name",
        "rocks_number",
        "taxonomy_class",
        "taxonomy_family",
        "n_frames",
        "uvw1_minus_vpred",
        "uvw1_minus_vpred_err_phot",
        "solar_uvw1_ab_minus_v",
        "delta_uv_solar",
        "reflectance_uv_v",
        "reflectance_err_phot",
        "reflectance_err_external_lo",
        "reflectance_err_external_hi",
        "pv",
        "puvw1_proxy",
        "puvw1_proxy_err_phot",
        "diameter_km",
        "snr_raw_median",
        "snr_raw_min",
        "snr_raw_max",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(object_rows, key=lambda r: str(r.get("target_name", "")).lower()):
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[PLOTS][REFLECTANCE] Saved: {csv_path}")

    order = ["C", "S", "X", "D", "V", "B", "A", "L", "K", "Q", "OTHER", "UNK"]
    families_present = [
        fam for fam in order
        if any(r.get("taxonomy_family") == fam for r in object_rows)
    ]

    # ------------------------------------------------------------
    # 1) Taxonomy -- R: object points + median and P16/P84
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    for xpos, fam in enumerate(families_present):
        rows = [r for r in object_rows if r.get("taxonomy_family") == fam]
        if not rows:
            continue

        n = len(rows)
        if n == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.18, 0.18, n)

        rvals = np.array([float(r["reflectance_uv_v"]) for r in rows], dtype=float)
        rerrs = np.array([
            float(r.get("reflectance_err_phot") or 0.0) for r in rows
        ], dtype=float)
        xx = np.full(n, xpos, dtype=float) + jitter

        ax.errorbar(
            xx,
            rvals,
            yerr=rerrs if np.any(rerrs > 0) else None,
            fmt="o",
            markersize=4,
            capsize=2,
            elinewidth=0.8,
            linewidth=0.8,
            alpha=0.80,
        )

        med = float(np.median(rvals))
        if n >= 2:
            p16, p84 = np.percentile(rvals, [16, 84])
            ax.errorbar(
                xpos,
                med,
                yerr=np.array([[med - p16], [p84 - med]]),
                fmt="D",
                markersize=6,
                capsize=4,
                linewidth=1.2,
            )
        else:
            ax.plot(xpos, med, "D", markersize=6)

    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(families_present)), families_present)
    ax.set_ylabel(r"$R_{\rm UVW1}/R_V$")
    ax.set_xlabel("Taxonomic family")
    ax.set_title("Solar-normalised UVW1 reflectance by taxonomic family")
    _save_plot_pair(fig, output_dir, "reflectance_by_taxonomy")

    # ------------------------------------------------------------
    # 2) p_UVW1 proxy vs p_V
    # ------------------------------------------------------------
    pv_rows = [
        r for r in object_rows
        if _finite_float(r.get("pv")) is not None
        and _finite_float(r.get("puvw1_proxy")) is not None
    ]
    if pv_rows:
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        for fam in families_present:
            rows = [r for r in pv_rows if r.get("taxonomy_family") == fam]
            if not rows:
                continue
            x = np.array([float(r["pv"]) for r in rows])
            y = np.array([float(r["puvw1_proxy"]) for r in rows])
            ye = np.array([float(r.get("puvw1_proxy_err_phot") or 0.0) for r in rows])
            ax.errorbar(
                x, y,
                yerr=ye if np.any(ye > 0) else None,
                fmt="o",
                markersize=4,
                capsize=2,
                elinewidth=0.8,
                linewidth=0.8,
                alpha=0.85,
                label=fam,
            )
        maxv = max(max(float(r["pv"]), float(r["puvw1_proxy"])) for r in pv_rows)
        lim = maxv * 1.08
        ax.plot([0.0, lim], [0.0, lim], linestyle="--", linewidth=1.0)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$p_V$")
        ax.set_ylabel(r"$p_{\rm UVW1}\simeq p_V\,(R_{\rm UVW1}/R_V)$")
        ax.set_title("UVW1 albedo proxy versus visible geometric albedo")
        ax.legend(fontsize="small", ncol=2)
        _save_plot_pair(fig, output_dir, "puvw1_proxy_vs_pv")

    # ------------------------------------------------------------
    # 3) R vs diameter
    # ------------------------------------------------------------
    d_rows = [
        r for r in object_rows
        if _finite_float(r.get("diameter_km")) is not None
        and float(r["diameter_km"]) > 0
    ]
    if d_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for fam in families_present:
            rows = [r for r in d_rows if r.get("taxonomy_family") == fam]
            if not rows:
                continue
            x = np.array([float(r["diameter_km"]) for r in rows])
            y = np.array([float(r["reflectance_uv_v"]) for r in rows])
            ye = np.array([float(r.get("reflectance_err_phot") or 0.0) for r in rows])
            ax.errorbar(
                x, y,
                yerr=ye if np.any(ye > 0) else None,
                fmt="o",
                markersize=4,
                capsize=2,
                elinewidth=0.8,
                linewidth=0.8,
                alpha=0.85,
                label=fam,
            )
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Diameter [km]")
        ax.set_ylabel(r"$R_{\rm UVW1}/R_V$")
        ax.set_title("Solar-normalised UVW1 reflectance versus diameter")
        ax.legend(fontsize="small", ncol=2)
        _save_plot_pair(fig, output_dir, "reflectance_vs_diameter")

    # ------------------------------------------------------------
    # 4) R vs raw frame S/N -- frame-level QC diagnostic
    # ------------------------------------------------------------
    sn_rows = [
        r for r in frame_rows
        if _finite_float(r.get("snr_raw")) is not None
        and float(r["snr_raw"]) > 0
    ]
    if sn_rows:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        frame_families = [
            fam for fam in order
            if any(r.get("taxonomy_family") == fam for r in sn_rows)
        ]
        for fam in frame_families:
            rows = [r for r in sn_rows if r.get("taxonomy_family") == fam]
            if not rows:
                continue
            x = np.array([float(r["snr_raw"]) for r in rows])
            y = np.array([float(r["reflectance_uv_v"]) for r in rows])
            ye = np.array([float(r.get("reflectance_err_phot") or 0.0) for r in rows])
            ax.errorbar(
                x, y,
                yerr=ye if np.any(ye > 0) else None,
                fmt="o",
                markersize=4,
                capsize=2,
                elinewidth=0.8,
                linewidth=0.8,
                alpha=0.75,
                label=fam,
            )
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel(r"Raw trail S/N ($c/\sigma_c$)")
        ax.set_ylabel(r"$R_{\rm UVW1}/R_V$")
        ax.set_title("Frame-level UVW1 reflectance versus detection S/N")
        ax.legend(fontsize="small", ncol=2)
        _save_plot_pair(fig, output_dir, "reflectance_vs_snr")


def _load_config(config_path: Path) -> ConfigParser:
    cfg = ConfigParser()
    read_ok = cfg.read(config_path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    return cfg

# def fit_line(x: list[float], y: list[float]) -> Optional[tuple[float, float]]:
#     if len(x) < 3:
#         return None
#     xarr = np.array(x, dtype=float)
#     yarr = np.array(y, dtype=float)
#     m, b = np.polyfit(xarr, yarr, 1)  # y = m x + b
#     return float(m), float(b)

def fit_line(
    x: list[float],
    y: list[float],
    yerr: Optional[list[Optional[float]]] = None,
) -> Optional[tuple[float, float, float, float]]:
    """
    Linear fit y = m*x + b.

    If yerr is available, perform weighted least squares with weights 1/sigma_y.
    Returns:
        slope, intercept, slope_err, intercept_err
    """
    if len(x) < 3:
        return None

    xarr = np.array(x, dtype=float)
    yarr = np.array(y, dtype=float)

    finite = np.isfinite(xarr) & np.isfinite(yarr)

    warr = None
    if yerr is not None:
        earr = np.array(
            [
                np.nan if e is None else float(e)
                for e in yerr
            ],
            dtype=float,
        )
        finite &= np.isfinite(earr) & (earr > 0)

    xarr = xarr[finite]
    yarr = yarr[finite]

    if len(xarr) < 3:
        return None

    if yerr is not None:
        earr = earr[finite]
        warr = 1.0 / earr

    try:
        if warr is not None:
            coeff, cov = np.polyfit(xarr, yarr, 1, w=warr, cov=True)
        else:
            coeff, cov = np.polyfit(xarr, yarr, 1, cov=True)

        m, b = coeff
        m_err = math.sqrt(cov[0, 0]) if cov is not None else math.nan
        b_err = math.sqrt(cov[1, 1]) if cov is not None else math.nan

        return float(m), float(b), float(m_err), float(b_err)

    except Exception:
        if warr is not None:
            m, b = np.polyfit(xarr, yarr, 1, w=warr)
        else:
            m, b = np.polyfit(xarr, yarr, 1)

        return float(m), float(b), math.nan, math.nan

def gaussian_intersection(mu1, s1, mu2, s2):
    # Resuelve N(mu1,s1)=N(mu2,s2). Devuelve 1 o 2 soluciones; elegimos la que cae entre medias.
    a = 1/(2*s1*s1) - 1/(2*s2*s2)
    b = mu2/(s2*s2) - mu1/(s1*s1)
    c = (mu1*mu1)/(2*s1*s1) - (mu2*mu2)/(2*s2*s2) + math.log(s2/s1)
    if abs(a) < 1e-12:
        # varianzas casi iguales -> solución lineal
        return -c / b
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    r1 = (-b + math.sqrt(disc)) / (2*a)
    r2 = (-b - math.sqrt(disc)) / (2*a)
    mid = (mu1 + mu2)/2
    # elige la solución más cercana al punto medio
    return r1 if abs(r1 - mid) < abs(r2 - mid) else r2


def _latex_escape(s: str) -> str:
    """
    Escape minimal LaTeX special chars.
    """
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)

def _round_sig(x: float, sig: int = 2) -> float:
    """Round to `sig` significant figures."""
    if x == 0:
        return 0.0
    exp = math.floor(math.log10(abs(x)))
    decimals = sig - 1 - exp
    return round(x, decimals)


def _decimals_in_number_str(s: str) -> int:
    """Count decimals in a plain decimal string (no scientific notation)."""
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


def _format_value_err_pair(value: Optional[float], err: Optional[float], sig_err: int = 2) -> tuple[str, str]:
    """
    Format (value, err) such that:
    - err has `sig_err` significant figures
    - value has the same number of decimal places as the formatted err
    Uses fixed-point formatting when possible; falls back to g-format for extreme values.
    """
    if value is None or err is None:
        return ("", "")

    if not (math.isfinite(value) and math.isfinite(err)):
        return ("", "")

    if err == 0:
        # error 0: just print value and 0 with sane formatting
        return (f"{value:g}", "0")

    err_r = _round_sig(err, sig_err)

    # Decide whether to use scientific notation
    # (avoid huge strings for very small/large numbers)
    abs_err = abs(err_r)
    use_sci = abs_err != 0 and (abs_err < 1e-4 or abs_err >= 1e4 or abs(value) >= 1e6)

    if use_sci:
        # scientific notation with sig figs for both, aligned by sig figs (best effort)
        err_s = f"{err_r:.{sig_err}g}"
        # format value to similar precision scale: use decimals derived from err exponent
        # For sci, keep 6 sig figs max to avoid noise; you can tune this.
        val_s = f"{value:.6g}"
        return (val_s, err_s)

    # Fixed-point: choose decimals from err_r
    err_s_plain = f"{err_r:f}".rstrip("0").rstrip(".")
    dec = _decimals_in_number_str(err_s_plain)

    err_s = f"{err_r:.{dec}f}"
    val_s = f"{value:.{dec}f}"
    return (val_s, err_s)



def _normalize_filter_code(value: object) -> str:
    """Normalize OM filter names/codes to L, M, S, U, B or V."""
    raw = str(value or "").strip().upper()
    aliases = {
        "L": "L",
        "UVW1": "L",
        "W1": "L",
        "M": "M",
        "UVM2": "M",
        "M2": "M",
        "S": "S",
        "UVW2": "S",
        "W2": "S",
        "U": "U",
        "B": "B",
        "V": "V",
    }
    return aliases.get(raw, raw)


def _mean_measurement(
    values: list[float],
    errors: list[Optional[float]],
) -> tuple[float, Optional[float], int]:
    """
    Combine repeated frames for one asteroid/observation/filter.

    Uses an inverse-variance weighted mean when valid errors are available;
    otherwise it falls back to the arithmetic mean and its standard error.
    """
    if not values:
        raise ValueError("Cannot combine an empty measurement list")

    valid_weighted = [
        (v, e)
        for v, e in zip(values, errors)
        if e is not None and math.isfinite(e) and e > 0
    ]

    if valid_weighted:
        vals = np.array([v for v, _ in valid_weighted], dtype=float)
        errs = np.array([e for _, e in valid_weighted], dtype=float)
        weights = 1.0 / errs**2
        mean = float(np.sum(weights * vals) / np.sum(weights))
        mean_err = float(math.sqrt(1.0 / np.sum(weights)))
        return mean, mean_err, len(values)

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) >= 2:
        mean_err = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    else:
        mean_err = errors[0] if errors and errors[0] is not None else None
    return mean, mean_err, len(values)


def _read_filter_photometry_csv(
    csv_path: Path,
    expected_filter: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Read one per-filter photometry CSV.

    The returned mapping is keyed by (normalized asteroid name, observation ID).
    Repeated frames in the same filter are combined into one magnitude.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[PLOTS][MULTIFILTER] CSV for filter {expected_filter} not found: {csv_path}"
        )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = set(reader.fieldnames or [])

        name_col = "target_name" if "target_name" in fieldnames else "sso_name" if "sso_name" in fieldnames else None
        if name_col is None:
            raise ValueError(
                f"[PLOTS][MULTIFILTER] Missing target_name/sso_name in {csv_path}. "
                f"Found: {reader.fieldnames}"
            )

        required = {"observation_id", "mag_ab"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"[PLOTS][MULTIFILTER] Missing columns {sorted(missing)} in {csv_path}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            row_filter = _normalize_filter_code(row.get("filter")) if "filter" in fieldnames else expected_filter
            if row_filter and row_filter != expected_filter:
                continue

            name_raw = str(row.get(name_col) or "").strip()
            name = _normalize_name(name_raw)
            obs_id = str(row.get("observation_id") or "").strip()
            mag = _to_float(row.get("mag_ab"))
            mag_err = _to_float(row.get("mag_err"))

            if not name or not obs_id or mag is None:
                continue

            key = (_rocks_lookup_key(name), obs_id)
            entry = grouped.setdefault(
                key,
                {
                    "target_name": name,
                    "observation_id": obs_id,
                    "values": [],
                    "errors": [],
                },
            )
            entry["values"].append(mag)
            entry["errors"].append(mag_err)

    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in grouped.items():
        mean, mean_err, n_frames = _mean_measurement(entry["values"], entry["errors"])
        combined[key] = {
            "target_name": entry["target_name"],
            "observation_id": entry["observation_id"],
            "mag": mean,
            "mag_err": mean_err,
            "n_frames": n_frames,
        }

    print(
        f"[PLOTS][MULTIFILTER] {expected_filter}: "
        f"{len(combined)} asteroid/observation pairs from {csv_path}"
    )
    return combined


def export_l_reference_multifilter_table(
    filter_csvs: dict[str, Path],
    out_tex: Path,
    rocks_cache_path: Optional[Path] = None,
) -> Path:
    """
    Create a wide LaTeX table for asteroid/observation pairs with a valid L
    measurement and at least one valid measurement in another filter.

    Matching is deliberately performed using asteroid name + observation ID,
    so measurements from different epochs are never mixed.
    """
    normalized_paths = {
        _normalize_filter_code(code): Path(path).expanduser()
        for code, path in filter_csvs.items()
        if str(path).strip()
    }

    if "L" not in normalized_paths:
        raise ValueError("[PLOTS][MULTIFILTER] Filter L must be configured as the reference")

    other_filters = [f for f in ("U", "B", "V", "M", "S") if f in normalized_paths]
    if not other_filters:
        print("[PLOTS][MULTIFILTER] No non-L filter CSVs configured; table skipped.")
        return out_tex

    data_by_filter = {
        filt: _read_filter_photometry_csv(path, filt)
        for filt, path in normalized_paths.items()
        if filt == "L" or filt in other_filters
    }

    l_data = data_by_filter["L"]
    selected_keys = [
        key
        for key in l_data
        if any(key in data_by_filter[filt] for filt in other_filters)
    ]

    selected_keys.sort(
        key=lambda key: (
            l_data[key]["target_name"].lower(),
            l_data[key]["observation_id"],
        )
    )

    rocks_cache = load_rocks_cache(rocks_cache_path) if rocks_cache_path else {}
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    table_filters = ["L"] + other_filters
    col_spec = "ll" + "c" * (1 + len(table_filters))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
    ]

    headers = ["Asteroid", "Obs. ID", "Filters"] + [rf"$m_{{\rm {f},AB}}$" for f in table_filters]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for key in selected_keys:
        l_entry = l_data[key]
        name = l_entry["target_name"]
        obs_id = l_entry["observation_id"]

        info = rocks_cache.get(name) or _get_rocks_info(rocks_cache, name)
        number = info.get("rocks_number") if isinstance(info, dict) else None
        display_name = name
        if number not in (None, "", "None", "nan"):
            try:
                display_name = f"({int(float(number))}) {name}"
            except Exception:
                pass

        present_filters = [f for f in table_filters if key in data_by_filter.get(f, {})]
        filter_label = "+".join(present_filters)

        vals = [
            _latex_escape(display_name),
            _latex_escape(obs_id),
            _latex_escape(filter_label),
        ]

        for filt in table_filters:
            entry = data_by_filter.get(filt, {}).get(key)
            if entry is None:
                vals.append("--")
                continue

            mag_s, err_s = _format_value_err_pair(entry["mag"], entry["mag_err"], sig_err=2)
            if mag_s and err_s:
                vals.append(rf"${mag_s} \pm {err_s}$")
            elif mag_s:
                vals.append(mag_s)
            else:
                vals.append("--")

        lines.append(" & ".join(vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Asteroid photometry for observation/filter combinations "
                r"containing a valid UVW1 ($L$) measurement and at least one additional OM filter. "
                r"Repeated frames within the same filter are combined using an inverse-variance weighted mean when uncertainties are available.}"
            ),
            r"\label{tab:photometry_multifilter_l_reference}",
            r"\end{table*}",
            "",
        ]
    )

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[PLOTS][MULTIFILTER] Saved {len(selected_keys)} rows: {out_tex}"
    )
    return out_tex


def export_l_reference_multifilter_table_from_config(
    config: ConfigParser,
    default_l_csv: Path,
    default_output_dir: Path,
    rocks_cache_path: Optional[Path] = None,
) -> Optional[Path]:
    """Read [PLOTS] multifilter paths and create the L-reference LaTeX table."""
    if not config.has_section("PLOTS"):
        return None

    enabled = config.getboolean("PLOTS", "MULTIFILTER_ENABLED", fallback=False)
    if not enabled:
        print("[PLOTS][MULTIFILTER] Disabled by MULTIFILTER_ENABLED=FALSE")
        return None

    filter_csvs: dict[str, Path] = {}
    for filt in ("L", "U", "B", "V", "M", "S"):
        option = f"MULTIFILTER_{filt}_FILE"
        raw = config.get("PLOTS", option, fallback="").strip()
        if raw:
            filter_csvs[filt] = Path(raw).expanduser()

    filter_csvs.setdefault("L", default_l_csv)

    out_raw = config.get("PLOTS", "MULTIFILTER_OUTPUT_FILE", fallback="").strip()
    out_tex = (
        Path(out_raw).expanduser()
        if out_raw
        else default_output_dir / "photometry_multifilter_L_reference_table.tex"
    )

    return export_l_reference_multifilter_table(
        filter_csvs=filter_csvs,
        out_tex=out_tex,
        rocks_cache_path=rocks_cache_path,
    )

def export_photometry_csv_to_latex(
    phot_csv: Path,
    out_tex: Path,
    columns: list[str],
    caption: str = "Photometry results.",
    label: str = "tab:photometry",
    max_rows: Optional[int] = None,
    rocks_cache_path: Optional[Path] = None,
    landscape: bool = True,
) -> None:
    """
    Export selected columns from photometry_output.csv to a LaTeX table.
    - Uses tabular with \\toprule/\\midrule/\\bottomrule (booktabs).
    - Escapes LaTeX special chars.
    """
    if not phot_csv.exists():
        raise FileNotFoundError(f"CSV not found: {phot_csv}")

    out_tex.parent.mkdir(parents=True, exist_ok=True)

    rocks_cache = load_rocks_cache(rocks_cache_path) if rocks_cache_path else {}

    with phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = reader.fieldnames or []

        missing = [c for c in columns if c not in fieldnames]
        if missing:
            raise ValueError(
                f"Cannot export LaTeX: missing columns {missing} in {phot_csv.name}. "
                f"Found: {fieldnames}"
            )

        lines: list[str] = []
        # # lines.append(r"\begin{table*}[t]")
        # # lines.append(r"\centering")
        # # # 14 columnas -> lo más seguro es usar \scriptsize y tabular con l's.
        # # lines.append(r"\scriptsize")
        # # col_spec = "l" * len(columns)
        # # lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        # if landscape:
        #     lines.append(r"\begin{sidewaystable*}[p]")
        # else:
        #     lines.append(r"\begin{table*}[t]")

        # lines.append(r"\centering")
        # lines.append(r"\scriptsize")
        # lines.append(r"\setlength{\tabcolsep}{3pt}")
        # lines.append(r"\renewcommand{\arraystretch}{1.05}")

        # col_spec = "l" * len(columns)
        # lines.append(r"\resizebox{\textwidth}{!}{%")
        # lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        # lines.append(r"\toprule")
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.03}")

        col_spec = "".join(
            "l" if c in {"target_name", "sso_name", "observation_id", "fits_name", "filter"}
            else "c"
            for c in columns
        )
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")

        # Header
        header = " & ".join(_latex_header_name(c) for c in columns) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Rows
        n = 0
        # for row in reader:
        #     vals = []
        #     for c in columns:
        #         v = row.get(c, "")
        #         # Normaliza None/"None"
        #         if v is None:
        #             v = ""
        #         v_str = str(v).strip()
        #         if v_str.lower() in {"none", "nan", "null"}:
        #             v_str = ""
        #         vals.append(_latex_escape(v_str))
        #     lines.append(" & ".join(vals) + r" \\")
        #     n += 1
        #     if max_rows is not None and n >= max_rows:
        #         break
                # Rows
        for row in reader:
            # Parse numeric pairs we want to format consistently
            cr = _to_float(row.get("count_rate"))
            cr_err = _to_float(row.get("count_rate_err"))
            mab = _to_float(row.get("mag_ab"))
            mab_err = _to_float(row.get("mag_err"))

            mab_ap = _to_float(row.get("mag_ab_apcorr"))
            mab_ap_err = _to_float(row.get("mag_ab_apcorr_err"))

            mab_s, mab_err_s = _format_value_err_pair(mab, mab_err, sig_err=2)
            mab_ap_s, mab_ap_err_s = _format_value_err_pair(mab_ap, mab_ap_err, sig_err=2)

            cr_s, cr_err_s = _format_value_err_pair(cr, cr_err, sig_err=2)
            mab_s, mab_err_s = _format_value_err_pair(mab, mab_err, sig_err=2)

            vals = []
            for c in columns:
                # Apply special formatting for the two value/error pairs
                if c == "count_rate":
                    v_str = rf"${cr_s} \pm {cr_err_s}$" if cr_s and cr_err_s else cr_s
                elif c == "mag_ab":
                    v_str = rf"${mab_s} \pm {mab_err_s}$" if mab_s and mab_err_s else mab_s
                elif c == "mag_ab_apcorr":
                    v_str = rf"${mab_ap_s} \pm {mab_ap_err_s}$" if mab_ap_s and mab_ap_err_s else mab_ap_s
                elif c == "v_mag_1":
                    v_pred = _to_float(row.get("v_mag_1"))
                    v_str = f"{v_pred:.2f}" if v_pred is not None else ""
                else:
                    v = row.get(c, "")
                    if v is None:
                        v = ""
                    v_str = str(v).strip()
                    if v_str.lower() in {"none", "nan", "null"}:
                        v_str = ""

                    if c == "fits_name" and v_str:
                        m = re.search(r"(OMS\d{3})", v_str)
                        if m:
                            v_str = m.group(1)

                    if c in {"target_name", "sso_name"} and v_str:
                        name_norm = _normalize_name(v_str)
                        info = rocks_cache.get(name_norm, {})
                        number = info.get("rocks_number")

                        if number not in (None, "", "None", "nan"):
                            try:
                                number_s = str(int(float(number)))
                                v_str = f"({number_s}) {name_norm}"
                            except Exception:
                                v_str = name_norm
                        else:
                            v_str = name_norm
                if c in {"count_rate", "mag_ab", "mag_ab_apcorr"}:
                    vals.append(v_str)
                else:
                    vals.append(_latex_escape(v_str))

            lines.append(" & ".join(vals) + r" \\")
            n += 1
            if max_rows is not None and n >= max_rows:
                break

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(rf"\caption{{{_latex_escape(caption)}}}")
        lines.append(rf"\label{{{_latex_escape(label)}}}")
        lines.append(r"\end{table*}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][LATEX] Saved: {out_tex}")

def export_taxonomy_colour_summary_tables(
    output_dir: Path,
    xs: list[float],
    ys: list[float],
    yerrs: list[Optional[float]],
    buckets: list[str],
) -> None:
    """
    Export summary table of UVW1−V colour by taxonomic family.

    Outputs:
      - taxonomy_colour_summary.csv
      - taxonomy_colour_summary.tex

    Notes:
      - UVW1−V = mag_ab_apcorr - v_mag_1
      - The weighted mean uses mag_ab_apcorr_err as sigma, since V errors
        are currently not propagated in the pipeline.
      - The unweighted std reflects object-to-object scatter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    order = ["C", "S", "X", "D", "V", "B", "A", "L", "K", "Q", "OTHER", "UNK"]

    for fam in order:
        idx = [i for i, b in enumerate(buckets) if b == fam]
        if not idx:
            continue

        v_vals = np.array([xs[i] for i in idx], dtype=float)
        uv_vals = np.array([ys[i] for i in idx], dtype=float)
        col_vals = uv_vals - v_vals

        err_vals = np.array(
            [
                np.nan if yerrs[i] is None else float(yerrs[i])
                for i in idx
            ],
            dtype=float,
        )

        n = len(idx)

        mean_v = float(np.mean(v_vals))
        mean_uv = float(np.mean(uv_vals))
        mean_col = float(np.mean(col_vals))

        if n >= 2:
            std_col = float(np.std(col_vals, ddof=1))
            sem_col = float(std_col / math.sqrt(n))
        else:
            std_col = math.nan
            sem_col = math.nan

        valid_w = np.isfinite(err_vals) & (err_vals > 0)

        if np.any(valid_w):
            weights = 1.0 / err_vals[valid_w] ** 2
            weighted_mean_col = float(
                np.sum(weights * col_vals[valid_w]) / np.sum(weights)
            )
            weighted_mean_col_err = float(math.sqrt(1.0 / np.sum(weights)))
        else:
            weighted_mean_col = mean_col
            weighted_mean_col_err = sem_col

        rows.append(
            {
                "family": fam,
                "N": n,
                "mean_V": mean_v,
                "mean_UVW1_AB": mean_uv,
                "mean_UVW1_minus_V": mean_col,
                "std_UVW1_minus_V": std_col,
                "sem_UVW1_minus_V": sem_col,
                "weighted_mean_UVW1_minus_V": weighted_mean_col,
                "weighted_mean_UVW1_minus_V_err": weighted_mean_col_err,
            }
        )

    csv_path = output_dir / "taxonomy_colour_summary.csv"
    tex_path = output_dir / "taxonomy_colour_summary.tex"

    fieldnames = [
        "family",
        "N",
        "mean_V",
        "mean_UVW1_AB",
        "mean_UVW1_minus_V",
        "std_UVW1_minus_V",
        "sem_UVW1_minus_V",
        "weighted_mean_UVW1_minus_V",
        "weighted_mean_UVW1_minus_V_err",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _fmt(x: Any, ndigits: int = 3) -> str:
        try:
            xf = float(x)
            if not math.isfinite(xf):
                return "--"
            return f"{xf:.{ndigits}f}"
        except Exception:
            return "--"

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write("\\caption{Mean UVW1--$V$ colour by taxonomic family.}\n")
        f.write("\\label{tab:taxonomy_colour_summary}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Family & $N$ & "
            "$\\langle V \\rangle$ & "
            "$\\langle m_{\\rm UVW1} \\rangle$ & "
            "$\\langle {\\rm UVW1}-V \\rangle$ \\\\\n"
        )
        f.write("\\hline\n")

        for r in rows:
            fam = r["family"]
            n = r["N"]
            mean_v = _fmt(r["mean_V"])
            mean_uv = _fmt(r["mean_UVW1_AB"])

            col = _fmt(r["weighted_mean_UVW1_minus_V"])
            col_err = _fmt(r["weighted_mean_UVW1_minus_V_err"])

            if col_err == "--":
                col_tex = col
            else:
                col_tex = f"{col} $\\pm$ {col_err}"

            f.write(
                f"{fam} & {n} & {mean_v} & {mean_uv} & {col_tex} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\n")
        f.write(
            "\\vspace{0.5ex}\n"
            "\\footnotesize\n"
            "The colour uncertainty corresponds to the uncertainty of the "
            "weighted mean using the UVW1 photometric errors. The intrinsic "
            "object-to-object scatter is reported in the accompanying CSV file.\n"
        )
        f.write("\\end{table}\n")

    print(f"[PLOTS][LATEX] Saved: {tex_path}")
    print(f"[PLOTS][CSV] Saved: {csv_path}")

    print("[PLOTS][COLOUR] Taxonomy colour summary:")
    for r in rows:
        print(
            f"  {r['family']:>5s} "
            f"N={r['N']:2d} "
            f"<UVW1-V>={_fmt(r['weighted_mean_UVW1_minus_V'])}"
            f"±{_fmt(r['weighted_mean_UVW1_minus_V_err'])} "
            f"scatter={_fmt(r['std_UVW1_minus_V'])}"
        )

if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
    config = _load_config(cfg_path)
    action_plots(config)