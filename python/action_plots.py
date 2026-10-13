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
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

try:
    from scipy.stats import chi2 as scipy_chi2
except Exception:
    scipy_chi2 = None



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
# FITS timing / Pluto time-series helpers
# -----------------------------
def _resolve_photometry_fits_path(
    config: ConfigParser,
    observation_id: str,
    filt: str,
    fits_name: str,
) -> Optional[Path]:
    """Resolve a photometry FITS using the same DOWNLOAD_DIRECTORY layout as the pipeline."""
    raw = config.get("INPUT", "DOWNLOAD_DIRECTORY", fallback="").strip()
    if not raw or not fits_name:
        return None

    base = Path(raw).expanduser()
    obs = str(observation_id or "").strip()
    ff = _normalize_filter_code(filt) or str(filt or "").strip()

    direct = base / obs / ff / fits_name
    if direct.exists():
        return direct

    # Robust fallback for legacy directory layouts: restrict the search to this OBSID.
    obs_dir = base / obs
    if obs_dir.exists():
        matches = list(obs_dir.rglob(fits_name))
        if matches:
            if len(matches) > 1:
                print(
                    f"[PLOTS][TIME] WARN: multiple FITS matches for {fits_name}; "
                    f"using {matches[0]}"
                )
            return matches[0]

    return None


def _frame_timing_from_fits(fits_path: Path) -> tuple[Time, Time, float]:
    """Return UTC start time, mid-exposure time, and exposure [s] from a FITS file."""
    with fits.open(fits_path, memmap=False) as hdul:
        headers = [hdu.header for hdu in hdul if hasattr(hdu, "header")]

        date_obs = None
        mjd_obs = None
        for hdr in headers:
            if date_obs is None:
                date_obs = hdr.get("DATE-OBS") or hdr.get("DATE_OBS")
            if mjd_obs is None and hdr.get("MJD-OBS") is not None:
                mjd_obs = _to_float(hdr.get("MJD-OBS"))
            if date_obs is not None or mjd_obs is not None:
                break

        exposure = None
        for hdr in headers:
            for key in ("EXPOSURE", "EXPTIME", "ONTIME", "EXP_TIME", "TELAPSE"):
                value = _to_float(hdr.get(key))
                if value is not None and value > 0:
                    exposure = value
                    break
            if exposure is not None:
                break

        if exposure is None:
            for hdr in headers:
                mjd_start = _to_float(hdr.get("MJD-OBS"))
                mjd_end = _to_float(hdr.get("MJD-END"))
                if mjd_start is not None and mjd_end is not None and mjd_end > mjd_start:
                    exposure = (mjd_end - mjd_start) * 86400.0
                    break

    if date_obs is not None:
        t_start = Time(str(date_obs), scale="utc")
    elif mjd_obs is not None:
        t_start = Time(float(mjd_obs), format="mjd", scale="utc")
    else:
        raise ValueError(f"DATE-OBS/MJD-OBS missing in {fits_path}")

    if exposure is None or not math.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"Exposure time missing/invalid in {fits_path}")

    t_mid = t_start + 0.5 * float(exposure) * u.s
    return t_start, t_mid, float(exposure)


def _read_object_filter_rows(
    csv_path: Path,
    target_name: str,
    observation_id: str,
    expected_filter: str,
) -> list[dict[str, str]]:
    """Read valid photometry rows for one object/OBSID/filter."""
    if not csv_path.exists():
        return []

    out: list[dict[str, str]] = []
    target_key = _rocks_lookup_key(target_name)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = set(reader.fieldnames or [])
        name_col = (
            "target_name" if "target_name" in fieldnames
            else "sso_name" if "sso_name" in fieldnames
            else None
        )
        if name_col is None:
            return []

        for row in reader:
            if _rocks_lookup_key(str(row.get(name_col) or "")) != target_key:
                continue
            if str(row.get("observation_id") or "").strip() != str(observation_id).strip():
                continue
            if "filter" in fieldnames:
                row_filter = _normalize_filter_code(row.get("filter"))
                if row_filter and row_filter != _normalize_filter_code(expected_filter):
                    continue
            if _to_float(row.get("mag_ab")) is None:
                continue
            out.append(row)
    return out



def _target_slug(name: str) -> str:
    s = _normalize_name(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "target"


def _config_variable_targets(config: ConfigParser) -> list[str]:
    raw = config.get("PLOTS", "VARIABLE_ANALYSIS", fallback="Pluto").strip()
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in re.split(r"[,;\n]+", raw):
        name = _normalize_name(item)
        if not name:
            continue
        key = _rocks_lookup_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _read_real_v_reference(
    config: ConfigParser,
    target_name: str,
    observation_id: str,
) -> tuple[Optional[float], Optional[float], str, Optional[Time]]:
    """Return real OM V reference photometry for one target/OBSID when configured."""
    v_obs: Optional[float] = None
    v_obs_err: Optional[float] = None
    v_frame = ""
    v_mid: Optional[Time] = None

    # Prefer the current multiband V product.  MULTIFILTER_V_FILE is retained
    # only as a backwards-compatible fallback for older configuration files.
    v_csv_raw = config.get("PLOTS", "MULTIBAND_V", fallback="").strip()
    if not v_csv_raw:
        v_csv_raw = config.get("PLOTS", "MULTIFILTER_V_FILE", fallback="").strip()
    if not v_csv_raw:
        return v_obs, v_obs_err, v_frame, v_mid

    v_csv = Path(v_csv_raw).expanduser()
    v_rows = _read_object_filter_rows(v_csv, target_name, observation_id, "V")
    if not v_rows:
        return v_obs, v_obs_err, v_frame, v_mid

    v_vals = [_to_float(r.get("mag_ab")) for r in v_rows]
    v_errs = [_to_float(r.get("mag_err")) for r in v_rows]
    vv = [float(v) for v in v_vals if v is not None]
    ee = [v_errs[i] for i, v in enumerate(v_vals) if v is not None]
    if vv:
        v_obs, v_obs_err, _ = _mean_measurement(vv, ee)

    first_v = v_rows[0]
    v_frame = str(first_v.get("fits_name") or "").strip()
    v_path = _resolve_photometry_fits_path(config, observation_id, "V", v_frame)
    if v_path is not None:
        try:
            _, v_mid, _ = _frame_timing_from_fits(v_path)
        except Exception as exc:
            print(f"[PLOTS][VAR] WARN: could not read V timing from {v_path}: {exc}")

    return v_obs, v_obs_err, v_frame, v_mid


def _weighted_mean_rate(
    rates: list[float],
    errors: list[Optional[float]],
) -> tuple[float, Optional[float]]:
    good = [
        (float(x), float(e))
        for x, e in zip(rates, errors)
        if e is not None and math.isfinite(float(e)) and float(e) > 0
    ]
    if good:
        x = np.array([v for v, _ in good], dtype=float)
        e = np.array([s for _, s in good], dtype=float)
        w = 1.0 / e**2
        return float(np.sum(w * x) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w)))

    arr = np.array(rates, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) >= 2:
        return mean, float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return mean, None


def _variability_statistics(
    rates: list[float],
    errors: list[Optional[float]],
) -> dict[str, Optional[float] | int | str]:
    """Variability diagnostics in linear count-rate space.

    The main quantities are:
      - constant-rate chi-square and p-value;
      - observed sample variance S^2;
      - mean squared measurement error <sigma_err^2>;
      - excess variance sigma_XS^2 = S^2 - <sigma_err^2>;
      - intrinsic rms = sqrt(max(sigma_XS^2, 0));
      - fractional variability F_var = intrinsic_rms / mean rate;
      - lag-1 residual autocorrelation as a simple whiteness diagnostic.
    """
    vals = np.array([float(x) for x in rates if x is not None and math.isfinite(float(x))], dtype=float)
    if len(vals) == 0:
        return {}

    mean_rate, mean_rate_err = _weighted_mean_rate(rates, errors)
    n = len(vals)

    valid_pairs = [
        (float(x), float(e))
        for x, e in zip(rates, errors)
        if x is not None and e is not None
        and math.isfinite(float(x)) and math.isfinite(float(e)) and float(e) > 0
    ]

    chi2_const = None
    dof = None
    p_const = None
    if len(valid_pairs) >= 2:
        chi2_const = float(sum(((x - mean_rate) / e) ** 2 for x, e in valid_pairs))
        dof = len(valid_pairs) - 1
        if scipy_chi2 is not None and dof > 0:
            p_const = float(scipy_chi2.sf(chi2_const, dof))

    sample_var = float(np.var(vals, ddof=1)) if n >= 2 else 0.0
    valid_errs = np.array(
        [float(e) for e in errors if e is not None and math.isfinite(float(e)) and float(e) > 0],
        dtype=float,
    )
    mean_err2 = float(np.mean(valid_errs**2)) if len(valid_errs) else None

    excess_var = None
    intrinsic_rms = None
    fvar = None
    fvar_err = None
    if mean_err2 is not None:
        excess_var = sample_var - mean_err2
        intrinsic_rms = math.sqrt(max(excess_var, 0.0))
        if mean_rate != 0:
            fvar = intrinsic_rms / abs(mean_rate)
            # Vaughan et al. (2003)-style measurement-noise uncertainty.
            if fvar > 0 and n > 0:
                term1 = math.sqrt(1.0 / (2.0 * n)) * mean_err2 / (fvar * mean_rate**2)
                term2 = math.sqrt(mean_err2 / n) / abs(mean_rate)
                fvar_err = math.sqrt(term1**2 + term2**2)

    # Residual lag-1 autocorrelation: useful as a simple diagnostic of correlated
    # structure, not as a formal proof of white noise.
    lag1 = None
    lag1_95 = None
    if n >= 3:
        residuals = vals - float(np.mean(vals))
        denom = float(np.dot(residuals, residuals))
        if denom > 0:
            lag1 = float(np.dot(residuals[:-1], residuals[1:]) / denom)
            lag1_95 = float(1.96 / math.sqrt(n))

    white_noise_consistent = None
    if p_const is not None:
        no_corr = (
            lag1 is None
            or lag1_95 is None
            or abs(lag1) <= lag1_95
        )
        white_noise_consistent = bool(p_const >= 0.05 and no_corr)

    if p_const is None:
        variability_class = "undetermined"
    elif p_const < 0.01 and excess_var is not None and excess_var > 0:
        variability_class = "variable"
    elif p_const >= 0.05 and white_noise_consistent:
        variability_class = "consistent_with_white_measurement_noise"
    elif p_const >= 0.05:
        variability_class = "constant_but_correlated_residuals"
    else:
        variability_class = "marginal"

    return {
        "n_frames": n,
        "weighted_mean_rate": mean_rate,
        "weighted_mean_rate_err": mean_rate_err,
        "chi2_const": chi2_const,
        "dof": dof,
        "p_const": p_const,
        "reduced_chi2": (chi2_const / dof) if chi2_const is not None and dof else None,
        "sample_std": math.sqrt(sample_var) if sample_var >= 0 else None,
        "mean_err2": mean_err2,
        "excess_variance": excess_var,
        "intrinsic_rms": intrinsic_rms,
        "fractional_variability": fvar,
        "fractional_variability_err": fvar_err,
        "lag1_autocorr": lag1,
        "lag1_95_limit": lag1_95,
        "white_noise_consistent": white_noise_consistent,
        "classification": variability_class,
    }


def _plot_rate_timeseries_on_axis(
    ax: Any,
    target_name: str,
    rows: list[dict[str, Any]],
    mean_rate: float,
    mean_rate_err: Optional[float],
    panel_label: Optional[str] = None,
) -> None:
    good = [r for r in rows if r.get("delta_t_h") is not None and r.get("count_rate") is not None]
    if not good:
        return
    xs = [float(r["delta_t_h"]) for r in good]
    ys = [float(r["count_rate"]) for r in good]
    yerrs = [float(r.get("count_rate_err") or 0.0) for r in good]
    ax.errorbar(xs, ys, yerr=yerrs if any(e > 0 for e in yerrs) else None,
                fmt="o", capsize=2, linestyle="none")
    ax.axhline(mean_rate, linestyle="--", linewidth=1.0)
    if mean_rate_err is not None and math.isfinite(mean_rate_err) and mean_rate_err > 0:
        ax.axhspan(mean_rate - mean_rate_err, mean_rate + mean_rate_err, alpha=0.18)
    ax.set_xlabel(r"$\Delta t$ (h)")
    ax.set_ylabel(r"Count rate (ct s$^{-1}$)")
    title = target_name if panel_label is None else f"{panel_label} {target_name}"
    ax.set_title(title)


def _save_variable_time_series_plot(
    target_name: str,
    rows: list[dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
    mean_rate: float,
    mean_rate_err: Optional[float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _plot_rate_timeseries_on_axis(ax, target_name, rows, mean_rate, mean_rate_err)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _save_combined_variable_time_series_plot(
    series: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    if not series:
        return None, None

    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.2), squeeze=False)
    axes_list = list(axes[0])
    for i, (ax, item) in enumerate(zip(axes_list, series)):
        _plot_rate_timeseries_on_axis(
            ax,
            item["target_name"],
            item["rows"],
            item["mean_rate"],
            item["mean_rate_err"],
            panel_label=f"({chr(97 + i)})",
        )
    fig.tight_layout()
    png = output_dir / "variable_time_series_combined.png"
    pdf = output_dir / "variable_time_series_combined.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOTS][VAR] Saved: {png}")
    print(f"[PLOTS][VAR] Saved: {pdf}")
    return png, pdf


def export_variable_time_series_products(
    config: ConfigParser,
    l_phot_csv: Path,
    output_dir: Path,
) -> list[Path]:
    """Export count-rate time series for PLOTS.VARIABLE_ANALYSIS targets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _config_variable_targets(config)
    if not targets:
        print("[PLOTS][VAR] No VARIABLE_ANALYSIS targets configured; skipping time-series products.")
        return []

    produced: list[Path] = []
    combined_series: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with l_phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = set(reader.fieldnames or [])
        name_col = (
            "target_name" if "target_name" in fieldnames
            else "sso_name" if "sso_name" in fieldnames
            else None
        )
        if name_col is None:
            print("[PLOTS][VAR] No target_name/sso_name column; time-series products skipped.")
            return []
        all_rows = list(reader)

    for target_name in targets:
        target_key = _rocks_lookup_key(target_name)
        target_rows: list[dict[str, str]] = []
        for row in all_rows:
            if _rocks_lookup_key(str(row.get(name_col) or "")) != target_key:
                continue
            row_filter = _normalize_filter_code(row.get("filter")) if "filter" in fieldnames else "L"
            if row_filter and row_filter != "L":
                continue
            rate = _to_float(row.get("count_rate"))
            rate_err = _to_float(row.get("count_rate_err"))
            if rate is None or rate_err is None or rate_err <= 0:
                continue
            target_rows.append(row)

        if not target_rows:
            print(f"[PLOTS][VAR] No valid UVW1 count-rate rows found for {target_name}; skipping.")
            continue

        by_obs: dict[str, list[dict[str, str]]] = {}
        for row in target_rows:
            obsid = str(row.get("observation_id") or "").strip()
            if obsid:
                by_obs.setdefault(obsid, []).append(row)
        if not by_obs:
            print(f"[PLOTS][VAR] {target_name}: no observation_id values; skipping.")
            continue

        # Choose the richest UVW1 sequence if the object appears in multiple OBSIDs.
        obsid, uv_rows = max(by_obs.items(), key=lambda kv: len(kv[1]))
        print(f"[PLOTS][VAR] {target_name}: using OBSID {obsid} with {len(uv_rows)} valid UVW1 frames")

        timed_rows: list[dict[str, Any]] = []
        for row in uv_rows:
            fits_name = str(row.get("fits_name") or "").strip()
            frame_match = re.search(r"(OMS\d{3})", fits_name)
            frame = frame_match.group(1) if frame_match else fits_name
            rate = _to_float(row.get("count_rate"))
            rate_err = _to_float(row.get("count_rate_err"))
            if rate is None or rate_err is None or rate_err <= 0:
                continue

            fits_path = _resolve_photometry_fits_path(config, obsid, "L", fits_name)
            t_start: Optional[Time] = None
            t_mid: Optional[Time] = None
            exposure_s: Optional[float] = None
            if fits_path is None:
                print(f"[PLOTS][VAR] WARN: FITS not found for timing: {fits_name}")
            else:
                try:
                    t_start, t_mid, exposure_s = _frame_timing_from_fits(fits_path)
                except Exception as exc:
                    print(f"[PLOTS][VAR] WARN: timing failed for {fits_path}: {exc}")

            timed_rows.append(
                {
                    "target_name": target_name,
                    "observation_id": obsid,
                    "frame": frame,
                    "fits_name": fits_name,
                    "start_utc": t_start.utc.isot if t_start is not None else None,
                    "mid_time": t_mid,
                    "mid_mjd": float(t_mid.utc.mjd) if t_mid is not None else None,
                    "exposure_s": exposure_s,
                    "count_rate": float(rate),
                    "count_rate_err": float(rate_err),
                }
            )

        def _frame_sort_key(r: dict[str, Any]):
            mt = r.get("mid_time")
            m = re.search(r"(\d+)$", str(r.get("frame") or ""))
            frame_num = int(m.group(1)) if m else 10**9
            return (0, float(mt.utc.mjd), frame_num) if mt is not None else (1, 0.0, frame_num)

        timed_rows.sort(key=_frame_sort_key)
        timed_with_time = [r for r in timed_rows if r.get("mid_time") is not None]
        t0 = timed_with_time[0]["mid_time"] if timed_with_time else None

        rates = [float(r["count_rate"]) for r in timed_rows]
        rate_errs = [float(r["count_rate_err"]) for r in timed_rows]
        mean_rate, mean_rate_err = _weighted_mean_rate(rates, rate_errs)

        for row in timed_rows:
            mt = row.get("mid_time")
            row["delta_t_h"] = (
                float((mt - t0).to_value(u.h)) if mt is not None and t0 is not None else None
            )
            row["delta_count_rate"] = float(row["count_rate"]) - mean_rate
            row["fractional_delta_rate"] = (
                float(row["delta_count_rate"]) / mean_rate if mean_rate != 0 else None
            )
            row.pop("mid_time", None)

        stats = _variability_statistics(rates, rate_errs)
        stats_row = {"target_name": target_name, "observation_id": obsid, **stats}
        summary_rows.append(stats_row)

        ptxt = "NA" if stats.get("p_const") is None else f"{float(stats['p_const']):.4g}"
        ftxt = "NA" if stats.get("fractional_variability") is None else f"{100*float(stats['fractional_variability']):.2f}%"
        print(
            f"[PLOTS][VAR] {target_name}: mean={mean_rate:.5g} ct/s; "
            f"chi2nu={stats.get('reduced_chi2')}; p_const={ptxt}; "
            f"Fvar={ftxt}; class={stats.get('classification')}"
        )

        slug = _target_slug(target_name)
        csv_path = output_dir / f"{slug}_time_series.csv"
        csv_fields = [
            "target_name", "observation_id", "frame", "fits_name", "start_utc",
            "mid_mjd", "delta_t_h", "exposure_s", "count_rate", "count_rate_err",
            "delta_count_rate", "fractional_delta_rate",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in timed_rows:
                writer.writerow({k: row.get(k) for k in csv_fields})
        print(f"[PLOTS][VAR] Saved: {csv_path}")
        produced.append(csv_path)

        tex_path = output_dir / f"{slug}_time_series_table.tex"
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.03}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Frame & $\Delta t$ (h) & $t_{\rm exp}$ (s) & $c$ (ct s$^{-1}$) & $\Delta c$ (ct s$^{-1}$) \\",
            r"\midrule",
        ]
        for row in timed_rows:
            rate_s, rate_e_s = _format_value_err_pair(
                _finite_float(row.get("count_rate")),
                _finite_float(row.get("count_rate_err")),
                sig_err=2,
            )
            rate_tex = rf"${rate_s} \pm {rate_e_s}$" if rate_s and rate_e_s else (rate_s or "--")
            lines.append(
                " & ".join(
                    [
                        _latex_escape(str(row.get("frame") or "--")),
                        f"{float(row['delta_t_h']):.3f}" if row.get("delta_t_h") is not None else "--",
                        f"{float(row['exposure_s']):.1f}" if row.get("exposure_s") is not None else "--",
                        rate_tex,
                        f"{float(row['delta_count_rate']):+.4f}" if row.get("delta_count_rate") is not None else "--",
                    ]
                ) + r" \\"
            )

        caption_parts = [
            rf"Time-resolved UVW1 count-rate measurements of {_latex_escape(target_name)} in observation {obsid}.",
            r"$\Delta t$ is measured between exposure mid-times derived from the FITS headers.",
            rf"The inverse-variance weighted mean count rate is $\langle c\rangle={mean_rate:.4f}$ ct s$^{{-1}}$.",
        ]
        if stats.get("reduced_chi2") is not None:
            caption_parts.append(
                rf"A constant-rate fit gives $\chi^2_\nu={float(stats['reduced_chi2']):.2f}$ with {int(stats['dof'])} degrees of freedom."
            )
        if stats.get("p_const") is not None:
            caption_parts.append(rf"The corresponding p-value for the constant-rate hypothesis is $p={float(stats['p_const']):.3g}$.")
        if stats.get("fractional_variability") is not None:
            fvar = 100.0 * float(stats["fractional_variability"])
            ferr = stats.get("fractional_variability_err")
            if ferr is not None:
                caption_parts.append(rf"The noise-subtracted fractional variability is $F_{{\rm var}}={fvar:.2f}\pm{100.0*float(ferr):.2f}\%$.")
            else:
                caption_parts.append(rf"The noise-subtracted fractional variability is $F_{{\rm var}}={fvar:.2f}\%$.")
        caption_parts.append(rf"The resulting classification is ``{stats.get('classification', 'undetermined').replace('_', ' ')}''.")

        lines.extend([
            r"\bottomrule", r"\end{tabular}",
            r"\caption{" + " ".join(caption_parts) + r"}",
            rf"\label{{tab:{slug}-time-series}}", r"\end{table*}", "",
        ])
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[PLOTS][VAR] Saved: {tex_path}")
        produced.append(tex_path)

        png_path = output_dir / f"{slug}_time_series.png"
        pdf_path = output_dir / f"{slug}_time_series.pdf"
        _save_variable_time_series_plot(target_name, timed_rows, png_path, pdf_path, mean_rate, mean_rate_err)
        print(f"[PLOTS][VAR] Saved: {png_path}")
        print(f"[PLOTS][VAR] Saved: {pdf_path}")
        produced.extend([png_path, pdf_path])

        combined_series.append({
            "target_name": target_name,
            "rows": timed_rows,
            "mean_rate": mean_rate,
            "mean_rate_err": mean_rate_err,
        })

    if summary_rows:
        summary_path = output_dir / "variability_summary.csv"
        summary_fields = [
            "target_name", "observation_id", "n_frames", "weighted_mean_rate",
            "weighted_mean_rate_err", "chi2_const", "dof", "p_const", "reduced_chi2",
            "sample_std", "mean_err2", "excess_variance", "intrinsic_rms",
            "fractional_variability", "fractional_variability_err",
            "lag1_autocorr", "lag1_95_limit", "white_noise_consistent", "classification",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k) for k in summary_fields})
        print(f"[PLOTS][VAR] Saved: {summary_path}")
        produced.append(summary_path)

    combined_png, combined_pdf = _save_combined_variable_time_series_plot(combined_series, output_dir)
    if combined_png is not None:
        produced.append(combined_png)
    if combined_pdf is not None:
        produced.append(combined_pdf)

    return produced


def export_pluto_time_series_products(
    config: ConfigParser,
    l_phot_csv: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Backward-compatible wrapper retained for older calls."""
    old = config.get("PLOTS", "VARIABLE_ANALYSIS", fallback="").strip()
    if not old:
        if not config.has_section("PLOTS"):
            config.add_section("PLOTS")
        config.set("PLOTS", "VARIABLE_ANALYSIS", "Pluto")
    produced = export_variable_time_series_products(config, l_phot_csv, output_dir)
    return produced[0] if produced else None



def export_repeated_photometry_variability(
    l_phot_csv: Path,
    output_dir: Path,
) -> list[Path]:
    """Export variability diagnostics for every UVW1 object/OBSID with N>=2 valid frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    with l_phot_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_iter_noncomment_lines(f))
        fieldnames = set(reader.fieldnames or [])
        name_col = (
            "target_name" if "target_name" in fieldnames
            else "sso_name" if "sso_name" in fieldnames
            else None
        )
        if name_col is None:
            print("[PLOTS][REPEAT] Missing target_name/sso_name; repeated-photometry table skipped.")
            return []

        for row in reader:
            row_filter = _normalize_filter_code(row.get("filter")) if "filter" in fieldnames else "L"
            if row_filter and row_filter != "L":
                continue
            name = _normalize_name(str(row.get(name_col) or ""))
            obsid = str(row.get("observation_id") or "").strip()
            rate = _to_float(row.get("count_rate"))
            err = _to_float(row.get("count_rate_err"))
            if not name or not obsid or rate is None or err is None or err <= 0:
                continue
            key = (_rocks_lookup_key(name), obsid)
            entry = groups.setdefault(
                key,
                {"target_name": name, "observation_id": obsid, "rates": [], "errors": []},
            )
            entry["rates"].append(float(rate))
            entry["errors"].append(float(err))

    rows: list[dict[str, Any]] = []
    for entry in groups.values():
        if len(entry["rates"]) < 2:
            continue
        stats = _variability_statistics(entry["rates"], entry["errors"])
        rows.append({
            "target_name": entry["target_name"],
            "observation_id": entry["observation_id"],
            **stats,
        })

    rows.sort(key=lambda r: (float(r.get("p_const")) if r.get("p_const") is not None else 1.0,
                             str(r.get("target_name") or "").lower()))
    if not rows:
        print("[PLOTS][REPEAT] No UVW1 object/OBSID has >=2 valid frames.")
        return []

    csv_path = output_dir / "repeated_photometry_variability.csv"
    csv_fields = [
        "target_name", "observation_id", "n_frames", "weighted_mean_rate",
        "weighted_mean_rate_err", "chi2_const", "dof", "p_const", "reduced_chi2",
        "fractional_variability", "fractional_variability_err",
        "lag1_autocorr", "lag1_95_limit", "white_noise_consistent", "classification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})
    print(f"[PLOTS][REPEAT] Saved {len(rows)} repeated sequences: {csv_path}")

    tex_path = output_dir / "repeated_photometry_variability.tex"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Asteroid & Obs. ID & $N$ & $\langle c\rangle_{\rm w}$ (ct s$^{-1}$) & $F_{\rm var}$ (\%) & $\chi^2_\nu$ ($p$) \\",
        r"\midrule",
    ]
    for row in rows:
        mean_s, mean_e_s = _format_value_err_pair(
            _finite_float(row.get("weighted_mean_rate")),
            _finite_float(row.get("weighted_mean_rate_err")), sig_err=2,
        )
        mean_tex = rf"${mean_s} \pm {mean_e_s}$" if mean_s and mean_e_s else (mean_s or "--")
        fvar = row.get("fractional_variability")
        ferr = row.get("fractional_variability_err")
        if fvar is None:
            fvar_tex = "--"
        elif ferr is not None:
            fvar_tex = rf"${100.0*float(fvar):.2f} \pm {100.0*float(ferr):.2f}$"
        else:
            fvar_tex = rf"${100.0*float(fvar):.2f}$"
        if int(row.get("n_frames") or 0) == 2 and fvar_tex != "--":
            # Keep the N=2 caveat marker outside the numerical math expression.
            fvar_tex += r"\textsuperscript{a}"
        red = row.get("reduced_chi2")
        p = row.get("p_const")
        stat_tex = "--" if red is None else rf"${float(red):.2f}$"
        if red is not None and p is not None:
            stat_tex = rf"${float(red):.2f}$ ($p={float(p):.3g}$)"
        lines.append(
            " & ".join([
                _latex_escape(str(row.get("target_name") or "")),
                _latex_escape(str(row.get("observation_id") or "")),
                str(int(row.get("n_frames") or 0)),
                mean_tex,
                fvar_tex,
                stat_tex,
            ]) + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Frame-to-frame UVW1 photometric repeatability for all asteroid--observation "
            r"pairs with at least two valid exposures. Count rates and their uncertainties are used "
            r"directly. $F_{\rm var}$ is the noise-subtracted fractional variability amplitude, while "
            r"$\chi^2_\nu$ and $p$ refer to a constant-count-rate model. Values of $F_{\rm var}$ based "
            r"on only two exposures (marked $^{a}$) are retained only as descriptive diagnostics and "
            r"must not be interpreted as robust estimates of an intrinsic variability amplitude; for "
            r"such pairs the significance of the inter-frame difference and visual inspection of the "
            r"individual frames are more informative.}"
        ),
        r"\label{tab:repeated-photometry-variability}",
        r"\end{table*}",
        "",
    ])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][REPEAT] Saved: {tex_path}")
    return [csv_path, tex_path]

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


    # Frame-level photometry table kept as the audit trail of the actual
    # measurements.  mag_ab_apcorr is intentionally omitted: the current
    # calibration already reports the final calibrated AB magnitude in mag_ab,
    # so keeping both columns would duplicate the same quantity in the paper.
    latex_columns = [
        "target_name",
        "observation_id",
        "fits_name",
        "count_rate",
        "mag_ab",
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
        caption=(
            "Frame-level UVW1 photometric measurements retained after screening and "
            "photometric extraction. The table reports the final corrected count rate, "
            "the calibrated AB magnitude, and the SSOSS predicted visual magnitude at "
            "the observation epoch. The time-resolved Pluto sequence is reported separately."
        ),
        label="tab:photometry_output",
        max_rows=None,
        rocks_cache_path=cache_path,
        landscape=False,
        valid_photometry_only=True,
        exclude_targets={"Pluto"},
    )

    # Always regenerate the current quasi-simultaneous UVW1--optical sample
    # when MULTIBAND_U/B/V paths are configured.
    export_quasi_simultaneous_multiband_from_config(
        config=config,
        default_l_csv=phot_csv,
        default_output_dir=out_dir,
        rocks_cache_path=cache_path,
    )

    # Legacy multifilter exporter retained for backwards compatibility.
    export_l_reference_multifilter_table_from_config(
        config=config,
        default_l_csv=phot_csv,
        default_output_dir=out_dir,
        rocks_cache_path=cache_path,
    )

    export_variable_time_series_products(
        config=config,
        l_phot_csv=phot_csv,
        output_dir=out_dir,
    )

    export_repeated_photometry_variability(
        l_phot_csv=phot_csv,
        output_dir=out_dir,
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


def export_reflectance_object_latex_table(
    output_dir: Path,
    object_rows: list[dict[str, Any]],
) -> Path:
    """Export the object-level UVW1 colour/reflectance catalogue used in Results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "reflectance_by_object.tex"

    def _display_name(row: dict[str, Any]) -> str:
        name = _normalize_name(str(row.get("target_name") or "").strip())
        number = row.get("rocks_number")
        if number not in (None, "", "None", "nan"):
            try:
                return f"({int(float(number))}) {name}"
            except Exception:
                pass
        return name

    def _fmt_plain(value: Any, ndigits: int, missing: str = "--") -> str:
        v = _finite_float(value)
        if v is None:
            return missing
        return f"{v:.{ndigits}f}"

    def _fmt_diameter(value: Any) -> str:
        v = _finite_float(value)
        if v is None:
            return "--"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    solar_values = [
        _finite_float(r.get("solar_uvw1_ab_minus_v"))
        for r in object_rows
        if _finite_float(r.get("solar_uvw1_ab_minus_v")) is not None
    ]
    solar_colour = solar_values[0] if solar_values else 2.8466
    solar_colour_s = f"{solar_colour:.4f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        (
            r"Asteroid & Tax. & Group & $N_{\rm fr}$ & "
            r"$\mathrm{UVW1}-V_{\rm pred}$ & $R_{\rm UVW1}/R_V$ & "
            r"$p_V$ & $p_{\rm UVW1}^{\rm proxy}$ & $D$ (km) \\"
        ),
        r"\midrule",
    ]

    for row in sorted(object_rows, key=lambda r: str(r.get("target_name", "")).lower()):
        display_name = _latex_escape(_display_name(row))

        tax = str(row.get("taxonomy_class") or "").strip()
        fam = str(row.get("taxonomy_family") or "").strip()
        tax_s = _latex_escape(tax) if tax else "--"
        fam_s = "--" if fam in {"", "UNK"} else _latex_escape("Other" if fam == "OTHER" else fam)

        n_frames = row.get("n_frames")
        try:
            n_frames_s = str(int(n_frames))
        except Exception:
            n_frames_s = "--"

        colour = _finite_float(row.get("uvw1_minus_vpred"))
        colour_err = _finite_float(row.get("uvw1_minus_vpred_err_phot"))
        colour_s, colour_err_s = _format_value_err_pair(colour, colour_err, sig_err=2)
        if colour_s and colour_err_s:
            colour_tex = rf"${colour_s} \pm {colour_err_s}$"
        elif colour_s:
            colour_tex = colour_s
        else:
            colour_tex = "--"

        refl = _finite_float(row.get("reflectance_uv_v"))
        refl_err = _finite_float(row.get("reflectance_err_phot"))
        ext_lo = _finite_float(row.get("reflectance_err_external_lo"))
        ext_hi = _finite_float(row.get("reflectance_err_external_hi"))
        refl_s, refl_err_s = _format_value_err_pair(refl, refl_err, sig_err=2)
        if refl_s:
            if refl_err_s:
                refl_tex = rf"${refl_s} \pm {refl_err_s}"
            else:
                refl_tex = rf"${refl_s}"
            if ext_lo is not None and ext_hi is not None:
                # Keep the asymmetric Vpred-validation envelope separate from the
                # formal photometric term.  Values are rounded to the same number
                # of decimals used for the displayed reflectance where possible.
                try:
                    ndec = len(refl_s.split(".", 1)[1]) if "." in refl_s else 0
                    lo_s = f"{ext_lo:.{ndec}f}"
                    hi_s = f"{ext_hi:.{ndec}f}"
                except Exception:
                    lo_s = f"{ext_lo:.3f}"
                    hi_s = f"{ext_hi:.3f}"
                refl_tex += rf"{{}}_{{-{lo_s}}}^{{+{hi_s}}}"
            refl_tex += "$"
        else:
            refl_tex = "--"

        pv_s = _fmt_plain(row.get("pv"), 4)

        puv = _finite_float(row.get("puvw1_proxy"))
        puv_err = _finite_float(row.get("puvw1_proxy_err_phot"))
        puv_s, puv_err_s = _format_value_err_pair(puv, puv_err, sig_err=2)
        if puv_s and puv_err_s:
            puv_tex = rf"${puv_s} \pm {puv_err_s}$"
        elif puv_s:
            puv_tex = puv_s
        else:
            puv_tex = "--"

        d_s = _fmt_diameter(row.get("diameter_km"))

        lines.append(
            " & ".join(
                [
                    display_name,
                    tax_s,
                    fam_s,
                    n_frames_s,
                    colour_tex,
                    refl_tex,
                    pv_s,
                    puv_tex,
                    d_s,
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Object-level UVW1 photometric and reflectance results. "
                r"$N_{\rm fr}$ is the number of valid UVW1 frames combined for each object. "
                r"The colour is defined as $\mathrm{UVW1}-V_{\rm pred}$. "
                rf"The solar-normalised reflectance adopts $(\mathrm{{UVW1}}-V)_\odot={solar_colour_s}$ mag. "
                r"For $R_{\rm UVW1}/R_V$, the symmetric term is the propagated formal photometric "
                r"uncertainty; the asymmetric term is the envelope obtained after including the "
                r"empirical $V_{\rm pred}$ validation scatter. "
                r"The quantity $p_{\rm UVW1}^{\rm proxy}=p_V(R_{\rm UVW1}/R_V)$ is a visible-albedo-scaled "
                r"UVW1 reflectance proxy, not an independently measured UV geometric albedo.}"
            ),
            r"\label{tab:uvw1_reflectance_results}",
            r"\end{table*}",
            "",
        ]
    )

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][REFLECTANCE] Saved: {tex_path}")
    return tex_path


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

    export_reflectance_object_latex_table(
        output_dir=output_dir,
        object_rows=object_rows,
    )

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

def _format_value_err_pair(
    value: Optional[float],
    err: Optional[float],
    sig_err: int = 2,
) -> tuple[str, str]:
    """
    Format value +/- uncertainty following the usual scientific convention:

    - uncertainty is rounded to `sig_err` significant figures;
    - value is rounded to the same decimal place as the uncertainty.

    Examples for sig_err=2:
        18.72, 0.03    -> 18.720, 0.030
        19.9,  0.1     -> 19.90,  0.10
        21.47, 0.31    -> 21.47,  0.31
        15.883, 0.025  -> 15.883, 0.025
    """
    if value is None or err is None:
        return "", ""

    if not (math.isfinite(value) and math.isfinite(err)):
        return "", ""

    if err < 0:
        err = abs(err)

    if err == 0:
        return f"{value:g}", "0"

    # Round uncertainty to the requested significant figures
    err_r = _round_sig(err, sig_err)

    if err_r == 0:
        return f"{value:g}", "0"

    exp = math.floor(math.log10(abs(err_r)))
    decimals = sig_err - 1 - exp

    # Very large numbers: round to tens/hundreds/etc.
    if decimals < 0:
        value_r = round(value, decimals)
        err_r = round(err_r, decimals)
        return f"{value_r:.0f}", f"{err_r:.0f}"

    return (
        f"{value:.{decimals}f}",
        f"{err_r:.{decimals}f}",
    )

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
        r"\small",
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



def export_quasi_simultaneous_multiband_from_config(
    config: ConfigParser,
    default_l_csv: Path,
    default_output_dir: Path,
    rocks_cache_path: Optional[Path] = None,
) -> Optional[tuple[Path, Path]]:
    """
    Regenerate the quasi-simultaneous UVW1--optical products from the latest
    per-filter photometry outputs.

    Configuration (section [PLOTS]):
        MULTIBAND_U = /path/to/U/photometry_output.csv
        MULTIBAND_B = /path/to/B/photometry_output.csv
        MULTIBAND_V = /path/to/V/photometry_output.csv

    The UVW1/L input is always PHOTOMETRY.FILEPATH (``default_l_csv``), so it
    cannot accidentally point to an older L reduction.  Rows are matched by
    normalized target name + OBSID; repeated frames in a filter are combined
    with the same inverse-variance weighted mean used elsewhere in this file.

    Outputs (by default under PLOTS.OUTPUT_DIRECTORY):
        quasi_simultaneous_multiband_current.csv
        quasi_simultaneous_multiband_current.tex

    Optional overrides:
        MULTIBAND_OUTPUT_CSV = /path/to/output.csv
        MULTIBAND_OUTPUT_TEX = /path/to/output.tex
    """
    if not config.has_section("PLOTS"):
        return None

    filter_csvs: dict[str, Path] = {"L": Path(default_l_csv).expanduser()}
    for filt in ("U", "B", "V"):
        raw = config.get("PLOTS", f"MULTIBAND_{filt}", fallback="").strip()
        if raw:
            filter_csvs[filt] = Path(raw).expanduser()

    optical_filters = [f for f in ("U", "B", "V") if f in filter_csvs]
    if not optical_filters:
        print("[PLOTS][MULTIBAND] No MULTIBAND_U/B/V paths configured; skipping.")
        return None

    data_by_filter = {
        filt: _read_filter_photometry_csv(path, filt)
        for filt, path in filter_csvs.items()
    }
    l_data = data_by_filter["L"]

    selected_keys = [
        key for key in l_data
        if any(key in data_by_filter[filt] for filt in optical_filters)
    ]
    selected_keys.sort(
        key=lambda key: (
            l_data[key]["target_name"].lower(),
            l_data[key]["observation_id"],
        )
    )

    out_csv_raw = config.get("PLOTS", "MULTIBAND_OUTPUT_CSV", fallback="").strip()
    out_tex_raw = config.get("PLOTS", "MULTIBAND_OUTPUT_TEX", fallback="").strip()
    out_csv = (
        Path(out_csv_raw).expanduser()
        if out_csv_raw
        else default_output_dir / "quasi_simultaneous_multiband_current.csv"
    )
    out_tex = (
        Path(out_tex_raw).expanduser()
        if out_tex_raw
        else default_output_dir / "quasi_simultaneous_multiband_current.tex"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = [
        "target_name", "observation_id",
        "uvw1_mag_ab", "uvw1_mag_err", "n_uvw1",
        "u_mag_ab", "u_mag_err", "n_u",
        "b_mag_ab", "b_mag_err", "n_b",
        "v_mag_ab", "v_mag_err", "n_v",
    ]

    csv_rows: list[dict[str, Any]] = []
    for key in selected_keys:
        l_entry = l_data[key]
        row: dict[str, Any] = {
            "target_name": l_entry["target_name"],
            "observation_id": l_entry["observation_id"],
            "uvw1_mag_ab": l_entry["mag"],
            "uvw1_mag_err": l_entry["mag_err"],
            "n_uvw1": l_entry["n_frames"],
        }
        for filt, prefix in (("U", "u"), ("B", "b"), ("V", "v")):
            entry = data_by_filter.get(filt, {}).get(key)
            row[f"{prefix}_mag_ab"] = entry["mag"] if entry is not None else None
            row[f"{prefix}_mag_err"] = entry["mag_err"] if entry is not None else None
            row[f"n_{prefix}"] = entry["n_frames"] if entry is not None else 0
        csv_rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[PLOTS][MULTIBAND] Saved {len(csv_rows)} rows: {out_csv}")

    # Manuscript-ready Table 2. Keep all four canonical columns even when a
    # particular filter path is not configured, so the LaTeX layout remains stable.
    rocks_cache = load_rocks_cache(rocks_cache_path) if rocks_cache_path else {}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Object & OBSID & $\langle m_{\rm AB,UVW1}\rangle$ & $\langle m_{\rm AB,U}\rangle$ & $\langle m_{\rm AB,B}\rangle$ & $\langle m_{\rm AB,V}\rangle$ \\",
        r"\midrule",
    ]

    def _tex_measurement(value: Any, err: Any) -> str:
        vv = _finite_float(value)
        ee = _finite_float(err)
        if vv is None:
            return "--"
        val_s, err_s = _format_value_err_pair(vv, ee, sig_err=2)
        if val_s and err_s:
            return rf"${val_s} \pm {err_s}$"
        return val_s or "--"

    for row in csv_rows:
        name = str(row["target_name"])
        info = rocks_cache.get(name) or _get_rocks_info(rocks_cache, name)
        number = info.get("rocks_number") if isinstance(info, dict) else None
        display_name = name
        if number not in (None, "", "None", "nan"):
            try:
                display_name = f"({int(float(number))}) {name}"
            except Exception:
                pass

        lines.append(
            " & ".join(
                [
                    _latex_escape(display_name),
                    _latex_escape(str(row["observation_id"])),
                    _tex_measurement(row["uvw1_mag_ab"], row["uvw1_mag_err"]),
                    _tex_measurement(row["u_mag_ab"], row["u_mag_err"]),
                    _tex_measurement(row["b_mag_ab"], row["b_mag_err"]),
                    _tex_measurement(row["v_mag_ab"], row["v_mag_err"]),
                ]
            ) + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Quasi-simultaneous UVW1--optical photometry of the retained "
                r"Solar System object sample. All magnitudes are given in the AB system. "
                r"When more than one valid exposure is available for a given observation "
                r"and filter, the reported value is the inverse-variance weighted mean.}"
            ),
            r"\label{tab:quasi-simultaneous-multiband}",
            r"\end{table*}",
            "",
        ]
    )
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][MULTIBAND] Saved: {out_tex}")

    return out_csv, out_tex


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
    valid_photometry_only: bool = False,
    exclude_targets: Optional[set[str]] = None,
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
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.03}")

        col_spec = "".join(
            "l" if c in {"target_name", "sso_name", "observation_id", "fits_name", "filter"}
            else "c"
            for c in columns
        )
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

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
        excluded_keys = {_rocks_lookup_key(x) for x in (exclude_targets or set())}

        for row in reader:
            # The manuscript table is a record of actual frame-level photometry,
            # not of screening placeholders/null rows.
            if valid_photometry_only and _to_float(row.get("mag_ab")) is None:
                continue

            if excluded_keys:
                row_name = row.get("target_name") or row.get("sso_name") or ""
                if _rocks_lookup_key(str(row_name)) in excluded_keys:
                    continue

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
        lines.append(rf"\label{{{label}}}")
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