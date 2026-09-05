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

from src.photometry.phot import PhotTable, predicted_om_ab_mag


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
    out_png_family = out_dir / "uvw1_vpred_colour_by_taxonomy.png"
    out_png_scatter = out_dir/"uvw1_vs_vpred_taxonomy.png"
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
        reader = csv.DictReader(f)
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

            # Ignore manually commented-out photometry rows.
            if name_raw.startswith("#"):
                continue

            name = _normalize_name(name_raw)
            if not name:
                continue

            # Keep every object for table/name diagnostics, even if not plottable.
            all_table_names.append(name)

            v = _to_float(row.get("v_mag_1"))
            y = _to_float(row.get(mag_col))
            yerr = _to_float(row.get(mag_err_col))

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
    xs: list[float] = []
    ys: list[float] = []
    xerrs: list[Optional[float]] = []
    yerrs: list[Optional[float]] = []
    uv_minus_v: list[float] = []
    uv_minus_v_errs: list[Optional[float]] = []
    buckets: list[str] = []
    tax_labels: list[str] = []

    resolved_tax = 0
    resolved_pv = 0
    resolved_d = 0


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
        color_vals = [
            r["color"]
            for r in rows_for_obj
            if r["color"] is not None
        ]

        color_err_vals = [
            r["yerr"]
            for r in rows_for_obj
            if r["color"] is not None
        ]


        if not v_vals or not y_vals:
            continue

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

        color_obj, color_obj_err = _weighted_mean_and_err(color_vals, color_err_vals,)

        info = _get_rocks_info(rocks_info, name)
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
        xerrs.append(xerr)
        yerrs.append(yerr)
        uv_minus_v.append(color_obj)
        uv_minus_v_errs.append(color_obj_err)
        buckets.append(_taxonomy_bucket(tax_class))
        tax_labels.append(str(tax_class or "UNK"))

    export_taxonomy_colour_summary_tables(
        output_dir=out_dir,
        colors=uv_minus_v,
        color_errs=uv_minus_v_errs,
        buckets=buckets,
    )

    # ------------------------------------------------------------
    # 4) UVW1 - Vpred colour distribution by taxonomic family
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Families shown in the plot.
    # UNK is deliberately excluded because it is not a physical
    # taxonomic family.
    family_order = [
        fam
        for fam in ("C", "S", "X", "B", "D", "V", "K", "L", "A", "Q", "OTHER")
        if fam in buckets
    ]

    # Main scientific comparison and secondary families of interest.
    primary_families = {"C", "S"}
    secondary_families = {"X", "B"}

    # Deterministic horizontal offsets so overlapping objects remain visible.
    # No random jitter -> fully reproducible figure.
    def _x_offsets(n: int, width: float = 0.22) -> np.ndarray:
        if n <= 1:
            return np.array([0.0])
        return np.linspace(-width, width, n)

    family_styles = {
        "C": {
            "point_color": "#4C78A8",
            "summary_color": "#1F4E79",
        },
        "S": {
            "point_color": "#F28E2B",
            "summary_color": "#C55A11",
        },
        "X": {
            "point_color": "#59A14F",
            "summary_color": "#2F6B2F",
        },
        "B": {
            "point_color": "#B07AA1",
            "summary_color": "#7A4E75",
        },
    }

    default_style = {
        "point_color": "0.75",
        "summary_color": "0.35",
    }

    for xpos, fam in enumerate(family_order):
        idx = [
            i for i, bb in enumerate(buckets)
            if bb == fam
        ]

        if not idx:
            continue

        colours = np.array(
            [uv_minus_v[i] for i in idx],
            dtype=float,
        )

        colour_errs = np.array(
            [
                np.nan if uv_minus_v_errs[i] is None
                else float(uv_minus_v_errs[i])
                for i in idx
            ],
            dtype=float,
        )

        offsets = _x_offsets(len(idx))
        x_points = xpos + offsets

        # --------------------------------------------------------
        # Individual asteroid colours
        # --------------------------------------------------------
        # if fam in primary_families:
        #     point_kwargs = {
        #         "markersize": 5.5,
        #         "alpha": 0.95,
        #     }
        # elif fam in secondary_families:
        #     point_kwargs = {
        #         "markersize": 5.0,
        #         "alpha": 0.75,
        #     }
        # else:
        #     point_kwargs = {
        #         "markersize": 4.0,
        #         "alpha": 0.40,
        #         "color": "0.60",
        #         "ecolor": "0.70",
        #     }
        style = family_styles.get(fam, default_style)

        point_color = style["point_color"]
        summary_color = style["summary_color"]

        valid_err = np.isfinite(colour_errs) & (colour_errs > 0)

        if np.any(valid_err):
            yerr_plot = np.where(valid_err, colour_errs, 0.0)
        else:
            yerr_plot = None

        # ax.errorbar(
        #     x_points,
        #     colours,
        #     yerr=yerr_plot,
        #     fmt="o",
        #     capsize=2,
        #     elinewidth=0.7,
        #     linewidth=0.7,
        #     **point_kwargs,
        # )
        ax.errorbar(
            x_points,
            colours,
            yerr=yerr_plot,
            fmt="o",
            markersize=3.8,
            capsize=2,
            elinewidth=0.7,
            linewidth=0.7,
            alpha=0.45,
            color=point_color,
            ecolor=point_color,
            markerfacecolor=point_color,
            markeredgecolor=point_color,
        )
        # --------------------------------------------------------
        # Family median and percentile scatter
        # --------------------------------------------------------
        median_colour = float(np.median(colours))

        if len(colours) >= 2:
            p16, p84 = np.percentile(colours, [16, 84])
            scatter = float(0.5 * (p84 - p16))
            ax.errorbar(
                xpos,
                median_colour,
                yerr=scatter,
                fmt="_",
                markersize=3.2,
                markeredgewidth=0.0,
                capsize=4,
                elinewidth=1.5,
                linewidth=1.5,
                color=summary_color,
                ecolor=summary_color,
                zorder=5,
            )

        else:
            scatter = math.nan
            ax.plot(
                xpos,
                median_colour,
                marker="_",
                markersize=13,
                markeredgewidth=2.2,
                color=summary_color,
                zorder=5,
            )

        # This must be outside the if/else above.
        if math.isfinite(scatter):
            scatter_text = f"{scatter:.3f}"
        else:
            scatter_text = "--"

        print(
            f"[PLOTS][COLOUR FIG] {fam}: "
            f"N={len(colours)} "
            f"median={median_colour:.3f} "
            f"scatter16-84={scatter_text}"
        )


    ax.set_xticks(range(len(family_order)))
    ax.set_xticklabels(family_order)

    ax.set_xlabel("Taxonomic family")
    ax.set_ylabel(r"$m_{\rm UVW1,AB} - V_{\rm pred}$ (mag)")

    # Zero/reference grid only for readability.
    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    fig.tight_layout()
    fig.savefig(out_png_family, dpi=200)
    plt.close(fig)

    print(f"[PLOTS] Read: {phot_csv}")
    print(f"[PLOTS] Total rows: {total_rows} | Valid rows: {valid_rows}")
    n_points_plotted = sum(
        1 for b in buckets
        if b in family_order
    )

    print(
        f"[PLOTS] Objects with valid colours: {len(uv_minus_v)} "
        f"| Points plotted: {n_points_plotted}"
    )

    print(
        f"[PLOTS] rocks resolved: "
        f"taxonomy={resolved_tax}/{len(measurements)}, "
        f"pV={resolved_pv}/{len(measurements)}, "
        f"D={resolved_d}/{len(measurements)}"
    )
    print(f"[PLOTS] Saved: {out_png_family}")

    # return out_png_family

    # ------------------------------------------------------------
    # 5) Vpred vs UVW1 scatter plot by taxonomy (paper figure)
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    # --------------------------------------------------------
    # Family statistics from the same object-level colours used
    # in Table 3. No hardcoded scientific values.
    # --------------------------------------------------------
    family_stats = {}
    for fam in sorted(set(buckets)):
        vals = np.array(
            [uv_minus_v[i] for i, bb in enumerate(buckets) if bb == fam],
            dtype=float,
        )
        if len(vals) == 0:
            continue

        median_val = float(np.median(vals))

        if len(vals) >= 2:
            p16, p84 = np.percentile(vals, [16, 84])
            scatter_val = float(0.5 * (p84 - p16))
        else:
            scatter_val = math.nan

        family_stats[fam] = {
            "N": len(vals),
            "median": median_val,
            "scatter16_84": scatter_val,
        }

    # --------------------------------------------------------
    # Visual style
    # --------------------------------------------------------
    style_map = {
        "C": {
            "point_color": "#4C78A8",
            "line_color": "#1F4E79",
            "label": "C-complex",
            "alpha": 0.65,
            "markersize": 4.5,
            "zorder": 4,
        },
        "S": {
            "point_color": "#F28E2B",
            "line_color": "#C55A11",
            "label": "S-complex",
            "alpha": 0.65,
            "markersize": 4.5,
            "zorder": 4,
        },
        "X": {
            "point_color": "#59A14F",
            "line_color": "#2F6B2F",
            "label": "X type",
            "alpha": 0.50,
            "markersize": 4.0,
            "zorder": 3,
        },
        "B": {
            "point_color": "#B07AA1",
            "line_color": "#7A4E75",
            "label": "B type",
            "alpha": 0.50,
            "markersize": 4.0,
            "zorder": 3,
        },
    }

    default_style = {
        "point_color": "0.70",
        "line_color": "0.40",
        "label": "Other taxonomies",
        "alpha": 0.35,
        "markersize": 3.5,
        "zorder": 2,
    }

    highlight_families = ["C", "S"]
    secondary_families = ["X", "B"]

    # --------------------------------------------------------
    # First plot "other" taxonomies in grey background
    # (known taxonomy only, excluding C/S/X/B and UNK)
    # --------------------------------------------------------
    other_idx = [
        i for i, b in enumerate(buckets)
        if b not in highlight_families
        and b not in secondary_families
        and b != "UNK"
    ]

    if other_idx:
        xb = [xs[i] for i in other_idx]
        yb = [ys[i] for i in other_idx]
        yeb = [yerrs[i] if yerrs[i] is not None else 0.0 for i in other_idx]
        has_yerr = any(e > 0 for e in yeb)

        ax.errorbar(
            xb,
            yb,
            yerr=yeb if has_yerr else None,
            fmt="o",
            markersize=default_style["markersize"],
            capsize=2,
            elinewidth=0.7,
            linewidth=0.7,
            alpha=default_style["alpha"],
            color=default_style["point_color"],
            ecolor=default_style["point_color"],
            markerfacecolor=default_style["point_color"],
            markeredgecolor=default_style["point_color"],
            label=default_style["label"],
            zorder=default_style["zorder"],
        )

    # --------------------------------------------------------
    # Then plot X and B as secondary families
    # --------------------------------------------------------
    for fam in secondary_families:
        idx = [i for i, bb in enumerate(buckets) if bb == fam]
        if not idx:
            continue

        style = style_map[fam]

        xb = [xs[i] for i in idx]
        yb = [ys[i] for i in idx]
        yeb = [yerrs[i] if yerrs[i] is not None else 0.0 for i in idx]
        has_yerr = any(e > 0 for e in yeb)

        ax.errorbar(
            xb,
            yb,
            yerr=yeb if has_yerr else None,
            fmt="o",
            markersize=style["markersize"],
            capsize=2,
            elinewidth=0.7,
            linewidth=0.7,
            alpha=style["alpha"],
            color=style["point_color"],
            ecolor=style["point_color"],
            markerfacecolor=style["point_color"],
            markeredgecolor=style["point_color"],
            label=style["label"],
            zorder=style["zorder"],
        )

    # --------------------------------------------------------
    # Finally plot C and S highlighted
    # --------------------------------------------------------
    for fam in highlight_families:
        idx = [i for i, bb in enumerate(buckets) if bb == fam]
        if not idx:
            continue

        style = style_map[fam]

        xb = [xs[i] for i in idx]
        yb = [ys[i] for i in idx]
        yeb = [yerrs[i] if yerrs[i] is not None else 0.0 for i in idx]
        has_yerr = any(e > 0 for e in yeb)

        ax.errorbar(
            xb,
            yb,
            yerr=yeb if has_yerr else None,
            fmt="o",
            markersize=style["markersize"],
            capsize=2,
            elinewidth=0.8,
            linewidth=0.8,
            alpha=style["alpha"],
            color=style["point_color"],
            ecolor=style["point_color"],
            markerfacecolor=style["point_color"],
            markeredgecolor=style["point_color"],
            label=style["label"],
            zorder=style["zorder"],
        )

    # --------------------------------------------------------
    # Overplot colour-constant lines for C and S medians:
    # m_UVW1 = Vpred + median(UVW1 - Vpred)
    # --------------------------------------------------------
    xmin = min(xs)
    xmax = max(xs)
    xline = np.linspace(xmin, xmax, 200)

    for fam in highlight_families:
        if fam not in family_stats:
            continue

        style = style_map[fam]
        med = family_stats[fam]["median"]

        ax.plot(
            xline,
            xline + med,
            linestyle="--",
            linewidth=1.4,
            color=style["line_color"],
            label=rf"{fam} median colour = {med:.3f} mag",
            zorder=1,
        )

    # --------------------------------------------------------
    # Labels and cosmetics
    # --------------------------------------------------------
    ax.set_xlabel(r"$V_{\rm pred}$ (mag)")
    ax.set_ylabel(r"$m_{\rm UVW1,AB}$ (mag)")

    # ax.set_title(r"Predicted visible magnitude vs. UVW1 photometry")

    ax.grid(
        alpha=0.20,
        linewidth=0.6,
    )

    ax.legend(
        fontsize=8,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(out_png_scatter, dpi=200)
    plt.close(fig)

    print(f"[PLOTS] Saved: {out_png_scatter}")


    plot_vpred_validation(
        config=config,
        output_dir=out_dir,
    )


def _load_config(config_path: Path) -> ConfigParser:
    cfg = ConfigParser()
    read_ok = cfg.read(config_path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    return cfg

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



# def _normalize_filter_code(value: object) -> str:
#     """Normalize OM filter names/codes to L, M, S, U, B or V."""
#     raw = str(value or "").strip().upper()
#     aliases = {
#         "L": "L",
#         "UVW1": "L",
#         "W1": "L",
#         "M": "M",
#         "UVM2": "M",
#         "M2": "M",
#         "S": "S",
#         "UVW2": "S",
#         "W2": "S",
#         "U": "U",
#         "B": "B",
#         "V": "V",
#     }
#     return aliases.get(raw, raw)


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

def plot_vpred_validation(
    config: ConfigParser,
    output_dir: Path,
) -> Optional[Path]:
    """
    Validate the SSOSS theoretical V prediction against real OM V-band
    photometry.

    For each asteroid/observation pair:
      1. Repeated OM-V frames are combined with the existing
         inverse-variance weighted mean.
      2. v_mag_1 (= V_pred) is transformed to the OM V filter
         in the AB system.
      3. The residual is defined as:

            Delta V = m_V,AB(obs) - m_V,AB(pred)

    Outputs
    -------
    - vpred_validation.csv
    - vpred_vs_omv_validation.png
    """

    # ---------------------------------------------------------
    # Locate V-band photometry CSV
    # ---------------------------------------------------------
    v_csv = Path(
    config["PHOTOMETRY"]["FILEPATH"]
    ).expanduser()

    if not v_csv.exists():
        print(
            f"[PLOTS][VPRED] V photometry CSV not found: {v_csv}"
        )
        return None

    if not v_csv:
        print(
            "[PLOTS][VPRED] No MULTIFILTER_V_FILE configured; "
            "Vpred validation skipped."
        )
        return None

    v_csv = Path(v_csv).expanduser()

    if not v_csv.exists():
        print(
            f"[PLOTS][VPRED] V photometry CSV not found: {v_csv}"
        )
        return None

    # Existing helper:
    # one entry per (asteroid, observation_id),
    # with repeated V frames already combined.
    v_data = _read_filter_photometry_csv(
        csv_path=v_csv,
        expected_filter="V",
    )

    # ---------------------------------------------------------
    # Build validation sample
    # ---------------------------------------------------------
    rows: list[dict[str, Any]] = []

    for key, entry in sorted(
        v_data.items(),
        key=lambda item: (
            item[1]["target_name"].lower(),
            item[1]["observation_id"],
        ),
    ):
        v_pred = entry.get("v_pred")
        v_obs = entry.get("mag")
        v_obs_err = entry.get("mag_err")

        if v_pred is None or v_obs is None:
            continue

        # Transform theoretical Vpred to expected OM-V magnitude
        # in exactly the same AB system as our measured photometry.
        try:
            v_pred_om_ab = predicted_om_ab_mag(
                config,
                float(v_pred),
                "V",
            )
        except Exception as exc:
            print(
                f"[PLOTS][VPRED] WARN: prediction failed for "
                f"{entry['target_name']} "
                f"{entry['observation_id']}: {exc}"
            )
            continue

        delta_v = float(v_obs) - float(v_pred_om_ab)

        rows.append(
            {
                "target_name": entry["target_name"],
                "observation_id": entry["observation_id"],
                "n_v_frames": entry["n_frames"],
                "v_pred": float(v_pred),
                "v_pred_om_ab": float(v_pred_om_ab),
                "v_obs_ab": float(v_obs),
                "v_obs_ab_err": (
                    float(v_obs_err)
                    if v_obs_err is not None
                    else None
                ),
                "delta_v": delta_v,
            }
        )

    if not rows:
        print(
            "[PLOTS][VPRED] No valid asteroid/OBSID pairs "
            "with both OM-V photometry and Vpred."
        )
        return None

    # ---------------------------------------------------------
    # Global diagnostic statistics
    # ---------------------------------------------------------
    delta = np.asarray(
        [r["delta_v"] for r in rows],
        dtype=float,
    )

    median_delta = float(np.median(delta))
    mean_delta = float(np.mean(delta))

    if len(delta) >= 2:
        std_delta = float(np.std(delta, ddof=1))

        p16, p84 = np.percentile(
            delta,
            [16, 84],
        )
        scatter_16_84 = float(
            0.5 * (p84 - p16)
        )
    else:
        std_delta = math.nan
        scatter_16_84 = math.nan

    print("")
    print("[PLOTS][VPRED] --------------------------------")
    print(
        f"[PLOTS][VPRED] Validation sample: N={len(rows)}"
    )
    print(
        f"[PLOTS][VPRED] mean DeltaV   = "
        f"{mean_delta:+.3f} mag"
    )
    print(
        f"[PLOTS][VPRED] median DeltaV = "
        f"{median_delta:+.3f} mag"
    )

    if math.isfinite(std_delta):
        print(
            f"[PLOTS][VPRED] std          = "
            f"{std_delta:.3f} mag"
        )

    if math.isfinite(scatter_16_84):
        print(
            f"[PLOTS][VPRED] s16-84       = "
            f"{scatter_16_84:.3f} mag"
        )

    print("[PLOTS][VPRED] Individual measurements:")

    for r in rows:
        err = r["v_obs_ab_err"]

        err_txt = (
            f" +/- {err:.3f}"
            if err is not None
            else ""
        )

        print(
            f"[PLOTS][VPRED] "
            f"{r['target_name']} "
            f"obs={r['observation_id']} | "
            f"Vpred={r['v_pred']:.3f} | "
            f"Vpred_OM_AB={r['v_pred_om_ab']:.3f} | "
            f"Vobs_AB={r['v_obs_ab']:.3f}{err_txt} | "
            f"DeltaV={r['delta_v']:+.3f}"
        )

    # ---------------------------------------------------------
    # Export diagnostic CSV
    # ---------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = output_dir / "vpred_validation.csv"

    fieldnames = [
        "target_name",
        "observation_id",
        "n_v_frames",
        "v_pred",
        "v_pred_om_ab",
        "v_obs_ab",
        "v_obs_ab_err",
        "delta_v",
    ]

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[PLOTS][VPRED] CSV saved: {out_csv}"
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    x = np.asarray(
        [r["v_pred_om_ab"] for r in rows],
        dtype=float,
    )

    y = np.asarray(
        [r["v_obs_ab"] for r in rows],
        dtype=float,
    )

    yerr = np.asarray(
        [
            np.nan
            if r["v_obs_ab_err"] is None
            else float(r["v_obs_ab_err"])
            for r in rows
        ],
        dtype=float,
    )

    residual = y - x

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.3, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2.2, 1.0],
        },
    )

    # ---------------------------------------------------------
    # Upper panel: observed vs predicted
    # ---------------------------------------------------------
    valid_err = (
        np.isfinite(yerr)
        & (yerr > 0)
    )

    yerr_plot = np.where(
        valid_err,
        yerr,
        0.0,
    )

    ax1.errorbar(
        x,
        y,
        yerr=yerr_plot,
        fmt="o",
        markersize=5.5,
        capsize=3,
        elinewidth=0.9,
        linewidth=0.9,
    )

    # 1:1 relation
    all_values = np.concatenate([x, y])

    pad = 0.15
    lim_min = float(np.min(all_values) - pad)
    lim_max = float(np.max(all_values) + pad)

    line = np.linspace(
        lim_min,
        lim_max,
        200,
    )

    ax1.plot(
        line,
        line,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
 #       label="1:1 relation",
    )

    ax1.set_xlim(
        lim_min,
        lim_max,
    )
    ax1.set_ylim(
        lim_min,
        lim_max,
    )

    ax1.set_ylabel(
        r"Observed $m_{\rm V,AB}$ (mag)"
    )

 #   ax1.set_title(
 #       r"Validation of $V_{\rm pred}$ with OM $V$ photometry"
 #   )

    ax1.grid(
        alpha=0.20,
        linewidth=0.6,
    )

    ax1.legend(
        frameon=False,
    )

    # Label objects while the validation sample is small.
    if len(rows) <= 12:
        for xx, yy, r in zip(x, y, rows):
            ax1.annotate(
                r["target_name"],
                (xx, yy),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    # ---------------------------------------------------------
    # Lower panel: residuals
    # ---------------------------------------------------------
    ax2.errorbar(
        x,
        residual,
        yerr=yerr_plot,
        fmt="o",
        markersize=5.0,
        capsize=3,
        elinewidth=0.9,
        linewidth=0.9,
    )

    # Perfect agreement
    ax2.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="0.35",
    )

    # Sample median residual
    ax2.axhline(
        median_delta,
        linestyle=":",
        linewidth=1.2,
        label=(
            rf"Median $\Delta V={median_delta:+.3f}$ mag"
        ),
    )

    ax2.set_xlabel(
        r"Predicted OM $V$ magnitude, "
        r"$m_{\rm V,pred,AB}$ (mag)"
    )

    ax2.set_ylabel(
        r"$\Delta V$ (mag)"
    )

    ax2.grid(
        alpha=0.20,
        linewidth=0.6,
    )

    ax2.legend(
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    out_png = (
        output_dir
        / "vpred_vs_omv_validation.png"
    )

    fig.savefig(
        out_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[PLOTS][VPRED] Plot saved: {out_png}"
    )
    print("[PLOTS][VPRED] --------------------------------")
    print("")

    return out_png


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
        reader = csv.DictReader(f)
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
            expected_band = PhotTable.om_filter_to_band(expected_filter)

            if not expected_band:
                raise ValueError(
                    f"Unsupported OM filter: {expected_filter!r}"
                )
            # row_filter = _normalize_filter_code(row.get("filter")) if "filter" in fieldnames else expected_filter

            row_filter_raw = str(row.get("filter") or "").strip()
            row_band = PhotTable.om_filter_to_band(row_filter_raw)

            if row_band != expected_band:
                continue

            name_raw = str(row.get(name_col) or "").strip()

            # Ignore manually disabled rows.
            if name_raw.startswith("#"):
                continue

            name = _normalize_name(name_raw)
            obs_id = str(row.get("observation_id") or "").strip()

            mag = _to_float(row.get("mag_ab"))
            mag_err = _to_float(row.get("mag_err"))
            v_pred = _to_float(row.get("v_mag_1"))

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
                    "v_pred_values": [],
                },
            )

            entry["values"].append(mag)
            entry["errors"].append(mag_err)

            if v_pred is not None:
                entry["v_pred_values"].append(v_pred)

    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in grouped.items():
        mean, mean_err, n_frames = _mean_measurement(
            entry["values"],
            entry["errors"],
        )

        v_pred_values = np.asarray(
            entry["v_pred_values"],
            dtype=float,
        )

        if len(v_pred_values) > 0:
            # v_mag_1 should normally be identical for all frames
            # belonging to the same asteroid/OBSID.
            v_pred = float(np.mean(v_pred_values))
        else:
            v_pred = None

        combined[key] = {
            "target_name": entry["target_name"],
            "observation_id": entry["observation_id"],
            "mag": mean,
            "mag_err": mean_err,
            "n_frames": n_frames,
            "v_pred": v_pred,
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
        reader = csv.DictReader(f)
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
        lines.append(rf"\caption{{{(caption)}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table*}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][LATEX] Saved: {out_tex}")

def export_taxonomy_colour_summary_tables(
    output_dir: Path,
    colors: list[float],
    color_errs: list[Optional[float]],
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

        col_vals = np.array(
            [colors[i] for i in idx],
            dtype=float,
        )

        err_vals = np.array(
            [
                np.nan if color_errs[i] is None else float(color_errs[i])
                for i in idx
            ],
            dtype=float,
        )

        n = len(col_vals)

        mean_col = float(np.mean(col_vals))
        median_col = float(np.median(col_vals))

        if n >= 2:
            std_col = float(np.std(col_vals, ddof=1))
            sem_col = float(std_col / math.sqrt(n))

            p16, p84 = np.percentile(col_vals, [16, 84])
            percentile_scatter = float(0.5 * (p84 - p16))
        else:
            std_col = math.nan
            sem_col = math.nan
            percentile_scatter = math.nan

        rows.append(
            {
                "family": fam,
                "N": n,
                "mean_UVW1_minus_V": mean_col,
                "median_UVW1_minus_V": median_col,
                "std_UVW1_minus_V": std_col,
                "sem_UVW1_minus_V": sem_col,
                "percentile_scatter_UVW1_minus_V": percentile_scatter,
            }
        )

    csv_path = output_dir / "taxonomy_colour_summary.csv"
    tex_path = output_dir / "taxonomy_colour_summary.tex"

    fieldnames = [
        "family",
        "N",
        "mean_UVW1_minus_V",
        "median_UVW1_minus_V",
        "std_UVW1_minus_V",
        "sem_UVW1_minus_V",
        "percentile_scatter_UVW1_minus_V",
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
        f.write("\\caption{UVW1--$V_{\\rm pred}$ colour by taxonomic family.}\n")
        f.write("\\label{tab:taxonomy_colour_summary}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Family & $N$ & "
            "$\\mathrm{median}(\\mathrm{UVW1}-V_{\\rm pred})$ & "
            "$\\sigma_{16-84}$ \\\\\n"
        )
        f.write("\\hline\n")

        for r in rows:
            fam = r["family"]
            n = r["N"]

            median_col = _fmt(r["median_UVW1_minus_V"])
            scatter_col = _fmt(r["percentile_scatter_UVW1_minus_V"])

            f.write(
                f"{fam} & {n} & {median_col} & {scatter_col} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\n")
        f.write(
            "\\vspace{0.5ex}\n"
            "\\footnotesize\n"
            "The reported colour is the median of the object-level "
            "$\\mathrm{UVW1}-V_{\\rm pred}$ colours within each taxonomic family. "
            "The scatter is defined as "
            "$\\sigma_{16-84}=(P_{84}-P_{16})/2$.\n"
        )
        f.write("\\end{table}\n")

    print(f"[PLOTS][LATEX] Saved: {tex_path}")
    print(f"[PLOTS][CSV] Saved: {csv_path}")

    print("[PLOTS][COLOUR] Taxonomy colour summary:")
    for r in rows:
        print(
            f"  {r['family']:>5s} "
            f"N={r['N']:2d} "
            f"median(UVW1-Vpred)={_fmt(r['median_UVW1_minus_V'])} "
            f"scatter16-84={_fmt(r['percentile_scatter_UVW1_minus_V'])}"
        )

if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
    config = _load_config(cfg_path)
    action_plots(config)
