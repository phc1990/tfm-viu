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

    latex_columns = [
        "target_name",
        "observation_id",
        "fits_name",
        "count_rate",
        "count_rate_err",
        "trail_height_pix",
        "mag_ab_apcorr",
        "mag_ab_apcorr_err",
        "v_mag_1",
        "mlim_obs",
    ]

    out_tex = out_dir / "photometry_output_table.tex"
    export_photometry_csv_to_latex(
        phot_csv=phot_csv,
        out_tex=out_tex,
        columns=latex_columns,
        caption="Photometry output exported from the pipeline.",
        label="tab:photometry_output",
        max_rows=None,  # o pon un número si quieres truncar
    )



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

        colors: list[float] = []
        colors.append(color_mean)  # UV−V


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


        c_colors = [col for col, bb in zip(colors, buckets) if bb == "C"]
        s_colors = [col for col, bb in zip(colors, buckets) if bb == "S"]

        if len(c_colors) >= 1 and len(s_colors) >= 1:
            muC, sC = float(np.mean(c_colors)), float(np.std(c_colors, ddof=1))
            muS, sS = float(np.mean(s_colors)), float(np.std(s_colors, ddof=1))
            thr = gaussian_intersection(muC, sC, muS, sS)
            print(f"[PLOTS][THR] UV−V: muC={muC:.3f}±{sC:.3f}, muS={muS:.3f}±{sS:.3f}, threshold≈{thr:.3f}")

            # opcional: dibuja líneas de “color constante” en el diagrama y vs x
            if thr is not None:
                # y = x + thr
                xx = np.linspace(min(xs), max(xs), 100)
                yy = xx + thr
                plt.plot(xx, yy, linestyle="--")
        else:
            print("[PLOTS][THR] Not enough C/S points to estimate threshold robustly.")

        fit = fit_line(xb, yb)

        if fit is not None:
            m, c = fit
            xx = np.linspace(min(xb), max(xb), 50)
            yy = m * xx + c
            plt.plot(xx, yy, linestyle="-")  # color por defecto
            print(f"[PLOTS][FIT] {b}: slope={m:.4f}, intercept={c:.4f}, N={len(xb)}")
        else:
            print(f"[PLOTS][FIT] {b}: not enough points (N={len(xb)})")

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

def fit_line(x: list[float], y: list[float]) -> Optional[tuple[float, float]]:
    if len(x) < 3:
        return None
    xarr = np.array(x, dtype=float)
    yarr = np.array(y, dtype=float)
    m, b = np.polyfit(xarr, yarr, 1)  # y = m x + b
    return float(m), float(b)


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


def export_photometry_csv_to_latex(
    phot_csv: Path,
    out_tex: Path,
    columns: list[str],
    caption: str = "Photometry results.",
    label: str = "tab:photometry",
    max_rows: Optional[int] = None,
) -> None:
    """
    Export selected columns from photometry_output.csv to a LaTeX table.
    - Uses tabular with \\toprule/\\midrule/\\bottomrule (booktabs).
    - Escapes LaTeX special chars.
    """
    if not phot_csv.exists():
        raise FileNotFoundError(f"CSV not found: {phot_csv}")

    out_tex.parent.mkdir(parents=True, exist_ok=True)

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
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        # 14 columnas -> lo más seguro es usar \scriptsize y tabular con l's.
        lines.append(r"\scriptsize")
        col_spec = "l" * len(columns)
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        # Header
        header = " & ".join(_latex_escape(c) for c in columns) + r" \\"
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
            cr = _to_float(row.get("count_rate")) if "count_rate" in columns else None
            cr_err = _to_float(row.get("count_rate_err")) if "count_rate_err" in columns else None
            mab = _to_float(row.get("mag_ab_apcorr")) if "mag_ab_apcorr" in columns else None
            mab_err = _to_float(row.get("mag_ab_apcorr_err")) if "mag_ab_apcorr_err" in columns else None

            cr_s, cr_err_s = _format_value_err_pair(cr, cr_err, sig_err=2)
            mab_s, mab_err_s = _format_value_err_pair(mab, mab_err, sig_err=2)

            vals = []
            for c in columns:
                # Apply special formatting for the two value/error pairs
                if c == "count_rate":
                    v_str = cr_s
                elif c == "count_rate_err":
                    v_str = cr_err_s
                elif c == "mag_ab_apcorr":
                    v_str = mab_s
                elif c == "mag_ab_apcorr_err":
                    v_str = mab_err_s
                else:
                    v = row.get(c, "")
                    if v is None:
                        v = ""
                    v_str = str(v).strip()
                    if v_str.lower() in {"none", "nan", "null"}:
                        v_str = ""

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
        lines.append("")  # trailing newline

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PLOTS][LATEX] Saved: {out_tex}")



if __name__ == "__main__":
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.ini")
    config = _load_config(cfg_path)
    action_plots(config)
