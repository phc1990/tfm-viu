#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, TextIO

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astroquery.imcce import Skybot


DEFAULT_BASE = Path("/Users/eracero/workspace/tfm-viu")
DEFAULT_RESULTS = DEFAULT_BASE / "results/batch_L/elena"
DEFAULT_TEMP = DEFAULT_BASE / "temp"
DEFAULT_SCREENING_CSV = DEFAULT_RESULTS / "screening.csv"

LOCATION = "xmm-newton"

CSV_COLUMNS = [
    "observation_id",
    "sso_name",
    "ra_deg_1",
    "dec_deg_1",
    "ra_deg_2",
    "dec_deg_2",
    "filter",
    "xmatch_type",
    "v_mag_1",
    "v_mag_1_corrected",
    "mlim_obs",
    "FITS_FILE",
    "DECISION",
    "dra_arcsec",
    "ddec_arcsec",
    "trail_len_arcsec",
    "pa_deg_E_of_N",
    "frame_t1",
    "frame_t2",
    "exposure_s",
    "skybot_radius_arcmin",
    "sep_start_arcsec_from_frame_center",
    "sep_end_arcsec_from_frame_center",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# -----------------------------
# generic helpers
# -----------------------------
def norm(x) -> str:
    return str(x or "").strip()


def norm_name(x) -> str:
    """Normalize object names from CSV/SkyBoT.

    Examples:
      "(28482) Dowling" -> "dowling"
      "Dowling"         -> "dowling"
    """
    s = norm(x)
    import re
    s = re.sub(r"^\(\s*\d+\s*\)\s*", "", s)
    return s.lower().replace(" ", "_")


def to_float(x) -> float:
    try:
        s = norm(x)
        if not s:
            return math.nan
        return float(s)
    except Exception:
        return math.nan


def as_float(x) -> float:
    """Robust conversion for astropy Table scalar values."""
    try:
        if getattr(x, "mask", False):
            return math.nan
    except Exception:
        pass
    try:
        if hasattr(x, "value"):
            return as_float(x.value)
    except Exception:
        pass
    try:
        if hasattr(x, "item"):
            return float(x.item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return math.nan


def get_ra_dec(row) -> tuple[float, float]:
    """Prefer RA/DEC returned by SkyBoT, fallback to _raj2000/_decj2000."""
    colnames = set(getattr(row, "colnames", []))
    ra = as_float(row["RA"]) if "RA" in colnames else math.nan
    dec = as_float(row["DEC"]) if "DEC" in colnames else math.nan
    if (not math.isfinite(ra) or not math.isfinite(dec)) and {"_raj2000", "_decj2000"} <= colnames:
        ra = as_float(row["_raj2000"])
        dec = as_float(row["_decj2000"])
    return ra, dec


def skybot_name(row) -> str:
    for key in ("Name", "name", "Object", "object"):
        try:
            return norm(row[key])
        except Exception:
            pass
    return "UNKNOWN"


def skybot_vmag(row) -> str:
    """Return the best apparent V magnitude column available from SkyBoT."""
    colnames = set(getattr(row, "colnames", []))
    for key in ("V", "Mv", "VMag", "Vmag", "magV", "mag", "mV"):
        if key in colnames:
            v = as_float(row[key])
            if math.isfinite(v):
                return f"{v:.3f}"
    return ""


# -----------------------------
# screening/FITS helpers
# -----------------------------
def find_fits_path(temp_dir: Path, fits_name: str) -> Path:
    matches = list(temp_dir.rglob(fits_name))
    if not matches:
        raise FileNotFoundError(f"Could not find FITS in {temp_dir}: {fits_name}")
    if len(matches) > 1:
        log("[WARN] Multiple FITS matches found; using first:")
        for m in matches:
            log(f"  {m}")
    return matches[0]


def get_times_from_fits(fits_path: Path) -> tuple[Time, Time, float]:
    """Return frame start/end time using FITS DATE-OBS + exposure.

    This is intentionally frame-based, not observation-level metadata-based.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        hdrs = [h.header for h in hdul if hasattr(h, "header")]

        date_obs = None
        for h in hdrs:
            date_obs = h.get("DATE-OBS") or h.get("DATE_OBS")
            if date_obs:
                break

        exposure = None
        for h in hdrs:
            for k in ("EXPOSURE", "EXPTIME", "ONTIME", "EXP_TIME", "TELAPSE"):
                if h.get(k) is not None:
                    try:
                        exposure = float(h.get(k))
                        if math.isfinite(exposure) and exposure > 0:
                            break
                    except Exception:
                        pass
            if exposure is not None:
                break

        if exposure is None:
            h0 = hdul[0].header
            if h0.get("MJD-END") is not None and h0.get("MJD-OBS") is not None:
                exposure = (float(h0["MJD-END"]) - float(h0["MJD-OBS"])) * 86400.0

    if not date_obs:
        raise ValueError(f"DATE-OBS missing in {fits_path}")
    if exposure is None or not math.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"Exposure time missing/invalid in {fits_path}")

    t1 = Time(str(date_obs), scale="utc")
    t2 = t1 + float(exposure) * u.s
    return t1, t2, float(exposure)


def get_image_cone_from_fits(fits_path: Path, margin_arcmin: float = 5.0) -> tuple[SkyCoord, u.Quantity]:
    """Cone-search center/radius from the actual FITS WCS footprint.

    For xtype=3 rows, the old catalogue start/end positions can be outside the image.
    Therefore we query around the image center, not around the catalogue path midpoint.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        hdu = hdul[0]
        data = hdu.data
        hdr = hdu.header
        if data is None or getattr(data, "ndim", 0) < 2:
            raise ValueError(f"No 2-D image data in {fits_path}")
        ny, nx = data.shape[-2], data.shape[-1]
        wcs = WCS(hdr)

    center = SkyCoord.from_pixel((nx - 1) / 2.0, (ny - 1) / 2.0, wcs)
    corners_pix = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    corners = [SkyCoord.from_pixel(x, y, wcs) for x, y in corners_pix]
    radius = max(center.separation(c) for c in corners) + margin_arcmin * u.arcmin

    if not (math.isfinite(center.ra.deg) and math.isfinite(center.dec.deg)):
        raise ValueError(f"Invalid WCS center for {fits_path}")

    return center, radius


def iter_screening_rows(screening_csv: Path) -> Iterable[dict[str, str]]:
    with screening_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        yield from reader


def select_rows(
    screening_csv: Path,
    target: str,
    fits_name: Optional[str] = None,
    obsid: Optional[str] = None,
    row_index: Optional[int] = None,
    *,
    all_xmatches: bool = False,
) -> list[dict[str, str]]:
    rows = []
    target_key = norm_name(target)
    fits_key = norm(fits_name)
    obs_key = norm(obsid)

    for row in iter_screening_rows(screening_csv):
        name = row.get("target_name") or row.get("sso_name") or row.get("Name")
        if not all_xmatches and norm_name(name) != target_key:
            continue

        row_fits = norm(row.get("FITS_FILE") or row.get("fits_name"))
        row_obsid = norm(row.get("observation_id") or row.get("obsid"))

        if fits_key and row_fits != fits_key:
            continue
        if obs_key and row_obsid != obs_key:
            continue

        rows.append(row)

    if row_index is not None:
        if row_index < 0 or row_index >= len(rows):
            raise IndexError(f"--row-index={row_index} out of range for {len(rows)} matched rows")
        rows = [rows[row_index]]

    return rows


# -----------------------------
# SkyBoT helpers
# -----------------------------
# def query_skybot_table(epoch: Time, center: SkyCoord, radius: u.Quantity, *, location: str, label: str):
#     log(
#         f"[SKYBOT] Query {label} at {epoch.isot} from {location} "
#         f"center=({center.ra.deg:.8f},{center.dec.deg:.8f}) radius={radius.to(u.arcmin).value:.2f} arcmin"
#     )
#     tab = Skybot.cone_search(center, radius, epoch, location=location)
#     log(f"[SKYBOT] rows: {len(tab)}")
#     return tab

def query_skybot_table(epoch: Time, center: SkyCoord, radius: u.Quantity, *, location: str, label: str):
    log(
        f"[SKYBOT] Query {label} at {epoch.isot} from {location} "
        f"center=({center.ra.deg:.8f},{center.dec.deg:.8f}) radius={radius.to(u.arcmin).value:.2f} arcmin"
    )

    payload = Skybot.cone_search(
        center,
        radius,
        epoch,
        location=location,
        get_query_payload=True,
    )
    log(f"[SKYBOT][PAYLOAD] {payload}")

    tab = Skybot.cone_search(
        center,
        radius,
        epoch,
        location=location,
        cache=False,
    )

    log(f"[SKYBOT] uri: {getattr(Skybot, 'uri', None)}")
    log(f"[SKYBOT] rows: {len(tab)}")
    return tab

def query_object_at(epoch: Time, center: SkyCoord, radius: u.Quantity, target: str, *, location: str, strict: bool = False):
    tab = query_skybot_table(epoch, center, radius, location=location, label=target)

    target_key = norm_name(target)
    candidates = []
    for row in tab:
        name = skybot_name(row)
        key = norm_name(name)
        if key == target_key or target_key in key:
            candidates.append(row)

    if candidates:
        if len(candidates) > 1:
            log(f"[WARN] Multiple candidates matching {target}; using first")
        return candidates[0]

    log(f"[WARN] {target} not found. Returned objects nearest to cone center:")
    tmp = []
    for row in tab:
        ra, dec = get_ra_dec(row)
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        c = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        tmp.append((c.separation(center).arcsec, skybot_name(row), ra, dec))

    for sep, name, ra, dec in sorted(tmp)[:30]:
        log(f"  {name:24s} sep={sep:9.2f}\"  ra={ra:.8f} dec={dec:.8f}")

    if strict:
        raise RuntimeError(f"{target} not found in SkyBoT result at {epoch.isot}")
    return None


def motion_params(ra1: float, dec1: float, ra2: float, dec2: float) -> tuple[float, float, float, float]:
    mean_dec = 0.5 * (dec1 + dec2)
    dra = (ra2 - ra1) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec = (dec2 - dec1) * 3600.0
    length = math.hypot(dra, ddec)

    # PA east of north
    pa = math.degrees(math.atan2(dra, ddec))
    if pa < 0:
        pa += 360.0

    return dra, ddec, length, pa


def rows_by_skybot_name(tab) -> dict[str, object]:
    """Keep the first row per normalized SkyBoT name."""
    out = {}
    for row in tab:
        key = norm_name(skybot_name(row))
        if key and key not in out:
            out[key] = row
    return out


def build_output_row(
    *,
    base_row: dict[str, str],
    sso_name: str,
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    v_mag_1: str,
    decision: str,
    center: SkyCoord,
    fits_name: str,
    t1: Time,
    t2: Time,
    exposure: float,
    radius: u.Quantity,
) -> dict[str, str]:
    dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")

    return {
        "observation_id": norm(base_row.get("observation_id") or base_row.get("obsid")),
        "sso_name": sso_name,
        "ra_deg_1": f"{ra1:.9f}",
        "dec_deg_1": f"{dec1:.9f}",
        "ra_deg_2": f"{ra2:.9f}",
        "dec_deg_2": f"{dec2:.9f}",
        "filter": norm(base_row.get("filter")),
        "xmatch_type": norm(base_row.get("xmatch_type")) or "",
        "v_mag_1": v_mag_1,
        "v_mag_1_corrected": "",
        "mlim_obs": norm(base_row.get("mlim_obs")),
        "FITS_FILE": fits_name,
        "DECISION": decision,
        "dra_arcsec": f"{dra:.3f}",
        "ddec_arcsec": f"{ddec:.3f}",
        "trail_len_arcsec": f"{length:.3f}",
        "pa_deg_E_of_N": f"{pa:.3f}",
        "frame_t1": t1.isot,
        "frame_t2": t2.isot,
        "exposure_s": f"{exposure:.3f}",
        "skybot_radius_arcmin": f"{radius.to(u.arcmin).value:.3f}",
        "sep_start_arcsec_from_frame_center": f"{c1.separation(center).arcsec:.3f}",
        "sep_end_arcsec_from_frame_center": f"{c2.separation(center).arcsec:.3f}",
    }


def write_rows(rows: list[dict[str, str]], output: Optional[Path]) -> None:
    if output is None:
        f: TextIO = sys.stdout
        close = False
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        f = output.open("w", newline="", encoding="utf-8")
        close = True

    try:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close:
            f.close()
            log(f"[OK] Wrote {len(rows)} row(s) to {output}")


# -----------------------------
# per-row processing
# -----------------------------
def process_target_row(args, row: dict[str, str]) -> list[dict[str, str]]:
    obsid = norm(row.get("observation_id") or row.get("obsid"))
    fits_name = norm(row.get("FITS_FILE") or row.get("fits_name"))
    target = norm(row.get("target_name") or row.get("sso_name") or args.target) or args.target

    if not fits_name:
        log(f"[WARN] Row without FITS_FILE/fits_name for obs={obsid}; skipping")
        return []

    fits_path = find_fits_path(args.temp, fits_name)
    t1, t2, exposure = get_times_from_fits(fits_path)

    center, radius = get_image_cone_from_fits(fits_path, margin_arcmin=args.margin_arcmin)
    if args.radius_arcmin is not None:
        radius = args.radius_arcmin * u.arcmin

    r1 = query_object_at(t1, center, radius, target, location=args.location, strict=args.strict)
    r2 = query_object_at(t2, center, radius, target, location=args.location, strict=args.strict)

    if r1 is None or r2 is None:
        log(f"[FAIL] {target} not found at frame start/end for FITS={fits_name}; row skipped")
        return []

    ra1, dec1 = get_ra_dec(r1)
    ra2, dec2 = get_ra_dec(r2)

    if not all(math.isfinite(v) for v in (ra1, dec1, ra2, dec2)):
        log(f"[FAIL] Non-finite SkyBoT coordinates for FITS={fits_name}; row skipped")
        return []

    decision = norm(row.get("DECISION") or row.get("decision"))
    v_mag = skybot_vmag(r1) or norm(row.get("v_mag_1"))

    return [
        build_output_row(
            base_row=row,
            sso_name=target,
            ra1=ra1,
            dec1=dec1,
            ra2=ra2,
            dec2=dec2,
            v_mag_1=v_mag,
            decision=decision,
            center=center,
            fits_name=fits_name,
            t1=t1,
            t2=t2,
            exposure=exposure,
            radius=radius,
        )
    ]


def process_all_xmatches_row(args, row: dict[str, str]) -> list[dict[str, str]]:
    obsid = norm(row.get("observation_id") or row.get("obsid"))
    fits_name = norm(row.get("FITS_FILE") or row.get("fits_name"))
    if not fits_name:
        log(f"[WARN] Row without FITS_FILE/fits_name for obs={obsid}; skipping")
        return []

    fits_path = find_fits_path(args.temp, fits_name)
    t1, t2, exposure = get_times_from_fits(fits_path)

    center, radius = get_image_cone_from_fits(fits_path, margin_arcmin=args.margin_arcmin)
    if args.radius_arcmin is not None:
        radius = args.radius_arcmin * u.arcmin

    tab1 = query_skybot_table(t1, center, radius, location=args.location, label=f"ALL START {fits_name}")
    tab2 = query_skybot_table(t2, center, radius, location=args.location, label=f"ALL END {fits_name}")

    start = rows_by_skybot_name(tab1)
    end = rows_by_skybot_name(tab2)
    common = sorted(set(start) & set(end))
    log(f"[ALL] {fits_name}: start={len(start)} end={len(end)} common={len(common)}")

    out = []
    for key in common:
        r1 = start[key]
        r2 = end[key]
        ra1, dec1 = get_ra_dec(r1)
        ra2, dec2 = get_ra_dec(r2)
        if not all(math.isfinite(v) for v in (ra1, dec1, ra2, dec2)):
            continue

        # For all-xmatches we mark as D by default so it can go straight to rapid screening.
        out.append(
            build_output_row(
                base_row=row,
                sso_name=skybot_name(r1),
                ra1=ra1,
                dec1=dec1,
                ra2=ra2,
                dec2=dec2,
                v_mag_1=skybot_vmag(r1),
                decision="D",
                center=center,
                fits_name=fits_name,
                t1=t1,
                t2=t2,
                exposure=exposure,
                radius=radius,
            )
        )

    # More useful for manual inspection: nearest to frame center first.
    out.sort(key=lambda r: min(to_float(r["sep_start_arcsec_from_frame_center"]), to_float(r["sep_end_arcsec_from_frame_center"])))
    return out


# -----------------------------
# main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute SkyBoT start/end coordinates for screened OM FITS frames."
    )
    p.add_argument("target", nargs="?", help="Target asteroid name, e.g. Dowling. Optional with --all-xmatches if --fits-name/--obsid is enough.")
    p.add_argument("--fits-name", help="Exact FITS filename from screening.csv, e.g. P...SIMAGL000.FTZ")
    p.add_argument("--obsid", help="Observation ID filter, useful if target has multiple rows")
    p.add_argument("--row-index", type=int, help="0-based index among matched screening rows")
    p.add_argument("--screening", type=Path, default=DEFAULT_SCREENING_CSV)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--location", default=LOCATION)
    p.add_argument("--margin-arcmin", type=float, default=5.0, help="Extra radius beyond FITS WCS half-diagonal")
    p.add_argument("--radius-arcmin", type=float, default=None, help="Override WCS-derived radius")
    p.add_argument("--strict", action="store_true", help="Raise if target is not found instead of skipping row")
    p.add_argument("--all-xmatches", action="store_true", help="Return all SkyBoT start/end pairs in this FITS cone, not only the requested target")
    p.add_argument("--output", type=Path, help="Write CSV output here instead of stdout")
    args = p.parse_args(argv)

    if not args.all_xmatches and not args.target:
        p.error("target is required unless --all-xmatches is used")

    # In all-xmatches mode we still allow target filtering if target is passed,
    # but we can also select rows by --fits-name/--obsid alone.
    if args.all_xmatches and not args.target:
        args.target = ""

    rows = select_rows(
        args.screening,
        args.target or "",
        fits_name=args.fits_name,
        obsid=args.obsid,
        row_index=args.row_index,
        all_xmatches=args.all_xmatches and not args.target,
    )

    if not rows:
        raise RuntimeError(
            f"No rows found for target={args.target!r} fits={args.fits_name!r} obsid={args.obsid!r} in {args.screening}"
        )

    mode = "all-xmatches" if args.all_xmatches else "target"
    log(f"[INFO] Mode={mode}; found {len(rows)} screening row(s)")

    out_rows: list[dict[str, str]] = []
    for row in rows:
        if args.all_xmatches:
            out_rows.extend(process_all_xmatches_row(args, row))
        else:
            out_rows.extend(process_target_row(args, row))

    write_rows(out_rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
