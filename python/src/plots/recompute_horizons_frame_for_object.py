#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, TextIO

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u

from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons
try:
    from astroquery.jplhorizons import conf as horizons_conf
except Exception:  # pragma: no cover
    horizons_conf = None


DEFAULT_BASE = Path("/Users/eracero/workspace/tfm-viu")
DEFAULT_RESULTS = DEFAULT_BASE / "results/batch_L/elena"
DEFAULT_TEMP = DEFAULT_BASE / "temp"
DEFAULT_SCREENING_CSV = DEFAULT_RESULTS / "screening.csv"

# SkyBoT is used only as a candidate lister in --all-xmatches mode.
DEFAULT_SKYBOT_LOCATION = "500"

# Horizons recognizes XMM with @XMM and @XMM-Newton.
DEFAULT_HORIZONS_LOCATION = "@XMM"

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
    "horizons_location",
    "horizons_id_used",
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
    """Normalize object names from CSV/SkyBoT/Horizons."""
    s = norm(x)
    s = re.sub(r"^\(\s*\d+\s*\)\s*", "", s)
    return s.lower().replace(" ", "_")


def strip_number_prefix(name: str) -> str:
    """'(234) Barbara' -> 'Barbara'; otherwise unchanged."""
    return re.sub(r"^\(\s*\d+\s*\)\s*", "", norm(name)).strip()


def leading_number(name: str) -> str:
    """Return asteroid number if present, e.g. '(234) Barbara' -> '234'."""
    m = re.match(r"^\(\s*(\d+)\s*\)", norm(name))
    return m.group(1) if m else ""


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


def skybot_name(row) -> str:
    for key in ("Name", "name", "Object", "object"):
        try:
            return norm(row[key])
        except Exception:
            pass
    return "UNKNOWN"


def skybot_vmag(row) -> str:
    colnames = set(getattr(row, "colnames", []))
    for key in ("V", "Mv", "VMag", "Vmag", "magV", "mag", "mV"):
        if key in colnames:
            v = as_float(row[key])
            if math.isfinite(v):
                return f"{v:.3f}"
    return ""


def get_skybot_ra_dec(row) -> tuple[float, float]:
    colnames = set(getattr(row, "colnames", []))
    ra = as_float(row["RA"]) if "RA" in colnames else math.nan
    dec = as_float(row["DEC"]) if "DEC" in colnames else math.nan
    if (not math.isfinite(ra) or not math.isfinite(dec)) and {"_raj2000", "_decj2000"} <= colnames:
        ra = as_float(row["_raj2000"])
        dec = as_float(row["_decj2000"])
    return ra, dec


def horizons_id_candidates(name: str) -> list[str]:
    """Try robust Horizons IDs for names returned by SkyBoT/screening.csv."""
    raw = norm(name)
    out: list[str] = []

    n = leading_number(raw)
    if n:
        out.append(n)

    stripped = strip_number_prefix(raw)
    for candidate in (raw, stripped):
        candidate = norm(candidate)
        if candidate and candidate not in out:
            out.append(candidate)

    # Some scripts/users pass underscores for object names.
    if "_" in stripped:
        c = stripped.replace("_", " ")
        if c not in out:
            out.append(c)

    return out


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
    """Return frame start/end time using FITS DATE-OBS + exposure."""
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
    """Cone-search center/radius from the actual FITS WCS footprint."""
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
# SkyBoT candidate helpers
# -----------------------------
def query_skybot_table(epoch: Time, center: SkyCoord, radius: u.Quantity, *, location: str, label: str):
    log(
        f"[SKYBOT] Query {label} at {epoch.isot} from {location} "
        f"center=({center.ra.deg:.8f},{center.dec.deg:.8f}) radius={radius.to(u.arcmin).value:.2f} arcmin"
    )
    tab = Skybot.cone_search(center, radius, epoch, location=location, cache=False)
    log(f"[SKYBOT] rows: {len(tab)}")
    return tab


def rows_by_skybot_name(tab) -> dict[str, object]:
    out = {}
    for row in tab:
        key = norm_name(skybot_name(row))
        if key and key not in out:
            out[key] = row
    return out


# -----------------------------
# Horizons position helpers
# -----------------------------
def query_horizons_position(
    target_name: str,
    epoch: Time,
    *,
    location: str,
    timeout: float,
    retries: int,
    sleep_s: float,
) -> tuple[float, float, str, str]:
    """Return RA, DEC, Vmag string, and Horizons ID used.

    Uses Horizons smallbody search. Tries asteroid number first if present, then name.
    """
    if horizons_conf is not None:
        try:
            horizons_conf.timeout = timeout
        except Exception:
            pass

    last_exc: Optional[BaseException] = None
    for hid in horizons_id_candidates(target_name):
        for attempt in range(retries + 1):
            try:
                log(f"[HORIZONS] Query target={target_name!r} id={hid!r} epoch={epoch.isot} location={location} attempt={attempt + 1}/{retries + 1}")
                obj = Horizons(id=hid, id_type="smallbody", location=location, epochs=epoch.jd)
                eph = obj.ephemerides()
                if len(eph) == 0:
                    raise RuntimeError("Horizons returned an empty ephemerides table")

                row = eph[0]
                ra = as_float(row["RA"])
                dec = as_float(row["DEC"])
                if not (math.isfinite(ra) and math.isfinite(dec)):
                    raise RuntimeError(f"Non-finite Horizons RA/DEC for id={hid!r}")

                vmag = ""
                if "V" in eph.colnames:
                    v = as_float(row["V"])
                    if math.isfinite(v):
                        vmag = f"{v:.3f}"

                return ra, dec, vmag, hid

            except Exception as e:
                last_exc = e
                log(f"[HORIZONS][WARN] target={target_name!r} id={hid!r} failed: {type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(sleep_s)

    raise RuntimeError(f"Horizons failed for target={target_name!r}; last error={last_exc!r}")


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
    horizons_location: str,
    horizons_id_used: str,
) -> dict[str, str]:
    dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")

    # Keep corrected magnitude numeric for action_screening.py.
    v_corr = norm(base_row.get("v_mag_1_corrected")) or v_mag_1 or norm(base_row.get("v_mag_1"))

    return {
        "observation_id": norm(base_row.get("observation_id") or base_row.get("obsid")),
        "sso_name": sso_name,
        "ra_deg_1": f"{ra1:.9f}",
        "dec_deg_1": f"{dec1:.9f}",
        "ra_deg_2": f"{ra2:.9f}",
        "dec_deg_2": f"{dec2:.9f}",
        "filter": norm(base_row.get("filter")),
        "xmatch_type": norm(base_row.get("xmatch_type")) or "",
        "v_mag_1": v_mag_1 or norm(base_row.get("v_mag_1")),
        "v_mag_1_corrected": v_corr,
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
        "horizons_location": horizons_location,
        "horizons_id_used": horizons_id_used,
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
def common_frame_setup(args, row: dict[str, str]):
    obsid = norm(row.get("observation_id") or row.get("obsid"))
    fits_name = norm(row.get("FITS_FILE") or row.get("fits_name"))
    if not fits_name:
        log(f"[WARN] Row without FITS_FILE/fits_name for obs={obsid}; skipping")
        return None

    fits_path = find_fits_path(args.temp, fits_name)
    t1, t2, exposure = get_times_from_fits(fits_path)

    center, radius = get_image_cone_from_fits(fits_path, margin_arcmin=args.margin_arcmin)
    if args.radius_arcmin is not None:
        radius = args.radius_arcmin * u.arcmin

    return fits_name, t1, t2, exposure, center, radius


def process_target_row(args, row: dict[str, str]) -> list[dict[str, str]]:
    setup = common_frame_setup(args, row)
    if setup is None:
        return []
    fits_name, t1, t2, exposure, center, radius = setup

    target = norm(row.get("target_name") or row.get("sso_name") or args.target) or args.target
    if not target:
        log(f"[WARN] Cannot determine target for FITS={fits_name}; skipping")
        return []

    try:
        ra1, dec1, v1, hid1 = query_horizons_position(
            target,
            t1,
            location=args.horizons_location,
            timeout=args.horizons_timeout,
            retries=args.horizons_retries,
            sleep_s=args.horizons_retry_sleep,
        )
        ra2, dec2, v2, hid2 = query_horizons_position(
            target,
            t2,
            location=args.horizons_location,
            timeout=args.horizons_timeout,
            retries=args.horizons_retries,
            sleep_s=args.horizons_retry_sleep,
        )
    except Exception as e:
        msg = f"[FAIL] Horizons failed for {target} FITS={fits_name}: {type(e).__name__}: {e}"
        if args.strict:
            raise RuntimeError(msg) from e
        log(msg)
        return []

    vmag = v1 or v2 or norm(row.get("v_mag_1"))
    decision = norm(row.get("DECISION") or row.get("decision")) or "D"
    hid = hid1 if hid1 == hid2 else f"{hid1}|{hid2}"

    return [
        build_output_row(
            base_row=row,
            sso_name=target,
            ra1=ra1,
            dec1=dec1,
            ra2=ra2,
            dec2=dec2,
            v_mag_1=vmag,
            decision=decision,
            center=center,
            fits_name=fits_name,
            t1=t1,
            t2=t2,
            exposure=exposure,
            radius=radius,
            horizons_location=args.horizons_location,
            horizons_id_used=hid,
        )
    ]


def process_all_xmatches_row(args, row: dict[str, str]) -> list[dict[str, str]]:
    setup = common_frame_setup(args, row)
    if setup is None:
        return []
    fits_name, t1, t2, exposure, center, radius = setup

    # SkyBoT is only a fast candidate lister. Coordinates below are recomputed by Horizons @XMM.
    tab1 = query_skybot_table(t1, center, radius, location=args.skybot_location, label=f"ALL START {fits_name}")
    tab2 = query_skybot_table(t2, center, radius, location=args.skybot_location, label=f"ALL END {fits_name}")

    start = rows_by_skybot_name(tab1)
    end = rows_by_skybot_name(tab2)
    common = sorted(set(start) & set(end))
    log(f"[ALL] {fits_name}: skybot_start={len(start)} skybot_end={len(end)} common={len(common)}")

    out = []
    for key in common:
        r1 = start[key]
        sky_name = skybot_name(r1)

        try:
            ra1, dec1, hv1, hid1 = query_horizons_position(
                sky_name,
                t1,
                location=args.horizons_location,
                timeout=args.horizons_timeout,
                retries=args.horizons_retries,
                sleep_s=args.horizons_retry_sleep,
            )
            ra2, dec2, hv2, hid2 = query_horizons_position(
                sky_name,
                t2,
                location=args.horizons_location,
                timeout=args.horizons_timeout,
                retries=args.horizons_retries,
                sleep_s=args.horizons_retry_sleep,
            )
        except Exception as e:
            log(f"[HORIZONS][SKIP] {sky_name!r}: {type(e).__name__}: {e}")
            continue

        vmag = hv1 or hv2 or skybot_vmag(r1)
        hid = hid1 if hid1 == hid2 else f"{hid1}|{hid2}"

        out.append(
            build_output_row(
                base_row=row,
                sso_name=sky_name,
                ra1=ra1,
                dec1=dec1,
                ra2=ra2,
                dec2=dec2,
                v_mag_1=vmag,
                decision="D",
                center=center,
                fits_name=fits_name,
                t1=t1,
                t2=t2,
                exposure=exposure,
                radius=radius,
                horizons_location=args.horizons_location,
                horizons_id_used=hid,
            )
        )

    out.sort(key=lambda r: min(to_float(r["sep_start_arcsec_from_frame_center"]), to_float(r["sep_end_arcsec_from_frame_center"])))
    return out


# -----------------------------
# main
# -----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Recompute XMM-OM frame start/end asteroid coordinates using JPL Horizons. "
            "SkyBoT is only used as a candidate lister in --all-xmatches mode."
            "Example usage: python python/src/plots/recompute_horizons_frame_for_object.py "
              " --all-xmatches   "
              "--fits-name P0673002335OMS409SIMAGE0000.FTZ "
              "--radius-arcmin 10   "
              "--horizons-location @XMM   "
              "--output /Users/eracero/workspace/tfm-viu/results/batch_L/elena/screening_dowling_all_xmatches_horizons.csv"
        )
    )
    p.add_argument("target", nargs="?", help="Target asteroid name/number, e.g. Barbara or 234. Optional with --all-xmatches if --fits-name/--obsid is enough.")
    p.add_argument("--fits-name", help="Exact FITS filename from screening.csv, e.g. P...SIMAGL000.FTZ")
    p.add_argument("--obsid", help="Observation ID filter, useful if target has multiple rows")
    p.add_argument("--row-index", type=int, help="0-based index among matched screening rows")
    p.add_argument("--screening", type=Path, default=DEFAULT_SCREENING_CSV)
    p.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    p.add_argument("--margin-arcmin", type=float, default=5.0, help="Extra radius beyond FITS WCS half-diagonal for SkyBoT candidate listing")
    p.add_argument("--radius-arcmin", type=float, default=None, help="Override WCS-derived SkyBoT candidate radius")
    p.add_argument("--all-xmatches", action="store_true", help="Use SkyBoT to list all start/end candidate pairs, then recompute each with Horizons")
    p.add_argument("--skybot-location", default=DEFAULT_SKYBOT_LOCATION, help="SkyBoT candidate-list observer; default 500 because SkyBoT XMM appears effectively geocentric")
    p.add_argument("--horizons-location", default=DEFAULT_HORIZONS_LOCATION, help="Horizons observer location; use @XMM or @XMM-Newton")
    p.add_argument("--horizons-timeout", type=float, default=120.0, help="Astroquery Horizons timeout in seconds")
    p.add_argument("--horizons-retries", type=int, default=1, help="Retries per Horizons ID/epoch after a failure")
    p.add_argument("--horizons-retry-sleep", type=float, default=2.0, help="Seconds between Horizons retries")
    p.add_argument("--strict", action="store_true", help="Raise if a target cannot be resolved instead of skipping")
    p.add_argument("--output", type=Path, help="Write CSV output here instead of stdout")
    args = p.parse_args(argv)

    if not args.all_xmatches and not args.target:
        p.error("target is required unless --all-xmatches is used")

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

    mode = "all-xmatches+horizons" if args.all_xmatches else "target+horizons"
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
