#!/usr/bin/env python3
"""
Joint deblending testbed for (2249) Weisell / OBSID 0691070301 / XMM-OM UVW1.

The geometry is more difficult than Transvaalia: the Horizons trail crosses
almost directly through one catalogued stationary source (A1), while a second
source (A2) lies almost exactly on the continuation of the trail just before
Horizons START.  The fit therefore models *both* stationary sources and the
finite asteroid trail simultaneously:

    image = asteroid_finite_trail + star_A1 + star_A2 + local_background

Primary geometry
----------------
The asteroid trail is fixed to the frame-specific JPL Horizons START/END
coordinates supplied for this observation.  The finite trail template is a
normalized 2-D Gaussian PSF convolved with a uniform finite line segment from
HSTART to HEND.  Its fitted coefficient is therefore the total asteroid counts
in the finite trail before aperture truncation.

Primary PSF
-----------
Because A1 lies on top of the asteroid trail, its own SRCLIST FWHM/PA may be
biased by blending.  The PRIMARY fit therefore estimates one robust field PSF
from good, isolated SRCLIST sources in the same frame and uses that PSF for the
trail and both contaminating stars.  A source-specific-PSF fit is also run as a
sensitivity diagnostic.

Fits performed
--------------
C0  PRIMARY: field PSF, simultaneous finite trail + A1 + A2 + background.
    Unrelated catalogue sources in the cutout are masked automatically so that
    they do not bias the local background plane.
C1  TRAIL-ONLY OUTSIDE STARS: no stellar amplitudes are fitted at all.  The
    cores (and most PSF wings) of A1/A2 are masked, together with unrelated
    field sources, and only finite trail + local background are fitted.  This
    is the key validation that asks whether Weisell is recoverable from the
    unobscured parts of the Horizons trail alone.
C2  SOURCE-PSF SENSITIVITY: same simultaneous joint fit as C0, but A1/A2 use
    their own SRCLIST FWHM/PA values while the trail retains the field PSF.

Calibration-domain rule
-----------------------
The useful pipeline-domain quantity is the modeled asteroid rate *inside a
canonical h=13 rectangular aperture spanning Horizons START to END*:

    R_h,Horizons = model asteroid counts inside that rectangle / EXPOSURE

For h=13, C1=1, so this can be fed into the normal pipeline from R_6 onward:

    R_6 (= R_h) -> CoI -> C2 -> TDS -> m_AB

Do not subtract catalogued SRCLIST RATE/CORR_RATE from a calibrated final rate.
SRCLIST rates are printed only as diagnostics and are never used as priors.

The two historical/manual h=13 extractions supplied in the project are retained
only as closure checks.  The script automatically identifies which one has an
effective longitudinal width A_eff/h closest to the WCS Horizons trail length.

Outputs
-------
  weisell_joint_deblend_result.json
  weisell_joint_deblend_diagnostic.npz
  weisell_joint_deblend_diagnostic.png
  weisell_joint_deblend_validation.png
  weisell_deblended_final_row.csv

The final CSV contains the adopted C1 trail-only solution passed through the
closed pipeline calibration chain.  It is written as a standalone one-row file
and is never appended automatically to the main science catalogue.

Run from anywhere.  The SRCLIST is auto-discovered in the image directory.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


# =============================================================================
# CASE CONFIGURATION
# =============================================================================

BASE_DIR = Path("/Users/eracero/workspace/tfm-viu/temp/0691070301/L")
IMAGE_FITS = BASE_DIR / "P0691070301OMS401SIMAGE1000.FTZ"

# Frame-specific JPL Horizons positions supplied by the validation CSV.
HSTART_RA_DEG = 272.574810000
HSTART_DEC_DEG = -19.681580000
HEND_RA_DEG = 272.578280000
HEND_DEC_DEG = -19.683470000
HORIZONS_TRAIL_LEN_ARCSEC = 13.588
HORIZONS_PA_E_OF_N_DEG = 120.048
HORIZONS_UNC_3SIGMA_ARCSEC = 0.003
EXPTIME_EXPECTED = 2840.0

# Stationary sources selected in the UI.  Do NOT use UI idx as FITS row index;
# these RA/Dec values are used to recover the actual SRCLIST rows robustly.
CONTAMINANTS = [
    {
        "name": "A1",
        "ui_idx": 64,
        "ra_deg": 272.575375,
        "dec_deg": -19.681739,
        "ui_x": 116.50,
        "ui_y": 684.76,
    },
    {
        "name": "A2",
        "ui_idx": 59,
        "ra_deg": 272.573937,
        "dec_deg": -19.681048,
        "ui_x": 121.61,
        "ui_y": 687.37,
    },
]

# Two h=13 manual extractions supplied by the user.  We do not assume which is
# the scientifically preferred one; the code picks the one whose A_eff/h is
# closest to the Horizons trail length in image pixels for the closure test.
MANUAL_EXTRACTIONS = [
    {
        "label": "manual_h13_A",
        "rh": 0.0708666250580224,
        "rh_err": 0.018988789615067954,
        "net_counts": 201.2612151647836,
        "net_counts_err": 53.92816250679299,
        "aper_counts": 2585.578510109335,
        "bkg_per_pix": 18.09098252078943,
        "bkg_rms_per_pix": 2.798713724689457,
        "A_ap_eff": 131.7958984375,
        "A_bg_eff": 421.662109375,
        "height_pix": 13.0,
    },
    {
        "label": "manual_h13_B",
        "rh": 0.29877860469565976,
        "rh_err": 0.024210130532501265,
        "net_counts": 848.5312373356737,
        "net_counts_err": 68.75677071230359,
        "aper_counts": 4173.384249720722,
        "bkg_per_pix": 17.76261632806725,
        "bkg_rms_per_pix": 2.7344016945177576,
        "A_ap_eff": 187.1826171875,
        "A_bg_eff": 472.78125,
        "height_pix": 13.0,
    },
]

AP_HEIGHT_PIX = 13.0
CUTOUT_MARGIN_PIX = 12
SUBPIX = 4
TRAIL_SAMPLE_STEP_PIX = 0.20
ROBUST_CLIP_SIGMA = 5.0
ROBUST_MAXITER = 5
# Radius used by the independent trail-only validation.  With a field PSF
# FWHM ~2.3 pix, 3.5 pix masks >3 sigma of the stellar cores/wings.
TRAIL_ONLY_STAR_MASK_RADIUS_PIX = 3.5

# Catalogue sources unrelated to the Horizons trail are masked from all fits.
# This removes e.g. the bright residual source near (x,y)~(120,671) without
# adding another free PSF amplitude to the model.
UNRELATED_SOURCE_MASK_RADIUS_PIX = 3.0
UNRELATED_SOURCE_MIN_SIGNIFICANCE = 5.0
UNRELATED_SOURCE_MIN_RATE = 0.02
UNRELATED_SOURCE_MIN_TRAIL_DIST_PIX = 4.0

# Field-PSF source selection.
FIELD_PSF_MIN_SIGNIFICANCE = 10.0
FIELD_PSF_MIN_STARS = 5
FIELD_PSF_EXCLUDE_TRAIL_DIST_PIX = 7.0
FIELD_PSF_FWHM_MIN_PIX = 1.0
FIELD_PSF_FWHM_MAX_PIX = 6.0
FIELD_PSF_MAX_AXIS_RATIO = 2.5

OUT_JSON = Path("weisell_joint_deblend_result.json")
OUT_NPZ = Path("weisell_joint_deblend_diagnostic.npz")
OUT_PNG = Path("weisell_joint_deblend_diagnostic.png")
OUT_VALIDATION_PNG = Path("weisell_joint_deblend_validation.png")
OUT_FINAL_ROW_CSV = Path("weisell_deblended_final_row.csv")

# Closed UVW1 calibration constants from the final pipeline calibration.
ABM0_UVW1 = 18.5662
C2_UVW1 = 1.0744875201
C2_UVW1_ERR = 0.0349126848
VPRED = 16.45
VPRED_UVW1_AB = 17.9342
MLIM_OBS = 19.8031

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# =============================================================================
# FITS / SRCLIST helpers
# =============================================================================


def find_image_hdu(hdul: fits.HDUList):
    for i, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and np.ndim(data) == 2:
            return i, hdu
    raise RuntimeError("No 2-D image HDU found")


def find_table_hdu(hdul: fits.HDUList):
    for i, hdu in enumerate(hdul):
        if getattr(hdu, "columns", None) is not None and getattr(hdu, "data", None) is not None:
            return i, hdu
    raise RuntimeError("No binary-table HDU found")


def autodiscover_srclist(directory: Path) -> Path:
    patterns = [
        "P0691070301OMS401*SWSRLI*.FTZ",
        "*OMS401*SWSRLI*.FTZ",
        "*SWSRLI*.FTZ",
    ]
    for pat in patterns:
        candidates = sorted(directory.glob(pat))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Prefer a product with the exact observation/frame prefix.
            exact = [p for p in candidates if "P0691070301OMS401" in p.name]
            if len(exact) == 1:
                return exact[0]
    raise FileNotFoundError(
        f"Could not uniquely auto-discover SRCLIST in {directory}. "
        "Set SRCLIST_FITS manually near the top of the script."
    )


def row_to_dict(tab, names, j: int, ihdu: int) -> dict:
    row = {}
    for name in names:
        v = tab[name][j]
        row[name] = v.item() if hasattr(v, "item") else v
    row["_HDU"] = ihdu
    row["_ROW0"] = int(j)
    return row


def nearest_srclist_row(srclist: Path, ra0: float, dec0: float) -> tuple[dict, float]:
    with fits.open(srclist, memmap=False) as hdul:
        ihdu, hdu = find_table_hdu(hdul)
        tab = hdu.data
        names = list(tab.columns.names)
        up = {n.upper(): n for n in names}
        ra_col = up.get("RA_CORR") or up.get("RA")
        dec_col = up.get("DEC_CORR") or up.get("DEC")
        if ra_col is None or dec_col is None:
            raise KeyError("SRCLIST has neither RA_CORR/DEC_CORR nor RA/DEC")

        ra = np.asarray(tab[ra_col], dtype=float)
        dec = np.asarray(tab[dec_col], dtype=float)
        dra = (ra - float(ra0)) * np.cos(np.deg2rad(float(dec0)))
        ddec = dec - float(dec0)
        d2 = dra * dra + ddec * ddec
        j = int(np.nanargmin(d2))
        sep_arcsec = 3600.0 * math.sqrt(float(d2[j]))
        return row_to_dict(tab, names, j, ihdu), sep_arcsec


def sky_to_xy(wcs: WCS, ra_deg: float, dec_deg: float) -> np.ndarray:
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return np.array([float(x), float(y)], dtype=float)


def all_srclist_rows(srclist: Path) -> tuple[np.ndarray, list[str], int]:
    with fits.open(srclist, memmap=False) as hdul:
        ihdu, hdu = find_table_hdu(hdul)
        return hdu.data.copy(), list(hdu.columns.names), ihdu


def find_unrelated_srclist_sources(srclist: Path, wcs: WCS, bounds: tuple[int, int, int, int],
                                   hstart_xy: np.ndarray, hend_xy: np.ndarray,
                                   contaminant_rows: list[dict]) -> list[dict]:
    """Return catalogue sources in the fit cutout that are unrelated to Weisell.

    These objects are *masked*, not fitted.  A source qualifies when it is in
    the cutout, is sufficiently far from the finite Horizons segment, and has
    either a catalogue significance or RATE large enough to matter.  A1/A2 are
    explicitly excluded because they are modelled in the joint fits.
    """
    tab, names, ihdu = all_srclist_rows(srclist)
    up = {n.upper(): n for n in names}
    ra_col = up.get("RA_CORR") or up.get("RA")
    dec_col = up.get("DEC_CORR") or up.get("DEC")
    if ra_col is None or dec_col is None:
        return []

    ra = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(sky)
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])

    x1, x2, y1, y2 = bounds
    mask = np.isfinite(xy).all(axis=1)
    mask &= (xy[:, 0] >= x1) & (xy[:, 0] < x2)
    mask &= (xy[:, 1] >= y1) & (xy[:, 1] < y2)

    dseg = np.array([
        distance_point_to_segment(p, hstart_xy, hend_xy) if np.isfinite(p).all() else np.inf
        for p in xy
    ])
    mask &= dseg >= UNRELATED_SOURCE_MIN_TRAIL_DIST_PIX

    strength = np.zeros(len(tab), dtype=bool)
    if "SIGNIFICANCE" in up:
        sig = np.asarray(tab[up["SIGNIFICANCE"]], dtype=float)
        strength |= np.isfinite(sig) & (sig >= UNRELATED_SOURCE_MIN_SIGNIFICANCE)
    if "RATE" in up:
        rate = np.asarray(tab[up["RATE"]], dtype=float)
        strength |= np.isfinite(rate) & (rate >= UNRELATED_SOURCE_MIN_RATE)
    mask &= strength

    for row in contaminant_rows:
        j = int(row["_ROW0"])
        if 0 <= j < len(mask):
            mask[j] = False

    out = []
    for j in np.flatnonzero(mask):
        row = row_to_dict(tab, names, int(j), ihdu)
        out.append({
            "row0": int(j),
            "srcnum": int(row.get("SRCNUM", -1)),
            "src_id": int(row.get("SRC_ID", -1)),
            "x": float(xy[j, 0]),
            "y": float(xy[j, 1]),
            "distance_to_trail_pix": float(dseg[j]),
            "rate": float(row.get("RATE", np.nan)),
            "significance": float(row.get("SIGNIFICANCE", np.nan)),
        })

    # Brightest first for readable console output.
    out.sort(key=lambda q: (np.nan_to_num(q["rate"], nan=-np.inf),
                            np.nan_to_num(q["significance"], nan=-np.inf)), reverse=True)
    return out


def circular_exclusion_mask(xx: np.ndarray, yy: np.ndarray, centres: list[np.ndarray],
                            radius_pix: float, base_mask: np.ndarray | None = None) -> np.ndarray:
    mask = np.ones_like(xx, dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=bool).copy()
    for c in centres:
        c = np.asarray(c, dtype=float)
        mask &= np.hypot(xx - c[0], yy - c[1]) >= float(radius_pix)
    return mask


# =============================================================================
# Geometry helpers
# =============================================================================


def point_segment_geometry(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float, float]:
    """Return (t, along_pix, signed_perp_pix) relative to finite segment p0->p1."""
    v = np.asarray(p1, float) - np.asarray(p0, float)
    p = np.asarray(point, float) - np.asarray(p0, float)
    vv = float(v @ v)
    if vv <= 0:
        raise ValueError("Degenerate segment")
    L = math.sqrt(vv)
    t = float((p @ v) / vv)
    along = t * L
    perp = float((v[0] * p[1] - v[1] * p[0]) / L)
    return t, along, perp


def sky_point_segment_geometry_arcsec(ra: float, dec: float) -> tuple[float, float, float]:
    dec_ref = 0.5 * (HSTART_DEC_DEG + HEND_DEC_DEG)
    c = math.cos(math.radians(dec_ref))
    v = np.array([
        (HEND_RA_DEG - HSTART_RA_DEG) * c * 3600.0,
        (HEND_DEC_DEG - HSTART_DEC_DEG) * 3600.0,
    ])
    p = np.array([
        (float(ra) - HSTART_RA_DEG) * c * 3600.0,
        (float(dec) - HSTART_DEC_DEG) * 3600.0,
    ])
    vv = float(v @ v)
    L = math.sqrt(vv)
    t = float((p @ v) / vv)
    along = t * L
    perp = float((v[0] * p[1] - v[1] * p[0]) / L)
    return t, along, perp


def distance_point_to_segment(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    t, _, _ = point_segment_geometry(point, p0, p1)
    tclip = min(1.0, max(0.0, t))
    q = p0 + tclip * (p1 - p0)
    return float(np.linalg.norm(point - q))


# =============================================================================
# PSF helpers
# =============================================================================


def psf_covariance(fwhm_maj: float, fwhm_min: float, pa_deg: float) -> np.ndarray:
    smaj = float(fwhm_maj) * FWHM_TO_SIGMA
    smin = float(fwhm_min) * FWHM_TO_SIGMA
    pa = math.radians(float(pa_deg))
    umaj = np.array([math.cos(pa), math.sin(pa)], dtype=float)
    umin = np.array([-math.sin(pa), math.cos(pa)], dtype=float)
    return (smaj ** 2) * np.outer(umaj, umaj) + (smin ** 2) * np.outer(umin, umin)


def covariance_to_fwhm_pa(cov: np.ndarray) -> tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals = np.clip(vals, 1e-8, None)
    smaj, smin = np.sqrt(vals[0]), np.sqrt(vals[1])
    vmaj = vecs[:, 0]
    pa = math.degrees(math.atan2(vmaj[1], vmaj[0]))
    return smaj / FWHM_TO_SIGMA, smin / FWHM_TO_SIGMA, pa


def robust_field_psf(srclist: Path, wcs: WCS, hstart_xy: np.ndarray, hend_xy: np.ndarray,
                     contaminant_rows: list[dict]) -> dict:
    tab, names, ihdu = all_srclist_rows(srclist)
    up = {n.upper(): n for n in names}
    required = ["FWHM_MAJ", "FWHM_MIN", "PA"]
    if any(k not in up for k in required):
        raise KeyError("SRCLIST lacks FWHM_MAJ/FWHM_MIN/PA required for field PSF")

    ra_col = up.get("RA_CORR") or up.get("RA")
    dec_col = up.get("DEC_CORR") or up.get("DEC")
    ra = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)
    sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(sky)
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])

    fmaj = np.asarray(tab[up["FWHM_MAJ"]], dtype=float)
    fmin = np.asarray(tab[up["FWHM_MIN"]], dtype=float)
    pa = np.asarray(tab[up["PA"]], dtype=float)

    mask = np.isfinite(xy).all(axis=1) & np.isfinite(fmaj) & np.isfinite(fmin) & np.isfinite(pa)
    mask &= (fmaj >= FIELD_PSF_FWHM_MIN_PIX) & (fmaj <= FIELD_PSF_FWHM_MAX_PIX)
    mask &= (fmin >= FIELD_PSF_FWHM_MIN_PIX) & (fmin <= FIELD_PSF_FWHM_MAX_PIX)
    mask &= (fmaj / np.maximum(fmin, 1e-6) <= FIELD_PSF_MAX_AXIS_RATIO)

    if "SIGNIFICANCE" in up:
        sig = np.asarray(tab[up["SIGNIFICANCE"]], dtype=float)
        mask &= np.isfinite(sig) & (sig >= FIELD_PSF_MIN_SIGNIFICANCE)
    if "RATE" in up:
        rate = np.asarray(tab[up["RATE"]], dtype=float)
        mask &= np.isfinite(rate) & (rate > 0)
    for flag in ("QFLAG", "CFLAG", "EFLAG"):
        if flag in up:
            mask &= (np.asarray(tab[up[flag]]) == 0)

    # Exclude sources close to the Weisell trail, including A1/A2 and any other
    # object that may overlap the finite-trail region.
    dseg = np.array([
        distance_point_to_segment(p, hstart_xy, hend_xy) if np.isfinite(p).all() else np.inf
        for p in xy
    ])
    mask &= dseg > FIELD_PSF_EXCLUDE_TRAIL_DIST_PIX

    # Explicitly exclude recovered contaminant raw rows as an extra safeguard.
    for row in contaminant_rows:
        j = int(row["_ROW0"])
        if 0 <= j < len(mask):
            mask[j] = False

    idx = np.flatnonzero(mask)
    if len(idx) < FIELD_PSF_MIN_STARS:
        # Relax significance only; retain geometry/flags/isolation requirements.
        mask2 = np.isfinite(xy).all(axis=1) & np.isfinite(fmaj) & np.isfinite(fmin) & np.isfinite(pa)
        mask2 &= (fmaj >= FIELD_PSF_FWHM_MIN_PIX) & (fmaj <= FIELD_PSF_FWHM_MAX_PIX)
        mask2 &= (fmin >= FIELD_PSF_FWHM_MIN_PIX) & (fmin <= FIELD_PSF_FWHM_MAX_PIX)
        mask2 &= (fmaj / np.maximum(fmin, 1e-6) <= FIELD_PSF_MAX_AXIS_RATIO)
        if "SIGNIFICANCE" in up:
            sig = np.asarray(tab[up["SIGNIFICANCE"]], dtype=float)
            mask2 &= np.isfinite(sig) & (sig >= 5.0)
        if "RATE" in up:
            rate = np.asarray(tab[up["RATE"]], dtype=float)
            mask2 &= np.isfinite(rate) & (rate > 0)
        for flag in ("QFLAG", "CFLAG", "EFLAG"):
            if flag in up:
                mask2 &= (np.asarray(tab[up[flag]]) == 0)
        mask2 &= dseg > FIELD_PSF_EXCLUDE_TRAIL_DIST_PIX
        for row in contaminant_rows:
            j = int(row["_ROW0"])
            if 0 <= j < len(mask2):
                mask2[j] = False
        idx = np.flatnonzero(mask2)

    if len(idx) < 3:
        raise RuntimeError(f"Too few clean SRCLIST stars for field PSF: N={len(idx)}")

    covs = np.array([psf_covariance(fmaj[j], fmin[j], pa[j]) for j in idx])
    # Robust element-wise median covariance.  Symmetrize and force positive-definite.
    cov = np.median(covs, axis=0)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.05 ** 2, None)
    cov = vecs @ np.diag(vals) @ vecs.T
    fmj, fmn, padeg = covariance_to_fwhm_pa(cov)

    srcnums = []
    if "SRCNUM" in up:
        srcnums = [int(tab[up["SRCNUM"]][j]) for j in idx]

    return {
        "cov": cov,
        "fwhm_maj": float(fmj),
        "fwhm_min": float(fmn),
        "pa_deg": float(padeg),
        "n_stars": int(len(idx)),
        "row_indices": idx.astype(int).tolist(),
        "srcnums": srcnums,
    }


def _subpixel_offsets(nsub: int) -> np.ndarray:
    return (np.arange(nsub, dtype=float) + 0.5) / nsub - 0.5


def integrated_gaussian_template_cov(xx: np.ndarray, yy: np.ndarray, centre_xy: np.ndarray,
                                     cov: np.ndarray, nsub: int = SUBPIX) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(cov, float))
    det = float(np.linalg.det(cov))
    norm = 1.0 / (2.0 * math.pi * math.sqrt(det))
    x0, y0 = map(float, centre_xy)
    out = np.zeros_like(xx, dtype=float)
    offs = _subpixel_offsets(nsub)
    for oy in offs:
        for ox in offs:
            dx = (xx + ox) - x0
            dy = (yy + oy) - y0
            q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
            out += norm * np.exp(-0.5 * q)
    out /= float(nsub * nsub)
    return out


def integrated_finite_trail_template(xx: np.ndarray, yy: np.ndarray, start_xy: np.ndarray,
                                     end_xy: np.ndarray, cov: np.ndarray,
                                     step_pix: float = TRAIL_SAMPLE_STEP_PIX) -> np.ndarray:
    """Normalized PSF-convolved finite uniform line.

    The returned template integrates to unity over an infinite image, so its
    fitted coefficient is the total counts in the entire finite trail.
    """
    start_xy = np.asarray(start_xy, float)
    end_xy = np.asarray(end_xy, float)
    length = float(np.linalg.norm(end_xy - start_xy))
    n = max(2, int(math.ceil(length / float(step_pix))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    out = np.zeros_like(xx, dtype=float)
    for t in ts:
        pos = start_xy + t * (end_xy - start_xy)
        out += integrated_gaussian_template_cov(xx, yy, pos, cov, nsub=SUBPIX)
    out /= float(n)
    return out


# =============================================================================
# Cutout / aperture maps
# =============================================================================


def make_scene_cutout(data: np.ndarray, points: list[np.ndarray], margin: int):
    pts = np.asarray(points, dtype=float)
    ny, nx = data.shape
    x1 = max(0, int(math.floor(np.min(pts[:, 0]))) - int(margin))
    x2 = min(nx, int(math.ceil(np.max(pts[:, 0]))) + int(margin) + 1)
    y1 = max(0, int(math.floor(np.min(pts[:, 1]))) - int(margin))
    y2 = min(ny, int(math.ceil(np.max(pts[:, 1]))) + int(margin) + 1)
    cut = np.asarray(data[y1:y2, x1:x2], dtype=float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    return cut, xx.astype(float), yy.astype(float), (x1, x2, y1, y2)


def rectangular_fraction_map(xx: np.ndarray, yy: np.ndarray, start_xy: np.ndarray,
                             end_xy: np.ndarray, height_pix: float,
                             nsub: int = SUBPIX) -> np.ndarray:
    """Fractional pixel area inside rectangle spanning start->end with given height."""
    p0 = np.asarray(start_xy, float)
    p1 = np.asarray(end_xy, float)
    centre = 0.5 * (p0 + p1)
    v = p1 - p0
    width = float(np.linalg.norm(v))
    uhat = v / width
    nhat = np.array([-uhat[1], uhat[0]], dtype=float)
    hw = 0.5 * width
    hh = 0.5 * float(height_pix)
    out = np.zeros_like(xx, dtype=float)
    offs = _subpixel_offsets(nsub)
    for oy in offs:
        for ox in offs:
            dx = (xx + ox) - centre[0]
            dy = (yy + oy) - centre[1]
            along = dx * uhat[0] + dy * uhat[1]
            perp = dx * nhat[0] + dy * nhat[1]
            out += ((np.abs(along) <= hw) & (np.abs(perp) <= hh)).astype(float)
    out /= float(nsub * nsub)
    return out


# =============================================================================
# Joint linear fit
# =============================================================================


def _solve_weighted_linear(cut: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                           trail_t: np.ndarray, star_templates: list[np.ndarray],
                           scene_origin_xy: np.ndarray,
                           bkg_rms_ref: float,
                           initial_mask: np.ndarray | None = None) -> dict:
    """Robust linear fit: trail + N stars + tilted background plane."""
    x0, y0 = map(float, scene_origin_xy)
    dx = xx - x0
    dy = yy - y0

    finite = np.isfinite(cut) & np.isfinite(trail_t)
    for st in star_templates:
        finite &= np.isfinite(st)
    if initial_mask is not None:
        finite &= np.asarray(initial_mask, dtype=bool)
    mask = finite.copy()

    sigma0 = max(float(bkg_rms_ref), 1.0)
    sigma_img = np.full_like(cut, sigma0, dtype=float)

    def design(m):
        cols = [trail_t[m]] + [st[m] for st in star_templates]
        cols += [np.ones(np.count_nonzero(m)), dx[m], dy[m]]
        return np.column_stack(cols)

    beta = None
    for _ in range(ROBUST_MAXITER):
        if np.count_nonzero(mask) < 50:
            raise RuntimeError("Too few valid pixels for joint fit")
        X = design(mask)
        y = cut[mask]
        sig = sigma_img[mask]
        Xw = X / sig[:, None]
        yw = y / sig
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        nstars = len(star_templates)
        source_model = beta[0] * trail_t
        for i, st in enumerate(star_templates):
            source_model = source_model + beta[1 + i] * st
        j0 = 1 + nstars
        background = beta[j0] + beta[j0 + 1] * dx + beta[j0 + 2] * dy
        model = source_model + background
        resid = cut - model
        sigma_img = np.sqrt(sigma0 ** 2 + np.clip(source_model, 0.0, None))

        rr = resid[mask]
        med = float(np.nanmedian(rr))
        mad = float(np.nanmedian(np.abs(rr - med)))
        rsig = 1.4826 * mad if mad > 0 else float(np.nanstd(rr))
        if not np.isfinite(rsig) or rsig <= 0:
            rsig = sigma0
        newmask = finite & (np.abs(resid - med) <= ROBUST_CLIP_SIGMA * rsig)
        if np.array_equal(newmask, mask):
            break
        mask = newmask

    X = design(mask)
    y = cut[mask]
    sig = sigma_img[mask]
    Xw = X / sig[:, None]
    yw = y / sig
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    nstars = len(star_templates)
    source_model = beta[0] * trail_t
    star_components = []
    for i, st in enumerate(star_templates):
        comp = beta[1 + i] * st
        star_components.append(comp)
        source_model = source_model + comp
    j0 = 1 + nstars
    background = beta[j0] + beta[j0 + 1] * dx + beta[j0 + 2] * dy
    model = source_model + background
    resid = cut - model

    chi2 = float(np.sum((resid[mask] / sigma_img[mask]) ** 2))
    dof = max(int(np.count_nonzero(mask)) - len(beta), 1)
    chi2_red = chi2 / dof
    normal = Xw.T @ Xw
    cov_beta = np.linalg.pinv(normal) * max(chi2_red, 1.0)
    errs = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))

    corr = np.full((1 + nstars, 1 + nstars), np.nan)
    for i in range(1 + nstars):
        for j in range(1 + nstars):
            den = math.sqrt(max(cov_beta[i, i] * cov_beta[j, j], 0.0))
            if den > 0:
                corr[i, j] = cov_beta[i, j] / den

    return {
        "beta": beta.astype(float),
        "beta_err": errs.astype(float),
        "cov_beta": cov_beta.astype(float),
        "source_corr": corr.astype(float),
        "model": model.astype(float),
        "trail_component": (beta[0] * trail_t).astype(float),
        "star_components": [x.astype(float) for x in star_components],
        "background_component": background.astype(float),
        "residual": resid.astype(float),
        "fit_mask": mask,
        "sigma_image": sigma_img.astype(float),
        "chi2": chi2,
        "chi2_red": chi2_red,
        "dof": dof,
        "n_fit_pixels": int(np.count_nonzero(mask)),
    }


def fit_scene(cut, xx, yy, trail_template, star_templates, scene_origin_xy,
              bkg_rms_ref, extra_mask=None, core_mask_centres=None, core_radius=0.0):
    mask = np.isfinite(cut)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    if core_mask_centres and core_radius > 0:
        for c in core_mask_centres:
            rr = np.hypot(xx - c[0], yy - c[1])
            mask &= rr >= float(core_radius)
    return _solve_weighted_linear(
        cut, xx, yy, trail_template, star_templates,
        scene_origin_xy=scene_origin_xy,
        bkg_rms_ref=bkg_rms_ref,
        initial_mask=mask,
    )


# =============================================================================
# Derived quantities / diagnostics
# =============================================================================


def derive_fit(fit: dict, aperture_fraction: np.ndarray, exptime: float,
               manual_ref: dict | None) -> dict:
    trail_total_counts = float(fit["beta"][0])
    trail_total_err = float(fit["beta_err"][0])
    trail_comp = fit["trail_component"]
    trail_rect_counts = float(np.nansum(trail_comp * aperture_fraction))
    # Since trail component scales linearly with beta[0], the same fraction
    # maps the coefficient error into the aperture-domain count error.
    frac_rect = trail_rect_counts / trail_total_counts if trail_total_counts != 0 else np.nan
    trail_rect_err = abs(frac_rect) * trail_total_err

    star_rows = []
    star_rect_sum = 0.0
    star_rect_var = 0.0
    for i, comp in enumerate(fit["star_components"]):
        counts_total = float(fit["beta"][1 + i])
        err_total = float(fit["beta_err"][1 + i])
        counts_rect = float(np.nansum(comp * aperture_fraction))
        frac = counts_rect / counts_total if counts_total != 0 else np.nan
        err_rect = abs(frac) * err_total if np.isfinite(frac) else np.nan
        star_rect_sum += counts_rect
        if np.isfinite(err_rect):
            star_rect_var += err_rect ** 2
        star_rows.append({
            "total_counts": counts_total,
            "total_counts_err_formal": err_total,
            "total_rate": counts_total / exptime,
            "total_rate_err_formal": err_total / exptime,
            "fraction_in_h13_horizons_rect": frac,
            "counts_in_h13_horizons_rect": counts_rect,
            "rate_in_h13_horizons_rect": counts_rect / exptime,
            "rate_in_h13_horizons_rect_err_formal": err_rect / exptime if np.isfinite(err_rect) else np.nan,
        })

    predicted_source_rect = trail_rect_counts + star_rect_sum
    out = {
        "trail_total_counts": trail_total_counts,
        "trail_total_counts_err_formal": trail_total_err,
        "trail_total_rate": trail_total_counts / exptime,
        "trail_total_rate_err_formal": trail_total_err / exptime,
        "trail_fraction_in_h13_horizons_rect": frac_rect,
        "trail_counts_in_h13_horizons_rect": trail_rect_counts,
        "trail_counts_in_h13_horizons_rect_err_formal": trail_rect_err,
        "trail_Rh_h13_horizons_rect": trail_rect_counts / exptime,
        "trail_Rh_h13_horizons_rect_err_formal": trail_rect_err / exptime,
        "stars": star_rows,
        "stars_counts_in_h13_horizons_rect_sum": star_rect_sum,
        "stars_rate_in_h13_horizons_rect_sum": star_rect_sum / exptime,
        "predicted_source_counts_in_h13_horizons_rect": predicted_source_rect,
    }
    if manual_ref is not None:
        delta = predicted_source_rect - float(manual_ref["net_counts"])
        sigma = float(manual_ref["net_counts_err"])
        out["closure_manual_label"] = manual_ref["label"]
        out["closure_delta_counts"] = delta
        out["closure_delta_over_manual_sigma"] = delta / sigma if sigma > 0 else np.nan
    return out


def serialisable_fit(fit: dict) -> dict:
    out = {}
    for k, v in fit.items():
        if isinstance(v, np.ndarray):
            if v.ndim <= 2 and v.size <= 100:
                out[k] = v.tolist()
            continue
        if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def save_png(cut, fit, bounds, hstart_xy, hend_xy, star_xy, path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x1, x2, y1, y2 = bounds
    extent = [x1 - 0.5, x2 - 0.5, y1 - 0.5, y2 - 0.5]
    star1 = fit["star_components"][0]
    star2 = fit["star_components"][1]
    resid_snr = fit["residual"] / np.where(fit["sigma_image"] > 0, fit["sigma_image"], np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        (cut, "Data"),
        (fit["model"], "Joint model"),
        (fit["residual"], "Residual"),
        (resid_snr, "Residual / fit sigma"),
        (fit["trail_component"], "Finite trail component"),
        (star1, "Star A1 component"),
        (star2, "Star A2 component"),
        (fit["background_component"], "Background plane"),
    ]

    for iax, (ax, (arr, title)) in enumerate(zip(axes.flat, panels)):
        im = ax.imshow(arr, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.colorbar(im, ax=ax, shrink=0.80)
        if iax < 4:
            ax.plot([hstart_xy[0], hend_xy[0]], [hstart_xy[1], hend_xy[1]], marker="o", linewidth=1)
            for j, p in enumerate(star_xy, start=1):
                ax.plot(p[0], p[1], marker="x", markersize=8)
                ax.text(p[0] + 0.5, p[1] + 0.5, f"A{j}", fontsize=8)

    fig.suptitle("Weisell: joint finite-trail + two-star + background fit")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def save_validation_png(cut, fit, bounds, hstart_xy, hend_xy, star_xy, path: Path) -> bool:
    """Compact diagnostic for the independent trail-only validation fit."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x1, x2, y1, y2 = bounds
    extent = [x1 - 0.5, x2 - 0.5, y1 - 0.5, y2 - 0.5]
    used_data = np.where(fit["fit_mask"], cut, np.nan)
    used_resid = np.where(fit["fit_mask"], fit["residual"], np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [
        (used_data, "Pixels used: trail-only validation"),
        (fit["trail_component"], "Recovered finite trail"),
        (fit["model"], "Trail + background model"),
        (used_resid, "Residual on fitted pixels"),
    ]
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.colorbar(im, ax=ax, shrink=0.80)
        ax.plot([hstart_xy[0], hend_xy[0]], [hstart_xy[1], hend_xy[1]], marker="o", linewidth=1)
        for j, pxy in enumerate(star_xy, start=1):
            ax.plot(pxy[0], pxy[1], marker="x", markersize=8)
            ax.text(pxy[0] + 0.5, pxy[1] + 0.5, f"A{j}", fontsize=8)

    fig.suptitle("Weisell: independent trail-only validation outside stellar cores")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# =============================================================================
# Final adopted C1/trail-only photometry -> pipeline calibration
# =============================================================================


def _parse_float_list(raw: str) -> list[float]:
    out = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _find_project_config() -> Path | None:
    """Find config.ini without requiring a fixed working directory."""
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / "config.ini"]
    # Expected placement: PROJECT/python/src/plots/this_script.py
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "config.ini")
    candidates += [
        BASE_DIR.parents[2] / "config.ini",  # defensive fallback
        Path("/Users/eracero/workspace/tfm-viu/config.ini"),
    ]
    seen = set()
    for q in candidates:
        q = q.expanduser()
        if str(q) in seen:
            continue
        seen.add(str(q))
        if q.exists():
            return q
    return None


def _read_coi_coeffs_from_config() -> tuple[list[float], str | None]:
    cfg_path = _find_project_config()
    if cfg_path is None:
        return [], None
    cfg = ConfigParser()
    cfg.read(cfg_path)
    coeffs = _parse_float_list(cfg.get("PHOTOMETRY", "COI_F_COEFFS", fallback=""))
    return coeffs, str(cfg_path)


def _read_header_keyword_any_hdu(path: Path, key: str, default=np.nan) -> float:
    key = str(key).upper()
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if key in hdu.header:
                try:
                    val = float(hdu.header[key])
                    if np.isfinite(val):
                        return val
                except Exception:
                    pass
    return float(default)


def _coi_polynomial(x: float, coeffs: list[float]) -> float:
    if not coeffs:
        return 1.0
    y = 0.0
    xp = 1.0
    for a in coeffs:
        y += float(a) * xp
        xp *= float(x)
    return float(y)


def _apply_coi(R_raw: float, frame_time_s: float, dead_fraction: float,
               coeffs: list[float]) -> float:
    """Same rate-domain CoI transformation used by the photometry pipeline."""
    R_raw = float(R_raw)
    tf = float(frame_time_s)
    df = float(dead_fraction)
    x = R_raw * tf
    if not np.isfinite(x) or tf <= 0 or not (0 <= df < 1):
        raise ValueError("Invalid CoI inputs")
    if x >= 1.0:
        raise ValueError("CoI saturation: R_raw * frame_time >= 1")
    if x <= 0:
        return R_raw
    base = -math.log1p(-x) / (tf * (1.0 - df))
    return float(base * _coi_polynomial(x, coeffs))


def _robust_sigma(values: np.ndarray) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 0:
        return 1.4826 * mad
    return float(np.std(v))


def derive_c1_background_and_final_row(
    fit1: dict,
    der1: dict,
    apfrac: np.ndarray,
    cut: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    origin: np.ndarray,
    exptime: float,
    image_header,
    srclist: Path,
) -> tuple[dict, dict]:
    """Integrate the C1 local background plane and run the closed UVW1 chain.

    The adopted astronomical source rate is the C1 trail-only rate inside the
    canonical h=13 Horizons rectangle.  The background entering CoI comes from
    the *same fitted local plane*, integrated over that same rectangle.

    The returned CSV row is intentionally standalone and is not appended to the
    main pipeline output automatically.  This avoids silently modifying the
    science catalogue; copy/merge it only after inspection.
    """
    valid = np.isfinite(cut) & np.isfinite(apfrac)
    w = np.where(valid, apfrac, 0.0)
    A_ap_eff = float(np.sum(w))
    if A_ap_eff <= 0:
        raise RuntimeError("Canonical h=13 aperture has zero effective area")

    bkg = np.asarray(fit1["background_component"], float)
    bkg_counts_ap = float(np.sum(w * bkg))
    bkg_per_pix = bkg_counts_ap / A_ap_eff
    R_bkg_6 = bkg_counts_ap / float(exptime)

    # Propagate the fitted background-plane coefficient covariance into the
    # integrated background counts.  For C1 the beta vector is
    # [trail, b0, bx, by].
    dx = xx - float(origin[0])
    dy = yy - float(origin[1])
    q = np.array([
        float(np.sum(w)),
        float(np.sum(w * dx)),
        float(np.sum(w * dy)),
    ])
    cov_beta = np.asarray(fit1["cov_beta"], float)
    cov_bkg = cov_beta[1:4, 1:4]
    var_bkg_counts = float(q @ cov_bkg @ q)
    bkg_counts_ap_err = math.sqrt(max(var_bkg_counts, 0.0))

    residual_rms = _robust_sigma(np.asarray(fit1["residual"])[fit1["fit_mask"]])

    R6 = float(der1["trail_Rh_h13_horizons_rect"])
    R6_err = float(der1["trail_Rh_h13_horizons_rect_err_formal"])
    net_counts = R6 * float(exptime)
    net_counts_err = R6_err * float(exptime)
    aper_counts_model = net_counts + bkg_counts_ap

    # C1 here is the *deblend method label*; pipeline geometrical C1 = 1 for h=13.
    C1_pipeline = 1.0
    C1_pipeline_err = 0.0

    # Detector timing terms from the actual image. FRAMTIME is stored in ms in
    # the OM products used by the pipeline.
    frame_time_ms = float(image_header["FRAMTIME"])
    frame_time_s = frame_time_ms * 1e-3
    dead_fraction = float(image_header.get("DEADFRAC", 0.0))
    coi_coeffs, coi_config_path = _read_coi_coeffs_from_config()

    R_tot_6 = R6 + R_bkg_6
    R_tot_6_coi = _apply_coi(R_tot_6, frame_time_s, dead_fraction, coi_coeffs)
    R_bkg_6_coi = _apply_coi(R_bkg_6, frame_time_s, dead_fraction, coi_coeffs)
    R6_coi = R_tot_6_coi - R_bkg_6_coi

    # Final adopted uncertainty prescription from the closed pipeline review.
    # Background covariance was checked separately and found negligible for the
    # final science errors; do not double count it here.
    denom = (1.0 - R_tot_6 * frame_time_s) * (1.0 - dead_fraction)
    if denom <= 0:
        raise RuntimeError("Invalid CoI derivative denominator")
    R6_coi_err = R6_err / denom

    # TDS may live in a non-primary SRCLIST HDU; search every HDU.
    tds_corr = _read_header_keyword_any_hdu(srclist, "TDS_CORR", default=np.nan)
    if not np.isfinite(tds_corr):
        raise RuntimeError("TDS_CORR not found in any SRCLIST HDU")

    R_final = R6_coi * C2_UVW1 * tds_corr
    R_final_err = math.sqrt(
        (C2_UVW1 * tds_corr * R6_coi_err) ** 2
        + (R6_coi * tds_corr * C2_UVW1_ERR) ** 2
    )
    mag_ab = -2.5 * math.log10(R_final) + ABM0_UVW1
    mag_err = 1.0857362047581294 * R_final_err / R_final
    coi_factor = R6_coi / R6 if R6 > 0 else float("nan")

    final = {
        "method": "C1_TRAIL_ONLY_OUTSIDE_A1_A2",
        "Rh": R6,
        "Rh_err": R6_err,
        "net_counts": net_counts,
        "net_counts_err": net_counts_err,
        "A_ap_eff": A_ap_eff,
        "background_counts_ap": bkg_counts_ap,
        "background_counts_ap_err_model": bkg_counts_ap_err,
        "background_counts_per_pix_mean": bkg_per_pix,
        "background_residual_rms_per_pix": residual_rms,
        "background_fit_npix": int(fit1["n_fit_pixels"]),
        "background_rate_6": R_bkg_6,
        "background_rate_6_coi": R_bkg_6_coi,
        "rate_tot_6": R_tot_6,
        "rate_tot_6_coi": R_tot_6_coi,
        "count_rate_coi": R6_coi,
        "count_rate_coi_err": R6_coi_err,
        "coi_factor": coi_factor,
        "frame_time_s": frame_time_s,
        "dead_fraction": dead_fraction,
        "coi_coeffs": coi_coeffs,
        "coi_config_path": coi_config_path,
        "tds_corr": tds_corr,
        "c2": C2_UVW1,
        "c2_err": C2_UVW1_ERR,
        "count_rate_final": R_final,
        "count_rate_final_err": R_final_err,
        "mag_ab": mag_ab,
        "mag_err": mag_err,
    }

    row = {
        "target_name": "Weisell",
        "observation_id": "0691070301",
        "filter": "L",
        "fits_name": IMAGE_FITS.name,
        "mag_ab": mag_ab,
        "mag_err": mag_err,
        "mag_ab_apcorr": mag_ab,
        "mag_ab_apcorr_err": mag_err,
        "count_rate": R6,
        "count_rate_err": R6_err,
        "net_counts": net_counts,
        "net_counts_err": net_counts_err,
        # Model source+background counts in the canonical deblended aperture.
        "aper_counts": aper_counts_model,
        "bkg_per_pix": bkg_per_pix,
        "bkg_rms_per_pix": residual_rms,
        "bkg_counts_ap": bkg_counts_ap,
        "bkg_counts_ap_err": bkg_counts_ap_err,
        "A_ap_eff": A_ap_eff,
        # For a plane fit there is no annular A_bg_eff.  Leave it undefined
        # rather than pretending the fit-support area is an annulus area.
        "A_bg_eff": float("nan"),
        "zp_ab": ABM0_UVW1,
        "trail_height_pix": AP_HEIGHT_PIX,
        # No photometric annulus is used by the deblend background model.
        "trail_semi_out_pix": float("nan"),
        "trail_semi_in_pix": float("nan"),
        "tds_corr": tds_corr,
        "c2": C2_UVW1,
        "c2_err": C2_UVW1_ERR,
        "f_apcorr": 1.0,
        "f_apcorr_err": 0.0,
        "apcorr_mag": 0.0,
        "apcorr_mag_err": 0.0,
        "c1": C1_pipeline,
        "c1_err": C1_pipeline_err,
        "c1_detail_csv": "",
        "coi_factor": coi_factor,
        "rate_bkg_6": R_bkg_6,
        "rate_bkg_6_coi": R_bkg_6_coi,
        "rate_tot_6": R_tot_6,
        "rate_tot_6_coi": R_tot_6_coi,
        "count_rate_coi": R6_coi,
        "count_rate_coi_err": R6_coi_err,
        "count_rate_final": R_final,
        "count_rate_final_err": R_final_err,
        "count_rate_final_apcorr": R_final,
        "count_rate_final_apcorr_err": R_final_err,
        "v_mag_1": VPRED,
        "v_mag_1_corrected": VPRED_UVW1_AB,
        "mlim_obs": MLIM_OBS,
    }
    return final, row


def write_final_row_csv(path: Path, row: dict) -> None:
    fields = [
        "target_name","observation_id","filter","fits_name","mag_ab","mag_err",
        "mag_ab_apcorr","mag_ab_apcorr_err","count_rate","count_rate_err",
        "net_counts","net_counts_err","aper_counts","bkg_per_pix","bkg_rms_per_pix",
        "bkg_counts_ap","bkg_counts_ap_err","A_ap_eff","A_bg_eff","zp_ab",
        "trail_height_pix","trail_semi_out_pix","trail_semi_in_pix","tds_corr",
        "c2","c2_err","f_apcorr","f_apcorr_err","apcorr_mag","apcorr_mag_err",
        "c1","c1_err","c1_detail_csv","coi_factor","rate_bkg_6",
        "rate_bkg_6_coi","rate_tot_6","rate_tot_6_coi","count_rate_coi",
        "count_rate_coi_err","count_rate_final","count_rate_final_err",
        "count_rate_final_apcorr","count_rate_final_apcorr_err","v_mag_1",
        "v_mag_1_corrected","mlim_obs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    if not IMAGE_FITS.exists():
        raise FileNotFoundError(IMAGE_FITS)
    srclist = autodiscover_srclist(BASE_DIR)

    with fits.open(IMAGE_FITS, memmap=False) as hdul:
        image_hdu_idx, image_hdu = find_image_hdu(hdul)
        header = image_hdu.header.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            data = np.array(image_hdu.data, dtype=np.float64, copy=True)

    wcs = WCS(header)
    exptime = float(header.get("EXPOSURE", EXPTIME_EXPECTED))
    if not np.isfinite(exptime) or exptime <= 0:
        raise RuntimeError("Invalid EXPOSURE")

    hstart_xy = sky_to_xy(wcs, HSTART_RA_DEG, HSTART_DEC_DEG)
    hend_xy = sky_to_xy(wcs, HEND_RA_DEG, HEND_DEC_DEG)
    trail_vec = hend_xy - hstart_xy
    trail_length_pix = float(np.linalg.norm(trail_vec))
    theta_rad = math.atan2(trail_vec[1], trail_vec[0])
    pixel_scale_along_arcsec = HORIZONS_TRAIL_LEN_ARCSEC / trail_length_pix

    contam_rows = []
    star_xy = []
    for c in CONTAMINANTS:
        row, sep = nearest_srclist_row(srclist, c["ra_deg"], c["dec_deg"])
        pos = sky_to_xy(wcs, float(row.get("RA_CORR", row.get("RA"))), float(row.get("DEC_CORR", row.get("DEC"))))
        c["srclist_match_sep_arcsec"] = sep
        contam_rows.append(row)
        star_xy.append(pos)
    star_xy = [np.asarray(p, float) for p in star_xy]

    field_psf = robust_field_psf(srclist, wcs, hstart_xy, hend_xy, contam_rows)
    cov_field = np.asarray(field_psf["cov"], float)

    # Scene covers finite trail and both stars, with ample background margin.
    cut, xx, yy, bounds = make_scene_cutout(data, [hstart_xy, hend_xy] + star_xy, CUTOUT_MARGIN_PIX)
    trail_t = integrated_finite_trail_template(xx, yy, hstart_xy, hend_xy, cov_field)
    star_t_field = [integrated_gaussian_template_cov(xx, yy, p, cov_field) for p in star_xy]

    bkg_rms_ref = float(np.median([x["bkg_rms_per_pix"] for x in MANUAL_EXTRACTIONS]))
    origin = 0.5 * (hstart_xy + hend_xy)

    # Automatically mask unrelated catalogue sources in the local cutout.  This
    # removes the bright residual object seen well below the trail without
    # introducing another nuisance amplitude into the joint model.
    unrelated_sources = find_unrelated_srclist_sources(
        srclist, wcs, bounds, hstart_xy, hend_xy, contam_rows
    )
    unrelated_xy = [np.array([q["x"], q["y"]], dtype=float) for q in unrelated_sources]
    common_fit_mask = circular_exclusion_mask(
        xx, yy, unrelated_xy, UNRELATED_SOURCE_MASK_RADIUS_PIX
    )

    # C0: primary simultaneous joint fit.
    fit0 = fit_scene(
        cut, xx, yy, trail_t, star_t_field, origin, bkg_rms_ref,
        extra_mask=common_fit_mask,
    )

    # C1: independent trail-only validation.  A1/A2 are not modelled at all;
    # instead a generous radius around both stars is excluded and only the
    # unobscured portions of the finite Horizons trail constrain its amplitude.
    trail_only_mask = circular_exclusion_mask(
        xx, yy, star_xy, TRAIL_ONLY_STAR_MASK_RADIUS_PIX, base_mask=common_fit_mask
    )
    fit1 = fit_scene(
        cut, xx, yy, trail_t, [], origin, bkg_rms_ref,
        extra_mask=trail_only_mask,
    )

    # C2: source-specific star PSFs as sensitivity diagnostic; trail remains field PSF.
    star_t_src = []
    source_psf_info = []
    for row, p in zip(contam_rows, star_xy):
        cov_src = psf_covariance(float(row["FWHM_MAJ"]), float(row["FWHM_MIN"]), float(row["PA"]))
        star_t_src.append(integrated_gaussian_template_cov(xx, yy, p, cov_src))
        source_psf_info.append({
            "fwhm_maj": float(row["FWHM_MAJ"]),
            "fwhm_min": float(row["FWHM_MIN"]),
            "pa_deg": float(row["PA"]),
        })
    fit2 = fit_scene(
        cut, xx, yy, trail_t, star_t_src, origin, bkg_rms_ref,
        extra_mask=common_fit_mask,
    )

    # Canonical h=13 rectangle exactly spanning Horizons START->END.
    apfrac = rectangular_fraction_map(xx, yy, hstart_xy, hend_xy, AP_HEIGHT_PIX)

    # Which manual h=13 extraction is geometrically closest to the Horizons span?
    for m in MANUAL_EXTRACTIONS:
        m["effective_width_pix"] = float(m["A_ap_eff"] / m["height_pix"])
        m["delta_width_vs_horizons_pix"] = float(m["effective_width_pix"] - trail_length_pix)
    manual_ref = min(MANUAL_EXTRACTIONS, key=lambda m: abs(m["delta_width_vs_horizons_pix"]))

    der0 = derive_fit(fit0, apfrac, exptime, manual_ref)
    # C1 deliberately contains no stellar model, so a closure comparison to
    # the contaminated manual h=13 aperture would be meaningless.
    der1 = derive_fit(fit1, apfrac, exptime, None)
    der2 = derive_fit(fit2, apfrac, exptime, manual_ref)

    trail_template_sum = float(np.nansum(trail_t))
    trail_template_fraction_used_c1 = (
        float(np.nansum(trail_t[fit1["fit_mask"]])) / trail_template_sum
        if trail_template_sum > 0 else np.nan
    )

    # Final adopted Weisell solution: the trail-only C1 validation is used as
    # the science estimator.  Integrate its fitted background plane over the
    # same canonical h=13 Horizons aperture and pass the resulting R_h through
    # the closed CoI -> C2 -> TDS -> AB chain.
    final_c1, final_row = derive_c1_background_and_final_row(
        fit1=fit1, der1=der1, apfrac=apfrac, cut=cut, xx=xx, yy=yy,
        origin=origin, exptime=exptime, image_header=header, srclist=srclist,
    )

    # Console report.
    print("\n=== WEISELL JOINT DEBLEND ===")
    print(f"Image HDU                         : {image_hdu_idx}")
    print(f"Image                            : {IMAGE_FITS}")
    print(f"SRCLIST auto                     : {srclist}")
    print(f"EXPOSURE                         : {exptime:.6f} s")
    print(f"Horizons START xy                : {hstart_xy[0]:.4f}, {hstart_xy[1]:.4f}")
    print(f"Horizons END   xy                : {hend_xy[0]:.4f}, {hend_xy[1]:.4f}")
    print(f"Horizons trail length            : {trail_length_pix:.4f} pix = {HORIZONS_TRAIL_LEN_ARCSEC:.3f} arcsec")
    print(f"Approx pixel scale along trail   : {pixel_scale_along_arcsec:.4f} arcsec/pix")
    print(f"Trail theta in image             : {math.degrees(theta_rad):.5f} deg")
    print(f"Horizons 3-sigma positional unc. : {HORIZONS_UNC_3SIGMA_ARCSEC:.4f} arcsec")

    print("\n--- Contaminants recovered from SRCLIST ---")
    for i, (c, row, p) in enumerate(zip(CONTAMINANTS, contam_rows, star_xy), start=1):
        tpx, alongpx, perppx = point_segment_geometry(p, hstart_xy, hend_xy)
        tsky, alongas, perpas = sky_point_segment_geometry_arcsec(c["ra_deg"], c["dec_deg"])
        print(f"A{i}: UI idx={c['ui_idx']} -> raw FITS row0={row['_ROW0']} SRCNUM={row.get('SRCNUM')} SRC_ID={row.get('SRC_ID')}")
        print(f"    match separation             : {c['srclist_match_sep_arcsec']:.4f} arcsec")
        print(f"    WCS xy                       : {p[0]:.4f}, {p[1]:.4f}")
        print(f"    sky segment t/along/perp     : {tsky:+.4f}, {alongas:+.3f}, {perpas:+.3f} arcsec")
        print(f"    pixel segment t/along/perp   : {tpx:+.4f}, {alongpx:+.3f}, {perppx:+.3f} pix")
        print(f"    RATE diagnostic              : {float(row.get('RATE', np.nan)):.6f} +/- {float(row.get('RATE_ERR', np.nan)):.6f} ct/s")
        print(f"    FWHM maj,min / PA            : {float(row['FWHM_MAJ']):.3f}, {float(row['FWHM_MIN']):.3f} px / {float(row['PA']):.2f} deg")
        print(f"    flags Q/C/E                  : {row.get('QFLAG')} / {row.get('CFLAG')} / {row.get('EFLAG')}")

    print("\n--- Robust field PSF [PRIMARY] ---")
    print(f"N clean SRCLIST stars            : {field_psf['n_stars']}")
    print(f"FWHM maj,min                     : {field_psf['fwhm_maj']:.4f}, {field_psf['fwhm_min']:.4f} pix")
    print(f"PA                                : {field_psf['pa_deg']:.3f} deg")
    if field_psf["srcnums"]:
        print("SRCNUMs used                     : " + ",".join(map(str, field_psf["srcnums"])))

    print("\n--- Automatically masked unrelated SRCLIST sources ---")
    print(f"N unrelated sources masked       : {len(unrelated_sources)}")
    for k, q in enumerate(unrelated_sources, start=1):
        print(
            f"M{k}: SRCNUM={q['srcnum']} row0={q['row0']} "
            f"xy=({q['x']:.2f},{q['y']:.2f}) dtrail={q['distance_to_trail_pix']:.2f}px "
            f"RATE={q['rate']:.4f} sig={q['significance']:.2f}"
        )

    print("\n--- Manual h=13 geometry controls ---")
    for m in MANUAL_EXTRACTIONS:
        print(
            f"{m['label']}: R_h={m['rh']:.9f}+/-{m['rh_err']:.9f}, "
            f"Aeff/h={m['effective_width_pix']:.3f}px, "
            f"Delta width vs Horizons={m['delta_width_vs_horizons_pix']:+.3f}px"
        )
    print(f"Closure reference chosen         : {manual_ref['label']} (closest Aeff/h to Horizons length)")

    def print_joint_fit(label, fit, der):
        c = fit["source_corr"]
        print(f"\n--- {label} ---")
        print(f"chi2_red / Npix                  : {fit['chi2_red']:.3f} / {fit['n_fit_pixels']}")
        print(f"corr(trail,A1)                   : {c[0,1]:+.4f}")
        print(f"corr(trail,A2)                   : {c[0,2]:+.4f}")
        print(f"corr(A1,A2)                      : {c[1,2]:+.4f}")
        print(f"asteroid TOTAL finite-trail rate : {der['trail_total_rate']:.9f} +/- {der['trail_total_rate_err_formal']:.9f} ct/s")
        print(f"asteroid frac inside h13-Horizons: {der['trail_fraction_in_h13_horizons_rect']:.6f}")
        print(f"ASTEROID R_h h13-Horizons        : {der['trail_Rh_h13_horizons_rect']:.9f} +/- {der['trail_Rh_h13_horizons_rect_err_formal']:.9f} ct/s [formal fit]")
        for i, srow in enumerate(der["stars"], start=1):
            print(f"A{i} total joint-fit rate         : {srow['total_rate']:.6f} +/- {srow['total_rate_err_formal']:.6f} ct/s")
            print(f"A{i} rate inside h13-Horizons     : {srow['rate_in_h13_horizons_rect']:.6f} ct/s")
        print(f"A1+A2 inside h13-Horizons        : {der['stars_rate_in_h13_horizons_rect_sum']:.6f} ct/s")
        print(f"source closure in h13-Horizons   : {der['predicted_source_counts_in_h13_horizons_rect']:.2f} model counts")
        if "closure_delta_counts" in der:
            print(
                f"closure vs {der['closure_manual_label']}        : "
                f"Delta={der['closure_delta_counts']:+.2f} counts "
                f"({der['closure_delta_over_manual_sigma']:+.2f} x manual net-count sigma)"
            )

    def print_trail_only_fit(label, fit, der):
        print(f"\n--- {label} ---")
        print(f"chi2_red / Npix                  : {fit['chi2_red']:.3f} / {fit['n_fit_pixels']}")
        print(f"star exclusion radius            : {TRAIL_ONLY_STAR_MASK_RADIUS_PIX:.2f} pix")
        print(f"trail-template weight retained   : {trail_template_fraction_used_c1:.4f}")
        print("stellar amplitudes fitted         : NONE")
        print(f"asteroid TOTAL finite-trail rate : {der['trail_total_rate']:.9f} +/- {der['trail_total_rate_err_formal']:.9f} ct/s")
        print(f"asteroid frac inside h13-Horizons: {der['trail_fraction_in_h13_horizons_rect']:.6f}")
        print(f"ASTEROID R_h h13-Horizons        : {der['trail_Rh_h13_horizons_rect']:.9f} +/- {der['trail_Rh_h13_horizons_rect_err_formal']:.9f} ct/s [formal fit]")
        print("closure vs contaminated h13       : not applicable (stars deliberately excluded)")

    print_joint_fit("C0 FIELD-PSF JOINT FIT [PRIMARY]", fit0, der0)
    print_trail_only_fit(
        f"C1 TRAIL-ONLY OUTSIDE A1/A2 r<{TRAIL_ONLY_STAR_MASK_RADIUS_PIX:.1f}px [VALIDATION]",
        fit1, der1
    )
    print_joint_fit("C2 SOURCE-SPECIFIC STAR PSFs [SENSITIVITY]", fit2, der2)

    print("\n--- Stability diagnostics ---")
    print(f"trail-only - primary Delta R_h   : {der1['trail_Rh_h13_horizons_rect'] - der0['trail_Rh_h13_horizons_rect']:+.9f} ct/s")
    print(f"srcPSF - primary Delta R_h       : {der2['trail_Rh_h13_horizons_rect'] - der0['trail_Rh_h13_horizons_rect']:+.9f} ct/s")

    print("\n--- ADOPTED WEISELL PHOTOMETRY: C1 TRAIL-ONLY ---")
    print(f"A_ap_eff h13-Horizons            : {final_c1['A_ap_eff']:.6f} pix")
    print(f"background plane mean            : {final_c1['background_counts_per_pix_mean']:.9f} counts/pix")
    print(f"background residual RMS          : {final_c1['background_residual_rms_per_pix']:.9f} counts/pix")
    print(f"background counts in h13         : {final_c1['background_counts_ap']:.6f} +/- {final_c1['background_counts_ap_err_model']:.6f} counts [model]")
    print(f"R_bkg_6                          : {final_c1['background_rate_6']:.9f} ct/s")
    print(f"R_bkg_6_CoI                      : {final_c1['background_rate_6_coi']:.9f} ct/s")
    print(f"R_tot_6                          : {final_c1['rate_tot_6']:.9f} ct/s")
    print(f"R_tot_6_CoI                      : {final_c1['rate_tot_6_coi']:.9f} ct/s")
    print(f"R_6_CoI                          : {final_c1['count_rate_coi']:.9f} +/- {final_c1['count_rate_coi_err']:.9f} ct/s")
    print(f"coi_factor (=R6CoI/R6)           : {final_c1['coi_factor']:.9f}")
    print(f"FRAMTIME / DEADFRAC              : {final_c1['frame_time_s']:.9f} s / {final_c1['dead_fraction']:.9f}")
    print(f"COI_F_COEFFS                     : {final_c1['coi_coeffs']}")
    print(f"TDS / C2                         : {final_c1['tds_corr']:.12f} / {final_c1['c2']:.10f} +/- {final_c1['c2_err']:.10f}")
    print(f"FINAL corrected rate             : {final_c1['count_rate_final']:.9f} +/- {final_c1['count_rate_final_err']:.9f} ct/s")
    print(f"FINAL m_AB                       : {final_c1['mag_ab']:.9f} +/- {final_c1['mag_err']:.9f}")

    # Standalone row; do not mutate the main photometry catalogue automatically.
    write_final_row_csv(OUT_FINAL_ROW_CSV, final_row)
    row_values = [str(final_row[k]) for k in [
        "target_name","observation_id","filter","fits_name","mag_ab","mag_err",
        "mag_ab_apcorr","mag_ab_apcorr_err","count_rate","count_rate_err",
        "net_counts","net_counts_err","aper_counts","bkg_per_pix","bkg_rms_per_pix",
        "bkg_counts_ap","bkg_counts_ap_err","A_ap_eff","A_bg_eff","zp_ab",
        "trail_height_pix","trail_semi_out_pix","trail_semi_in_pix","tds_corr",
        "c2","c2_err","f_apcorr","f_apcorr_err","apcorr_mag","apcorr_mag_err",
        "c1","c1_err","c1_detail_csv","coi_factor","rate_bkg_6",
        "rate_bkg_6_coi","rate_tot_6","rate_tot_6_coi","count_rate_coi",
        "count_rate_coi_err","count_rate_final","count_rate_final_err",
        "count_rate_final_apcorr","count_rate_final_apcorr_err","v_mag_1",
        "v_mag_1_corrected","mlim_obs"
    ]]
    print("\nFINAL CSV ROW (copy into the adopted photometry table if desired):")
    print(",".join(row_values))

    # Save diagnostics.
    np.savez_compressed(
        OUT_NPZ,
        cutout=cut,
        x_grid=xx,
        y_grid=yy,
        hstart_xy=hstart_xy,
        hend_xy=hend_xy,
        star1_xy=star_xy[0],
        star2_xy=star_xy[1],
        aperture_fraction_h13_horizons=apfrac,
        trail_template=trail_t,
        star1_template_field=star_t_field[0],
        star2_template_field=star_t_field[1],
        primary_model=fit0["model"],
        primary_trail_component=fit0["trail_component"],
        primary_star1_component=fit0["star_components"][0],
        primary_star2_component=fit0["star_components"][1],
        primary_background_component=fit0["background_component"],
        primary_residual=fit0["residual"],
        primary_fit_mask=fit0["fit_mask"],
        common_fit_mask=common_fit_mask,
        trail_only_model=fit1["model"],
        trail_only_trail_component=fit1["trail_component"],
        trail_only_residual=fit1["residual"],
        trail_only_fit_mask=fit1["fit_mask"],
        srcpsf_model=fit2["model"],
        srcpsf_residual=fit2["residual"],
    )

    result = {
        "case": {
            "target": "Weisell",
            "obsid": "0691070301",
            "filter": "L/UVW1",
            "image_fits": str(IMAGE_FITS),
            "srclist_fits": str(srclist),
            "exptime_s": exptime,
        },
        "horizons": {
            "start_ra_deg": HSTART_RA_DEG,
            "start_dec_deg": HSTART_DEC_DEG,
            "end_ra_deg": HEND_RA_DEG,
            "end_dec_deg": HEND_DEC_DEG,
            "start_xy": hstart_xy.tolist(),
            "end_xy": hend_xy.tolist(),
            "trail_length_arcsec": HORIZONS_TRAIL_LEN_ARCSEC,
            "trail_length_pix": trail_length_pix,
            "theta_image_deg": math.degrees(theta_rad),
            "pa_E_of_N_deg_from_validation": HORIZONS_PA_E_OF_N_DEG,
            "unc_3sigma_arcsec": HORIZONS_UNC_3SIGMA_ARCSEC,
        },
        "contaminants": [],
        "field_psf_primary": {
            "fwhm_maj": field_psf["fwhm_maj"],
            "fwhm_min": field_psf["fwhm_min"],
            "pa_deg": field_psf["pa_deg"],
            "n_stars": field_psf["n_stars"],
            "srcnums": field_psf["srcnums"],
            "row_indices": field_psf["row_indices"],
        },
        "manual_extractions": MANUAL_EXTRACTIONS,
        "closure_reference": manual_ref["label"],
        "unrelated_sources_masked": unrelated_sources,
        "fit_primary_field_psf": {"fit": serialisable_fit(fit0), "derived": der0},
        "fit_trail_only_outside_stars": {
            "fit": serialisable_fit(fit1),
            "derived": der1,
            "star_mask_radius_pix": TRAIL_ONLY_STAR_MASK_RADIUS_PIX,
            "trail_template_fraction_used": trail_template_fraction_used_c1,
        },
        "fit_source_specific_psf_sensitivity": {"fit": serialisable_fit(fit2), "derived": der2},
        "adopted_photometry_C1_trail_only": {
            "calibration": final_c1,
            "pipeline_row": final_row,
            "standalone_csv": str(OUT_FINAL_ROW_CSV),
        },
        "method_notes": [
            "UI idx values are not treated as raw FITS row numbers; contaminants are matched by RA/Dec.",
            "The primary trail geometry is fixed to frame-specific JPL Horizons START/END.",
            "The primary fit uses a robust field PSF because the contaminant FWHM/PA can itself be biased by overlap with the asteroid trail.",
            "The fit is simultaneous: finite asteroid trail + A1 + A2 + tilted local background.",
            "SRCLIST RATE and CORR_RATE are diagnostics only and are not used as priors or direct subtractions.",
            "The primary pipeline-domain output is the asteroid model rate inside the canonical h=13 rectangle spanning Horizons START->END.",
            "The trail-only validation masks A1/A2 generously and fits no stellar amplitudes at all; Weisell is constrained only by unobscured trail pixels plus a local background plane.",
            "Unrelated catalogue sources in the cutout are masked from all fits so they cannot bias the background plane or chi-square.",
            "Formal fit errors are local model errors; model/PSF sensitivity should be considered separately before adopting a final uncertainty.",
            "If a deblended R_h is adopted, continue through the normal CoI -> C2 -> TDS -> m_AB chain; do not correct the final magnitude post hoc.",
        ],
    }

    for c, row, p, psi in zip(CONTAMINANTS, contam_rows, star_xy, source_psf_info):
        tsky, alongas, perpas = sky_point_segment_geometry_arcsec(c["ra_deg"], c["dec_deg"])
        result["contaminants"].append({
            "name": c["name"],
            "ui_idx": c["ui_idx"],
            "raw_row0": int(row["_ROW0"]),
            "srcnum": int(row.get("SRCNUM", -1)),
            "src_id": int(row.get("SRC_ID", -1)),
            "match_sep_arcsec": float(c["srclist_match_sep_arcsec"]),
            "ra_corr": float(row.get("RA_CORR", np.nan)),
            "dec_corr": float(row.get("DEC_CORR", np.nan)),
            "xy": p.tolist(),
            "segment_t_sky": tsky,
            "along_from_start_arcsec": alongas,
            "perp_from_trail_arcsec": perpas,
            "rate": float(row.get("RATE", np.nan)),
            "rate_err": float(row.get("RATE_ERR", np.nan)),
            "corr_rate": float(row.get("CORR_RATE", np.nan)),
            "significance": float(row.get("SIGNIFICANCE", np.nan)),
            "qflag": int(row.get("QFLAG", 0)),
            "cflag": int(row.get("CFLAG", 0)),
            "eflag": int(row.get("EFLAG", 0)),
            "source_specific_psf": psi,
        })

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    png_ok = save_png(cut, fit0, bounds, hstart_xy, hend_xy, star_xy, OUT_PNG)
    validation_png_ok = save_validation_png(
        cut, fit1, bounds, hstart_xy, hend_xy, star_xy, OUT_VALIDATION_PNG
    )
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_NPZ}")
    print(f"Saved: {OUT_FINAL_ROW_CSV}")
    if png_ok:
        print(f"Saved: {OUT_PNG}")
    else:
        print("Diagnostic PNG skipped (matplotlib unavailable).")
    if validation_png_ok:
        print(f"Saved: {OUT_VALIDATION_PNG}")
    else:
        print("Validation PNG skipped (matplotlib unavailable).")

    print("\nInterpretation checklist:")
    print("  (1) Inspect the PRIMARY residual for coherent remaining trail/star structure.")
    print("  (2) Inspect corr(trail,A1), corr(trail,A2), corr(A1,A2); large |corr| means degeneracy.")
    print("  (3) Compare PRIMARY vs TRAIL-ONLY-OUTSIDE-STARS R_h. This is now the key Weisell robustness test.")
    print("  (4) In the validation PNG, verify that the recovered trail is supported by pixels outside A1/A2 masks.")
    print("  (5) Compare field-PSF vs source-specific-PSF sensitivity.")
    print("  (6) Closure against the nearest-width manual h=13 extraction is diagnostic only.")


if __name__ == "__main__":
    main()
