#!/usr/bin/env python3
"""
Joint deblending of (715) Transvaalia / OBSID 0821871601 / XMM-OM UVW1.

This script is deliberately case-specific and reproducible.  It fits the local
image around the contaminating stationary source with a *simultaneous* model

    image = asteroid_trail + contaminating_star + local_background

where

  * the star is an elliptical 2-D Gaussian PSF whose centroid and shape are
    read directly from the OM SRCLIST row SRCNUM=56;
  * the asteroid is represented as the same PSF convolved with an effectively
    uniform straight line (the local limit of a finite trail).  The fitted
    coefficient is therefore the asteroid counts per pixel of trail length;
  * the background is a tilted plane b0 + bx*x + by*y.

The fit is linear in the flux parameters.  A small optional grid refinement of
trail angle / perpendicular offset is also performed as a geometry sensitivity
check; the fixed-geometry result remains the primary result because it maps
most directly to the h=13 aperture used by the original pipeline extraction.

Important calibration-domain rule
---------------------------------
The script stops at the deblended *raw/background-subtracted* rate R_h.  For
this h=13 extraction C1=1, so R_h is the input R_6 to the normal pipeline.
After deblending, re-run the standard nonlinear

    CoI -> C2 -> TDS -> m_AB

chain.  Do NOT subtract anything from count_rate_final / mag_ab afterwards.

The raw SRCLIST RATE is printed only as a diagnostic.  It is not used as a
prior or subtraction in the joint fit.

Outputs
-------
  transvaalia_joint_deblend_result.json
  transvaalia_joint_deblend_diagnostic.npz
  transvaalia_joint_deblend_diagnostic.png   (if matplotlib is available)

Run from anywhere; edit only the two FITS paths below if needed.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


# ============================================================================
# CASE CONFIGURATION -- Transvaalia values copied from the final rerun/QC
# ============================================================================

IMAGE_FITS = Path(
    "/Users/eracero/workspace/tfm-viu/temp/0821871601/L/"
    "P0821871601OMS006FSIMAGL000.FTZ"
)
SRCLIST_FITS = Path(
    "/Users/eracero/workspace/tfm-viu/temp/0821871601/L/"
    "P0821871601OMS006SWSRLIL000.FTZ"
)

SRCNUM = 56

# Large h=13 extraction containing the contaminating star.
RH_LARGE = 1.5185142560758396
RH_LARGE_ERR = 0.038237590348100675
NET_COUNTS_LARGE = 7592.571280379198
NET_COUNTS_ERR_LARGE = 191.18795174050337
APER_COUNTS_LARGE = 28525.892600506544
BKG_MEAN_LARGE = 21.1337826786801
BKG_RMS_LARGE = 3.151241630204799
A_AP_EFF_LARGE = 990.5146484375
A_BG_EFF_LARGE = 1213.7666015625
EXPTIME_EXPECTED = 5000.0

# Independent narrow-box solution, used ONLY as a cross-check.
RH_NARROW = 1.337193280403132
RH_NARROW_ERR = 0.026838516983272893
C1_NARROW = 1.066871148297062
C1_NARROW_ERR = 0.010459152041094774

# Original large-aperture geometry available from the photometry session.
# The exact width was not stored in the historical CSV; A_eff / h is therefore
# used as the reproducible effective longitudinal length.
AP_HEIGHT_PIX = 13.0
AP_THETA_RAD = -0.1461
TRAIL_ANCHOR_XY = np.array([453.74, 1121.63], dtype=float)

# Local joint-fit cutout around the contaminating point source.
JOINT_CUTOUT_HALF_SIZE = 18
SUBPIX = 5
ROBUST_CLIP_SIGMA = 5.0
ROBUST_MAXITER = 4

# Small geometry sensitivity search.  Primary = fixed AP_THETA_RAD, offset=0.
REFINE_GEOMETRY = True
COARSE_THETA_HALF_RANGE_DEG = 2.0
COARSE_THETA_STEP_DEG = 0.25
COARSE_OFFSET_HALF_RANGE_PIX = 1.0
COARSE_OFFSET_STEP_PIX = 0.20
REFINE_THETA_HALF_RANGE_DEG = 0.30
REFINE_THETA_STEP_DEG = 0.05
REFINE_OFFSET_HALF_RANGE_PIX = 0.30
REFINE_OFFSET_STEP_PIX = 0.05

OUT_JSON = Path("transvaalia_joint_deblend_result.json")
OUT_NPZ = Path("transvaalia_joint_deblend_diagnostic.npz")
OUT_PNG = Path("transvaalia_joint_deblend_diagnostic.png")


FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# ============================================================================
# FITS / SRCLIST helpers
# ============================================================================


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


def get_srclist_row(srclist: Path, srcnum: int) -> dict:
    with fits.open(srclist, memmap=False) as hdul:
        ihdu, hdu = find_table_hdu(hdul)
        tab = hdu.data
        names = list(tab.columns.names)
        up = {n.upper(): n for n in names}
        if "SRCNUM" not in up:
            raise KeyError("SRCLIST has no SRCNUM column")

        vals = np.asarray(tab[up["SRCNUM"]], dtype=int)
        idx = np.flatnonzero(vals == int(srcnum))
        if len(idx) != 1:
            raise RuntimeError(f"Expected exactly one SRCNUM={srcnum}; found {len(idx)}")

        j = int(idx[0])
        row = {}
        for name in names:
            v = tab[name][j]
            row[name] = v.item() if hasattr(v, "item") else v
        row["_HDU"] = ihdu
        row["_ROW0"] = j
        return row


def source_xy_from_wcs(image_header: fits.Header, row: dict) -> np.ndarray:
    ra = float(row.get("RA_CORR", row.get("RA")))
    dec = float(row.get("DEC_CORR", row.get("DEC")))
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    wcs = WCS(image_header)
    x, y = wcs.world_to_pixel(coord)
    return np.array([float(x), float(y)], dtype=float)


# ============================================================================
# PSF / trail templates
# ============================================================================


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def psf_covariance(fwhm_maj: float, fwhm_min: float, pa_deg: float) -> np.ndarray:
    """Elliptical Gaussian covariance in image x/y pixels.

    This uses the same PA convention as the previous Transvaalia diagnostic:
    PA is interpreted as anti-clockwise from +x in image pixel coordinates.
    """
    smaj = float(fwhm_maj) * FWHM_TO_SIGMA
    smin = float(fwhm_min) * FWHM_TO_SIGMA
    pa = math.radians(float(pa_deg))

    umaj = np.array([math.cos(pa), math.sin(pa)], dtype=float)
    umin = np.array([-math.sin(pa), math.cos(pa)], dtype=float)
    return (smaj ** 2) * np.outer(umaj, umaj) + (smin ** 2) * np.outer(umin, umin)


def sigma_perp_from_cov(cov: np.ndarray, theta_rad: float) -> float:
    n = np.array([-math.sin(theta_rad), math.cos(theta_rad)], dtype=float)
    return float(math.sqrt(n @ cov @ n))


def gaussian_aperture_fraction_1d(mu: float, sigma: float, full_height: float) -> float:
    half = 0.5 * float(full_height)
    z_hi = (half - float(mu)) / float(sigma)
    z_lo = (-half - float(mu)) / float(sigma)
    return float(normal_cdf(z_hi) - normal_cdf(z_lo))


def _subpixel_offsets(nsub: int) -> np.ndarray:
    # Pixel spans centre +/- 0.5.  Sample subpixel centres uniformly.
    return (np.arange(nsub, dtype=float) + 0.5) / nsub - 0.5


def integrated_star_template(
    xx: np.ndarray,
    yy: np.ndarray,
    star_xy: np.ndarray,
    fwhm_maj: float,
    fwhm_min: float,
    pa_deg: float,
    nsub: int = SUBPIX,
) -> np.ndarray:
    """Pixel-integrated normalized elliptical Gaussian PSF.

    Sum over an infinitely large image would be 1.  The coefficient multiplying
    this template is therefore the total point-source counts.
    """
    x0, y0 = map(float, star_xy)
    smaj = float(fwhm_maj) * FWHM_TO_SIGMA
    smin = float(fwhm_min) * FWHM_TO_SIGMA
    pa = math.radians(float(pa_deg))
    c, s = math.cos(pa), math.sin(pa)
    norm = 1.0 / (2.0 * math.pi * smaj * smin)

    out = np.zeros_like(xx, dtype=float)
    offs = _subpixel_offsets(nsub)
    for oy in offs:
        for ox in offs:
            dx = (xx + ox) - x0
            dy = (yy + oy) - y0
            xp = c * dx + s * dy
            yp = -s * dx + c * dy
            out += norm * np.exp(-0.5 * ((xp / smaj) ** 2 + (yp / smin) ** 2))
    out /= float(nsub * nsub)
    return out


def integrated_infinite_trail_template(
    xx: np.ndarray,
    yy: np.ndarray,
    anchor_xy: np.ndarray,
    theta_rad: float,
    cov: np.ndarray,
    perp_offset_pix: float = 0.0,
    nsub: int = SUBPIX,
) -> np.ndarray:
    """Pixel-integrated PSF-convolved infinite straight trail for unit line flux.

    The unconvolved trail has one count per pixel of distance along the trail.
    After convolution with a normalized 2-D Gaussian PSF, the image intensity is
    the 1-D Gaussian marginal perpendicular to the trail.  Therefore the fit
    coefficient has units counts / pixel-of-trail-length.
    """
    theta = float(theta_rad)
    n = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    sigma_perp = sigma_perp_from_cov(cov, theta)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_perp)

    out = np.zeros_like(xx, dtype=float)
    offs = _subpixel_offsets(nsub)
    ax, ay = map(float, anchor_xy)
    off = float(perp_offset_pix)

    for oy in offs:
        for ox in offs:
            v = ((xx + ox) - ax) * n[0] + ((yy + oy) - ay) * n[1] - off
            out += norm * np.exp(-0.5 * (v / sigma_perp) ** 2)
    out /= float(nsub * nsub)
    return out


# ============================================================================
# Linear joint fit
# ============================================================================


def make_cutout(data: np.ndarray, star_xy: np.ndarray, half_size: int):
    x0, y0 = map(float, star_xy)
    ny, nx = data.shape
    x1 = max(0, int(math.floor(x0)) - int(half_size))
    x2 = min(nx, int(math.floor(x0)) + int(half_size) + 1)
    y1 = max(0, int(math.floor(y0)) - int(half_size))
    y2 = min(ny, int(math.floor(y0)) + int(half_size) + 1)
    cut = np.asarray(data[y1:y2, x1:x2], dtype=float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    return cut, xx.astype(float), yy.astype(float), (x1, x2, y1, y2)


def _solve_weighted_linear(
    cut: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    trail_t: np.ndarray,
    star_t: np.ndarray,
    star_xy: np.ndarray,
    initial_mask: np.ndarray | None = None,
) -> dict:
    """Robust weighted linear fit.

    Parameters fitted:
      beta[0] = trail line density [counts / pix of trail length]
      beta[1] = total star counts
      beta[2] = background constant [counts / pix]
      beta[3] = background x gradient [counts / pix / pix]
      beta[4] = background y gradient [counts / pix / pix]
    """
    x0, y0 = map(float, star_xy)
    dx = xx - x0
    dy = yy - y0

    finite = np.isfinite(cut) & np.isfinite(trail_t) & np.isfinite(star_t)
    if initial_mask is not None:
        finite &= np.asarray(initial_mask, dtype=bool)
    mask = finite.copy()

    beta = None
    cov_beta = None
    sigma_img = np.full_like(cut, max(float(BKG_RMS_LARGE), 1.0), dtype=float)

    for iteration in range(ROBUST_MAXITER):
        if np.count_nonzero(mask) < 40:
            raise RuntimeError("Too few finite pixels for joint fit")

        X = np.column_stack([
            trail_t[mask],
            star_t[mask],
            np.ones(np.count_nonzero(mask)),
            dx[mask],
            dy[mask],
        ])
        y = cut[mask]
        sig = sigma_img[mask]
        Xw = X / sig[:, None]
        yw = y / sig

        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        source_model = beta[0] * trail_t + beta[1] * star_t
        model = source_model + beta[2] + beta[3] * dx + beta[4] * dy
        resid = cut - model

        # Processed OM image: combine measured local-background RMS with a
        # Poisson-like source term.  This is a fitting weight, not the pipeline
        # photometric uncertainty model.
        sigma_img = np.sqrt(
            max(float(BKG_RMS_LARGE), 1.0) ** 2 + np.clip(source_model, 0.0, None)
        )

        good_resid = resid[mask]
        med = float(np.nanmedian(good_resid))
        mad = float(np.nanmedian(np.abs(good_resid - med)))
        robust_sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(good_resid))
        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            robust_sigma = max(float(BKG_RMS_LARGE), 1.0)

        newmask = finite & (np.abs(resid - med) <= ROBUST_CLIP_SIGMA * robust_sigma)
        if np.array_equal(newmask, mask):
            break
        mask = newmask

    # Final solve using converged mask/weights.
    X = np.column_stack([
        trail_t[mask],
        star_t[mask],
        np.ones(np.count_nonzero(mask)),
        dx[mask],
        dy[mask],
    ])
    y = cut[mask]
    sig = sigma_img[mask]
    Xw = X / sig[:, None]
    yw = y / sig
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    source_model = beta[0] * trail_t + beta[1] * star_t
    model = source_model + beta[2] + beta[3] * dx + beta[4] * dy
    resid = cut - model
    chi2 = float(np.sum((resid[mask] / sigma_img[mask]) ** 2))
    dof = max(int(np.count_nonzero(mask)) - len(beta), 1)
    chi2_red = chi2 / dof

    normal = Xw.T @ Xw
    cov_beta = np.linalg.pinv(normal)
    # Never allow chi2_red < 1 to make formal errors artificially smaller.
    cov_beta *= max(chi2_red, 1.0)

    errs = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    denom = math.sqrt(max(cov_beta[0, 0] * cov_beta[1, 1], 0.0))
    corr_trail_star = float(cov_beta[0, 1] / denom) if denom > 0 else float("nan")

    return {
        "beta": beta.astype(float),
        "beta_err": errs.astype(float),
        "cov_beta": cov_beta.astype(float),
        "model": model.astype(float),
        "trail_component": (beta[0] * trail_t).astype(float),
        "star_component": (beta[1] * star_t).astype(float),
        "background_component": (beta[2] + beta[3] * dx + beta[4] * dy).astype(float),
        "residual": resid.astype(float),
        "fit_mask": mask,
        "sigma_image": sigma_img.astype(float),
        "chi2": chi2,
        "chi2_red": chi2_red,
        "dof": dof,
        "n_fit_pixels": int(np.count_nonzero(mask)),
        "corr_trail_star": corr_trail_star,
    }


def fit_joint_for_geometry(
    cut: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    star_template: np.ndarray,
    star_xy: np.ndarray,
    cov_psf: np.ndarray,
    theta_rad: float,
    perp_offset_pix: float,
) -> dict:
    trail_template = integrated_infinite_trail_template(
        xx=xx,
        yy=yy,
        anchor_xy=TRAIL_ANCHOR_XY,
        theta_rad=theta_rad,
        cov=cov_psf,
        perp_offset_pix=perp_offset_pix,
    )
    fit = _solve_weighted_linear(
        cut=cut,
        xx=xx,
        yy=yy,
        trail_t=trail_template,
        star_t=star_template,
        star_xy=star_xy,
    )
    fit["theta_rad"] = float(theta_rad)
    fit["theta_deg"] = float(math.degrees(theta_rad))
    fit["perp_offset_pix"] = float(perp_offset_pix)
    fit["trail_template"] = trail_template
    return fit


def grid_values(center: float, half_range: float, step: float) -> np.ndarray:
    n = int(round((2.0 * half_range) / step))
    return center + np.linspace(-half_range, half_range, n + 1)


def refine_geometry_grid(
    cut: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    star_template: np.ndarray,
    star_xy: np.ndarray,
    cov_psf: np.ndarray,
) -> dict:
    """Two-stage grid search in trail angle and perpendicular offset."""

    def search(theta_vals, offset_vals):
        best = None
        rows = []
        for th in theta_vals:
            for off in offset_vals:
                fit = fit_joint_for_geometry(
                    cut, xx, yy, star_template, star_xy, cov_psf, float(th), float(off)
                )
                # Penalize non-physical negative source amplitudes very strongly.
                trail_amp = float(fit["beta"][0])
                star_amp = float(fit["beta"][1])
                objective = float(fit["chi2"])
                if trail_amp < 0 or star_amp < 0:
                    objective += 1.0e9
                rows.append((float(th), float(off), objective, float(fit["chi2_red"])))
                if best is None or objective < best[0]:
                    best = (objective, fit)
        return best[1], np.asarray(rows, dtype=float)

    theta0_deg = math.degrees(AP_THETA_RAD)
    coarse_th_deg = grid_values(theta0_deg, COARSE_THETA_HALF_RANGE_DEG, COARSE_THETA_STEP_DEG)
    coarse_off = grid_values(0.0, COARSE_OFFSET_HALF_RANGE_PIX, COARSE_OFFSET_STEP_PIX)
    coarse_fit, coarse_grid = search(np.deg2rad(coarse_th_deg), coarse_off)

    th1_deg = float(coarse_fit["theta_deg"])
    off1 = float(coarse_fit["perp_offset_pix"])
    fine_th_deg = grid_values(th1_deg, REFINE_THETA_HALF_RANGE_DEG, REFINE_THETA_STEP_DEG)
    fine_off = grid_values(off1, REFINE_OFFSET_HALF_RANGE_PIX, REFINE_OFFSET_STEP_PIX)
    fine_fit, fine_grid = search(np.deg2rad(fine_th_deg), fine_off)

    fine_fit["coarse_grid"] = coarse_grid
    fine_fit["fine_grid"] = fine_grid
    return fine_fit


# ============================================================================
# Derived large-aperture quantities
# ============================================================================


def derive_large_aperture_quantities(
    fit: dict,
    star_xy: np.ndarray,
    cov_psf: np.ndarray,
    exptime: float,
    effective_width: float,
    assume_aperture_follows_fit_geometry: bool,
) -> dict:
    """Convert local joint-fit coefficients to the h=13 aperture domain.

    For fixed geometry this maps exactly to the reconstructed aperture model.
    For refined geometry it is explicitly a sensitivity diagnostic: we assume
    the aperture centreline followed the refined local trail geometry.
    """
    theta = float(fit["theta_rad"])
    off = float(fit["perp_offset_pix"])
    lam = float(fit["beta"][0])
    lam_err = float(fit["beta_err"][0])
    star_counts = float(fit["beta"][1])
    star_counts_err = float(fit["beta_err"][1])

    sigma_perp = sigma_perp_from_cov(cov_psf, theta)

    if assume_aperture_follows_fit_geometry:
        trail_mu_in_ap = 0.0
        # For the star, measure its offset from the fitted trail centreline.
        n = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
        star_mu = float((star_xy - TRAIL_ANCHOR_XY) @ n - off)
    else:
        # Primary fixed-geometry case: aperture and fitted trail are identical.
        trail_mu_in_ap = off
        n = np.array([-math.sin(AP_THETA_RAD), math.cos(AP_THETA_RAD)], dtype=float)
        star_mu = float((star_xy - TRAIL_ANCHOR_XY) @ n)

    trail_frac_h = gaussian_aperture_fraction_1d(trail_mu_in_ap, sigma_perp, AP_HEIGHT_PIX)
    star_frac_h = gaussian_aperture_fraction_1d(star_mu, sigma_perp, AP_HEIGHT_PIX)

    trail_counts_large = lam * effective_width * trail_frac_h
    trail_counts_large_err = lam_err * effective_width * trail_frac_h
    trail_rate_large = trail_counts_large / exptime
    trail_rate_large_err = trail_counts_large_err / exptime

    star_counts_in_large = star_counts * star_frac_h
    star_counts_in_large_err = star_counts_err * star_frac_h
    star_rate_in_large = star_counts_in_large / exptime
    star_rate_in_large_err = star_counts_in_large_err / exptime

    predicted_net = trail_counts_large + star_counts_in_large
    # This closure sigma is deliberately only the original aperture-statistical
    # error.  The local-fit prediction is correlated with the image and is not
    # treated as an independent measurement.
    closure_delta = predicted_net - NET_COUNTS_LARGE
    closure_sigma = closure_delta / NET_COUNTS_ERR_LARGE

    return {
        "sigma_perp_pix": sigma_perp,
        "trail_fraction_h13": trail_frac_h,
        "star_perp_offset_pix": star_mu,
        "star_fraction_h13": star_frac_h,
        "trail_linear_density_counts_per_pix": lam,
        "trail_linear_density_err": lam_err,
        "trail_counts_large": trail_counts_large,
        "trail_counts_large_err_formal": trail_counts_large_err,
        "trail_rate_large_rh": trail_rate_large,
        "trail_rate_large_rh_err_formal": trail_rate_large_err,
        "star_total_counts_fit": star_counts,
        "star_total_counts_fit_err": star_counts_err,
        "star_total_rate_fit": star_counts / exptime,
        "star_total_rate_fit_err": star_counts_err / exptime,
        "star_counts_in_large": star_counts_in_large,
        "star_counts_in_large_err_formal": star_counts_in_large_err,
        "star_rate_in_large": star_rate_in_large,
        "star_rate_in_large_err_formal": star_rate_in_large_err,
        "predicted_large_net_counts": predicted_net,
        "closure_delta_counts": closure_delta,
        "closure_delta_over_original_sigma": closure_sigma,
    }


def narrow_reference() -> dict:
    r6 = RH_NARROW * C1_NARROW
    err = math.sqrt(
        (C1_NARROW * RH_NARROW_ERR) ** 2
        + (RH_NARROW * C1_NARROW_ERR) ** 2
    )
    return {"r6": r6, "r6_err": err}


# ============================================================================
# Serialization / diagnostics
# ============================================================================


def serialisable_fit(fit: dict) -> dict:
    out = {}
    for k, v in fit.items():
        if isinstance(v, np.ndarray):
            if v.ndim <= 2 and v.size <= 36:
                out[k] = v.tolist()
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def save_diagnostic_png(cut: np.ndarray, fit: dict, bounds, path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x1, x2, y1, y2 = bounds
    extent = [x1 - 0.5, x2 - 0.5, y1 - 0.5, y2 - 0.5]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        (cut, "Data"),
        (fit["model"], "Joint model"),
        (fit["residual"], "Residual"),
        (fit["trail_component"], "Trail component"),
        (fit["star_component"], "Star component"),
        (fit["background_component"], "Background plane"),
    ]

    for ax, (arr, title) in zip(axes.flat, panels):
        im = ax.imshow(arr, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Transvaalia: joint trail + star + background fit")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    if not IMAGE_FITS.exists():
        raise FileNotFoundError(IMAGE_FITS)
    if not SRCLIST_FITS.exists():
        raise FileNotFoundError(SRCLIST_FITS)

    row = get_srclist_row(SRCLIST_FITS, SRCNUM)

    with fits.open(IMAGE_FITS, memmap=False) as hdul:
        image_hdu_idx, image_hdu = find_image_hdu(hdul)
        header = image_hdu.header.copy()
        # Some OM products can emit an unhelpful RuntimeWarning while casting
        # special/blank values to float; they are handled as non-finite below.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            data = np.array(image_hdu.data, dtype=np.float64, copy=True)

    exptime = float(header.get("EXPOSURE", EXPTIME_EXPECTED))
    if not np.isfinite(exptime) or exptime <= 0:
        raise RuntimeError("Invalid EXPOSURE in image")

    star_xy = source_xy_from_wcs(header, row)
    fwhm_maj = float(row["FWHM_MAJ"])
    fwhm_min = float(row["FWHM_MIN"])
    pa_deg = float(row["PA"])
    cov_psf = psf_covariance(fwhm_maj, fwhm_min, pa_deg)

    cut, xx, yy, bounds = make_cutout(data, star_xy, JOINT_CUTOUT_HALF_SIZE)
    star_template = integrated_star_template(
        xx, yy, star_xy, fwhm_maj, fwhm_min, pa_deg
    )

    effective_width = A_AP_EFF_LARGE / AP_HEIGHT_PIX
    rh_from_counts = NET_COUNTS_LARGE / exptime
    narrow = narrow_reference()

    # ------------------------------------------------------------------
    # C0: primary fixed-geometry joint fit.
    # ------------------------------------------------------------------
    fixed = fit_joint_for_geometry(
        cut=cut,
        xx=xx,
        yy=yy,
        star_template=star_template,
        star_xy=star_xy,
        cov_psf=cov_psf,
        theta_rad=AP_THETA_RAD,
        perp_offset_pix=0.0,
    )
    fixed_derived = derive_large_aperture_quantities(
        fixed,
        star_xy,
        cov_psf,
        exptime,
        effective_width,
        assume_aperture_follows_fit_geometry=False,
    )

    # ------------------------------------------------------------------
    # C1: geometry-refined sensitivity fit.
    # ------------------------------------------------------------------
    refined = None
    refined_derived = None
    if REFINE_GEOMETRY:
        refined = refine_geometry_grid(
            cut=cut,
            xx=xx,
            yy=yy,
            star_template=star_template,
            star_xy=star_xy,
            cov_psf=cov_psf,
        )
        refined_derived = derive_large_aperture_quantities(
            refined,
            star_xy,
            cov_psf,
            exptime,
            effective_width,
            assume_aperture_follows_fit_geometry=True,
        )

    # ------------------------------------------------------------------
    # Console report.
    # ------------------------------------------------------------------
    print("\n=== TRANSVAALIA JOINT DEBLEND ===")
    print(f"Image HDU                         : {image_hdu_idx}")
    print(f"EXPOSURE                          : {exptime:.6f} s")
    print(f"Large R_h copied                  : {RH_LARGE:.9f} +/- {RH_LARGE_ERR:.9f} ct/s")
    print(f"Large R_h from net_counts         : {rh_from_counts:.9f} ct/s")
    print(f"Effective longitudinal width      : {effective_width:.3f} px (= A_eff / h)")
    print(f"Narrow+C1 independent R_6         : {narrow['r6']:.9f} +/- {narrow['r6_err']:.9f} ct/s")

    print("\n--- SRCLIST star (shape/position only in joint fit) ---")
    print(f"SRCNUM / SRC_ID                   : {row['SRCNUM']} / {row.get('SRC_ID')}")
    print(f"WCS x,y                           : {star_xy[0]:.4f}, {star_xy[1]:.4f}")
    print(f"FWHM maj,min                      : {fwhm_maj:.4f}, {fwhm_min:.4f} px")
    print(f"PA                                : {pa_deg:.4f} deg")
    print(f"SRCLIST RATE diagnostic           : {float(row['RATE']):.9f} +/- {float(row['RATE_ERR']):.9f} ct/s")
    print(f"SRCLIST CORR_RATE (not used)      : {float(row.get('CORR_RATE', np.nan)):.9f} ct/s")

    def print_joint(label: str, fit: dict, der: dict):
        print(f"\n--- {label} ---")
        print(f"theta / offset                    : {fit['theta_deg']:.5f} deg / {fit['perp_offset_pix']:+.4f} px")
        print(f"chi2_red / Npix                   : {fit['chi2_red']:.3f} / {fit['n_fit_pixels']}")
        print(f"corr(trail, star)                 : {fit['corr_trail_star']:+.4f}")
        print(
            "trail density                      : "
            f"{der['trail_linear_density_counts_per_pix']:.6f} +/- "
            f"{der['trail_linear_density_err']:.6f} counts/pix"
        )
        print(
            "star total rate from joint fit      : "
            f"{der['star_total_rate_fit']:.6f} +/- {der['star_total_rate_fit_err']:.6f} ct/s"
        )
        print(
            "star contribution inside h=13       : "
            f"{der['star_rate_in_large']:.6f} +/- {der['star_rate_in_large_err_formal']:.6f} ct/s"
        )
        print(
            "ASTEROID R_h (deblended)             : "
            f"{der['trail_rate_large_rh']:.9f} +/- {der['trail_rate_large_rh_err_formal']:.9f} ct/s [formal fit]"
        )
        print(
            "closure: trail + star net counts     : "
            f"{der['predicted_large_net_counts']:.2f} vs {NET_COUNTS_LARGE:.2f} observed"
        )
        print(
            "closure residual                     : "
            f"{der['closure_delta_counts']:+.2f} counts "
            f"({der['closure_delta_over_original_sigma']:+.2f} x original net-count sigma)"
        )
        delta_narrow = der['trail_rate_large_rh'] - narrow['r6']
        sig_narrow = math.sqrt(der['trail_rate_large_rh_err_formal']**2 + narrow['r6_err']**2)
        print(
            "vs narrow+C1                         : "
            f"Delta={delta_narrow:+.6f} ct/s "
            f"({delta_narrow/sig_narrow:+.2f} sigma using formal errors)"
        )

    print_joint("Method C0: FIXED-GEOMETRY JOINT FIT [PRIMARY]", fixed, fixed_derived)
    if refined is not None and refined_derived is not None:
        print_joint("Method C1: REFINED-GEOMETRY JOINT FIT [SENSITIVITY]", refined, refined_derived)
        print(
            "geometry sensitivity in R_h          : "
            f"{refined_derived['trail_rate_large_rh'] - fixed_derived['trail_rate_large_rh']:+.6f} ct/s"
        )

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------
    npz_payload = {
        "cutout": cut,
        "x_grid": xx,
        "y_grid": yy,
        "star_template": star_template,
        "fixed_trail_template": fixed["trail_template"],
        "fixed_model": fixed["model"],
        "fixed_trail_component": fixed["trail_component"],
        "fixed_star_component": fixed["star_component"],
        "fixed_background_component": fixed["background_component"],
        "fixed_residual": fixed["residual"],
        "fixed_fit_mask": fixed["fit_mask"],
    }
    if refined is not None:
        npz_payload.update({
            "refined_trail_template": refined["trail_template"],
            "refined_model": refined["model"],
            "refined_trail_component": refined["trail_component"],
            "refined_star_component": refined["star_component"],
            "refined_background_component": refined["background_component"],
            "refined_residual": refined["residual"],
            "refined_fit_mask": refined["fit_mask"],
            "coarse_geometry_grid": refined["coarse_grid"],
            "fine_geometry_grid": refined["fine_grid"],
        })
    np.savez_compressed(OUT_NPZ, **npz_payload)

    result = {
        "case": {
            "target": "Transvaalia",
            "number": 715,
            "obsid": "0821871601",
            "filter": "L/UVW1",
            "image_fits": str(IMAGE_FITS),
            "srclist_fits": str(SRCLIST_FITS),
            "srcnum": int(SRCNUM),
        },
        "large_aperture_input": {
            "rh": RH_LARGE,
            "rh_err": RH_LARGE_ERR,
            "net_counts": NET_COUNTS_LARGE,
            "net_counts_err": NET_COUNTS_ERR_LARGE,
            "aper_counts": APER_COUNTS_LARGE,
            "bkg_mean": BKG_MEAN_LARGE,
            "bkg_rms": BKG_RMS_LARGE,
            "A_ap_eff": A_AP_EFF_LARGE,
            "A_bg_eff": A_BG_EFF_LARGE,
            "height_pix": AP_HEIGHT_PIX,
            "effective_width_pix_Aeff_over_h": effective_width,
            "theta_rad": AP_THETA_RAD,
            "theta_deg": math.degrees(AP_THETA_RAD),
            "trail_anchor_xy": TRAIL_ANCHOR_XY.tolist(),
            "exptime": exptime,
        },
        "narrow_c1_reference": {
            "rh_narrow": RH_NARROW,
            "rh_narrow_err": RH_NARROW_ERR,
            "c1": C1_NARROW,
            "c1_err": C1_NARROW_ERR,
            "r6": narrow["r6"],
            "r6_err": narrow["r6_err"],
        },
        "srclist": {
            "row0": int(row["_ROW0"]),
            "srcnum": int(row["SRCNUM"]),
            "src_id": int(row.get("SRC_ID", -1)),
            "ra_corr": float(row.get("RA_CORR", np.nan)),
            "dec_corr": float(row.get("DEC_CORR", np.nan)),
            "star_xy_wcs": star_xy.tolist(),
            "rate": float(row["RATE"]),
            "rate_err": float(row["RATE_ERR"]),
            "corr_rate": float(row.get("CORR_RATE", np.nan)),
            "fwhm_maj": fwhm_maj,
            "fwhm_min": fwhm_min,
            "pa_deg": pa_deg,
            "qflag": int(row.get("QFLAG", 0)),
            "cflag": int(row.get("CFLAG", 0)),
            "eflag": int(row.get("EFLAG", 0)),
        },
        "joint_fit_fixed_primary": {
            "fit": serialisable_fit(fixed),
            "derived": fixed_derived,
        },
        "joint_fit_refined_sensitivity": None,
        "method_notes": [
            "The joint fit does not use SRCLIST RATE as a prior or subtraction.",
            "The star shape/centroid are taken from SRCLIST FWHM_MAJ/FWHM_MIN/PA and RA_CORR/DEC_CORR.",
            "The asteroid template is a normalized Gaussian PSF convolved with a locally uniform straight line.",
            "The fitted trail coefficient is counts per pixel of trail length; A_ap_eff/h reconstructs the historical large-box effective length.",
            "The fixed-geometry fit is primary because it maps directly to the historical h=13 aperture geometry.",
            "The refined-geometry fit is a sensitivity diagnostic for the rounded/uncertain stored trail angle and centreline.",
            "Formal joint-fit errors are local model-fit errors only.  Do not replace the pipeline R_h uncertainty with them without an explicit uncertainty decision.",
            "The closure comparison is diagnostic, not an independent chi-square constraint, because both quantities come from the same image.",
            "After selecting a deblended R_h, run the normal CoI -> C2 -> TDS -> m_AB chain from that rate.",
        ],
    }
    if refined is not None and refined_derived is not None:
        result["joint_fit_refined_sensitivity"] = {
            "fit": serialisable_fit(refined),
            "derived": refined_derived,
            "delta_rh_vs_fixed": float(
                refined_derived["trail_rate_large_rh"] - fixed_derived["trail_rate_large_rh"]
            ),
        }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    png_ok = save_diagnostic_png(cut, fixed, bounds, OUT_PNG)

    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_NPZ}")
    if png_ok:
        print(f"Saved: {OUT_PNG}")
    else:
        print("Diagnostic PNG skipped (matplotlib unavailable).")

    print("\nInterpretation rule:")
    print("  Use the joint-fit R_h only after checking:")
    print("    (1) residual image has no coherent star/trail structure,")
    print("    (2) |corr(trail,star)| is not near 1,")
    print("    (3) closure is reasonable, and")
    print("    (4) fixed/refined geometry and narrow+C1 solutions are mutually consistent.")


if __name__ == "__main__":
    main()
