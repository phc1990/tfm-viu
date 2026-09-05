#!/usr/bin/env python3
"""
Transvaalia / OBSID 0821871601 / UVW1 deblending diagnostic.

Purpose
-------
Estimate and remove the contribution of the neighbouring OM point source
(SRCLIST SRCNUM=56) from the *large* h=13 px asteroid trail aperture.

Two normalisations are reported:

1) catalog_rate
   Reconstruct an elliptical Gaussian core from FWHM_MAJ/FWHM_MIN/PA and
   normalise it with the raw SRCLIST RATE. This is the direct implementation
   of the catalogue-anchored subtraction.

2) clean_pixel_fit  [recommended diagnostic]
   Use the same SRCLIST position/shape, but fit the star amplitude directly
   from pixels on the side of the source that is not occupied by the asteroid
   trail. This avoids assuming that the SRCLIST RATE itself is uncontaminated
   by the asteroid.

The script intentionally stops at the deblended raw/background-subtracted
trail rate R_h. For this h=13 px extraction C1=1, so this is also the R_6 input
to the normal calibration chain. Re-run CoI -> C2 -> TDS after deblending;
do not subtract the star from the already-corrected final rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


# =====================================================================
# USER / CASE CONFIGURATION
# =====================================================================

IMAGE_FITS = Path(
    "/Users/eracero/workspace/tfm-viu/temp/0821871601/L/"
    "P0821871601OMS006FSIMAGL000.FTZ"
)
SRCLIST_FITS = Path(
    "/Users/eracero/workspace/tfm-viu/temp/0821871601/L/"
    "P0821871601OMS006SWSRLIL000.FTZ"
)

# Identify the contaminating source robustly by catalogue source number.
# The same row also has SRC_ID=160 in the file you inspected.
SRCNUM = 56

# ---- Large-box Transvaalia measurement copied from the pipeline output ----
RH_LARGE = 1.5185142560758396             # ct/s, raw background-subtracted R_h
RH_LARGE_ERR = 0.038237590348100675       # ct/s
NET_COUNTS_LARGE = 7592.571280379198      # counts
NET_COUNTS_ERR_LARGE = 191.18795174050337 # counts
APER_COUNTS_LARGE = 28525.892600506544    # counts
BKG_MEAN_LARGE = 21.1337826786801         # counts/pix
BKG_RMS_LARGE = 3.151241630204799         # counts/pix
A_AP_EFF_LARGE = 990.5146484375            # pix
A_BG_EFF_LARGE = 1213.7666015625           # pix
EXPTIME_EXPECTED = 5000.0                  # s

# Large aperture geometry.
# h=13 px is the standard 6"-equivalent box used in this extraction.
AP_HEIGHT_PIX = 13.0
TRAIL_THETA_RAD = -0.1461

# A point on the asteroid centreline from the UI log.  Only the centreline
# is needed for the perpendicular PSF fraction.  Use the same pixel convention
# as astropy WCS / matplotlib / photutils (0-based pixel coordinates).
TRAIL_ANCHOR_XY = np.array([453.74, 1121.63], dtype=float)

# Because the large box was deliberately drawn long enough to contain the
# whole neighbouring star longitudinally, the code treats the aperture as
# effectively infinite along the trail direction.  This is safe here because
# width ~ A_eff/h ~ 76 px whereas the PSF sigma is ~1 px.
ASSUME_FULL_LONGITUDINAL_ENCLOSURE = True

# Pixel-fit settings.  We fit star amplitude + local background plane while
# masking the asteroid trail core.  Running several mask widths is a useful
# sensitivity test.
CUTOUT_HALF_SIZE = 12  # pixels around star
TRAIL_MASK_HALF_WIDTHS = (2.0, 2.5, 3.0)  # px

# Monte-Carlo propagation of PSF-shape/centroid uncertainty into aperture fraction.
N_MC = 10000
RNG_SEED = 1871601

# Output products
OUT_JSON = Path("transvaalia_deblend_result.json")
OUT_DIAGNOSTIC_NPZ = Path("transvaalia_deblend_diagnostic.npz")


# =====================================================================
# HELPERS
# =====================================================================

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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


def get_srclist_row(srclist: Path, srcnum: int):
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
        row = {name: tab[name][j].item() if hasattr(tab[name][j], "item") else tab[name][j] for name in names}
        row["_HDU"] = ihdu
        row["_ROW0"] = j
        return row


def source_xy_from_wcs(image_hdu, row: dict) -> tuple[float, float]:
    ra = float(row.get("RA_CORR", row.get("RA")))
    dec = float(row.get("DEC_CORR", row.get("DEC")))
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    wcs = WCS(image_hdu.header)
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def psf_covariance(fwhm_maj: float, fwhm_min: float, pa_deg: float) -> np.ndarray:
    """
    Elliptical Gaussian covariance in image x/y pixels.

    SAS PA is measured anti-clockwise from +x for the source ellipse.
    """
    smaj = float(fwhm_maj) * FWHM_TO_SIGMA
    smin = float(fwhm_min) * FWHM_TO_SIGMA
    pa = math.radians(float(pa_deg))

    umaj = np.array([math.cos(pa), math.sin(pa)], dtype=float)
    umin = np.array([-math.sin(pa), math.cos(pa)], dtype=float)
    return (smaj ** 2) * np.outer(umaj, umaj) + (smin ** 2) * np.outer(umin, umin)


def aperture_fraction_infinite_length(
    star_xy: np.ndarray,
    trail_anchor_xy: np.ndarray,
    trail_theta_rad: float,
    aperture_height_pix: float,
    cov_xy: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fraction of a 2-D Gaussian inside an infinitely long trail rectangle.

    The integral collapses exactly to the 1-D Gaussian projected onto the
    direction perpendicular to the trail.

    Returns
    -------
    fraction, signed_perpendicular_offset_pix, sigma_perpendicular_pix
    """
    theta = float(trail_theta_rad)
    normal = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

    delta = np.asarray(star_xy, dtype=float) - np.asarray(trail_anchor_xy, dtype=float)
    mu_perp = float(delta @ normal)
    sigma_perp = float(math.sqrt(normal @ cov_xy @ normal))

    half_h = 0.5 * float(aperture_height_pix)
    z_hi = (half_h - mu_perp) / sigma_perp
    z_lo = (-half_h - mu_perp) / sigma_perp
    frac = phi(z_hi) - phi(z_lo)
    return float(frac), mu_perp, sigma_perp


def mc_fraction(row: dict, star_xy: np.ndarray) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)

    fmaj0 = float(row["FWHM_MAJ"])
    fmin0 = float(row["FWHM_MIN"])
    pa0 = float(row["PA"])
    poserr = abs(float(row.get("POSERR", 0.0)))
    emaj = abs(float(row.get("FWHM_MAJ_ERR", 0.0)))
    emin = abs(float(row.get("FWHM_MIN_ERR", 0.0)))
    epa = abs(float(row.get("PA_ERR", 0.0)))

    vals = []
    for _ in range(int(N_MC)):
        fmaj = rng.normal(fmaj0, emaj) if emaj > 0 else fmaj0
        fmin = rng.normal(fmin0, emin) if emin > 0 else fmin0
        if fmaj <= 0 or fmin <= 0:
            continue

        pa = rng.normal(pa0, epa) if epa > 0 else pa0
        xy = np.asarray(star_xy, dtype=float).copy()
        if poserr > 0:
            xy += rng.normal(0.0, poserr, size=2)

        cov = psf_covariance(fmaj, fmin, pa)
        frac, _, _ = aperture_fraction_infinite_length(
            xy,
            TRAIL_ANCHOR_XY,
            TRAIL_THETA_RAD,
            AP_HEIGHT_PIX,
            cov,
        )
        if np.isfinite(frac):
            vals.append(frac)

    arr = np.asarray(vals, dtype=float)
    if arr.size < 10:
        return float("nan"), float("nan"), arr
    return float(np.median(arr)), float(np.std(arr, ddof=1)), arr


def gaussian_psf_at_pixel_centres(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    fwhm_maj: float,
    fwhm_min: float,
    pa_deg: float,
) -> np.ndarray:
    """
    Continuous elliptical Gaussian evaluated at pixel centres, in pix^-2.

    For this PSF width (~1 pix sigma) the pixel-centre approximation is good
    enough for the amplitude diagnostic; the aperture fraction itself is
    computed analytically above.
    """
    smaj = float(fwhm_maj) * FWHM_TO_SIGMA
    smin = float(fwhm_min) * FWHM_TO_SIGMA
    pa = math.radians(float(pa_deg))

    dx = xx - float(x0)
    dy = yy - float(y0)

    xp = math.cos(pa) * dx + math.sin(pa) * dy
    yp = -math.sin(pa) * dx + math.cos(pa) * dy

    norm = 1.0 / (2.0 * math.pi * smaj * smin)
    return norm * np.exp(-0.5 * ((xp / smaj) ** 2 + (yp / smin) ** 2))


def fit_star_from_clean_pixels(
    data: np.ndarray,
    star_xy: np.ndarray,
    row: dict,
    trail_mask_half_width: float,
) -> dict:
    """
    Fit star amplitude + local background plane, masking the asteroid trail.

    Model: data = A * PSF + b0 + bx*dx + by*dy
    A is the fitted Gaussian-core counts for the point source.
    """
    x0, y0 = map(float, star_xy)
    ny, nx = data.shape

    x1 = max(0, int(math.floor(x0)) - CUTOUT_HALF_SIZE)
    x2 = min(nx, int(math.floor(x0)) + CUTOUT_HALF_SIZE + 1)
    y1 = max(0, int(math.floor(y0)) - CUTOUT_HALF_SIZE)
    y2 = min(ny, int(math.floor(y0)) + CUTOUT_HALF_SIZE + 1)

    cut = np.asarray(data[y1:y2, x1:x2], dtype=float)
    yy, xx = np.mgrid[y1:y2, x1:x2]

    psf = gaussian_psf_at_pixel_centres(
        xx,
        yy,
        x0,
        y0,
        float(row["FWHM_MAJ"]),
        float(row["FWHM_MIN"]),
        float(row["PA"]),
    )

    # Signed distance to asteroid centreline.
    normal = np.array([-math.sin(TRAIL_THETA_RAD), math.cos(TRAIL_THETA_RAD)])
    perp = (xx - TRAIL_ANCHOR_XY[0]) * normal[0] + (yy - TRAIL_ANCHOR_XY[1]) * normal[1]

    finite = np.isfinite(cut)
    clean = np.abs(perp) >= float(trail_mask_half_width)

    # Avoid fitting extreme pixels/artifacts with a very simple robust guard.
    valid = finite & clean
    if np.count_nonzero(valid) < 20:
        raise RuntimeError("Too few clean pixels for star fit")

    dx = xx - x0
    dy = yy - y0

    X = np.column_stack([
        psf[valid],
        np.ones(np.count_nonzero(valid)),
        dx[valid],
        dy[valid],
    ])
    y = cut[valid]

    # Approximate Poisson weighting in counts domain.
    sigma = np.sqrt(np.clip(y, 1.0, None))
    Xw = X / sigma[:, None]
    yw = y / sigma

    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    amp, b0, bx, by = [float(v) for v in beta]

    model_valid = X @ beta
    resid = y - model_valid
    chi2 = float(np.sum((resid / sigma) ** 2))
    dof = max(len(y) - X.shape[1], 1)
    chi2_red = chi2 / dof

    normal_matrix = Xw.T @ Xw
    cov_beta = np.linalg.pinv(normal_matrix)
    # Do not let an accidentally small reduced chi2 shrink formal errors.
    cov_beta *= max(chi2_red, 1.0)
    amp_err = float(math.sqrt(max(cov_beta[0, 0], 0.0)))

    model_full = amp * psf + b0 + bx * dx + by * dy
    resid_full = cut - model_full

    return {
        "trail_mask_half_width": float(trail_mask_half_width),
        "amplitude_counts": amp,
        "amplitude_counts_err": amp_err,
        "background_b0": b0,
        "background_bx": bx,
        "background_by": by,
        "chi2_red": chi2_red,
        "n_fit_pixels": int(np.count_nonzero(valid)),
        "cutout_bounds": [x1, x2, y1, y2],
        "psf_sum_in_cutout": float(np.sum(psf)),
        "cutout": cut,
        "psf": psf,
        "model": model_full,
        "residual": resid_full,
        "fit_mask": valid,
        "perp": perp,
    }


def propagated_deblend_error(
    rh_err: float,
    star_rate: float,
    star_rate_err: float,
    frac: float,
    frac_err: float,
) -> float:
    star_in_ap_err = math.sqrt((frac * star_rate_err) ** 2 + (star_rate * frac_err) ** 2)
    return math.sqrt(rh_err ** 2 + star_in_ap_err ** 2)


def serialisable_fit(fit: dict) -> dict:
    return {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    if not IMAGE_FITS.exists():
        raise FileNotFoundError(IMAGE_FITS)
    if not SRCLIST_FITS.exists():
        raise FileNotFoundError(SRCLIST_FITS)

    row = get_srclist_row(SRCLIST_FITS, SRCNUM)

    with fits.open(IMAGE_FITS, memmap=False) as hdul:
        image_hdu_idx, image_hdu = find_image_hdu(hdul)
        data = np.asarray(image_hdu.data, dtype=float)
        header = image_hdu.header.copy()

    exptime = float(header.get("EXPOSURE", EXPTIME_EXPECTED))
    if not np.isfinite(exptime) or exptime <= 0:
        raise RuntimeError("Invalid EXPOSURE in image")

    star_xy = np.array(source_xy_from_wcs(image_hdu, row), dtype=float)

    fwhm_maj = float(row["FWHM_MAJ"])
    fwhm_min = float(row["FWHM_MIN"])
    pa_deg = float(row["PA"])
    cov = psf_covariance(fwhm_maj, fwhm_min, pa_deg)

    frac, dperp, sigma_perp = aperture_fraction_infinite_length(
        star_xy,
        TRAIL_ANCHOR_XY,
        TRAIL_THETA_RAD,
        AP_HEIGHT_PIX,
        cov,
    )

    frac_mc_med, frac_mc_err, frac_samples = mc_fraction(row, star_xy)
    # Preserve the nominal fraction as central value; MC scatter is its uncertainty.
    if not np.isfinite(frac_mc_err):
        frac_mc_err = 0.0

    rate_cat = float(row["RATE"])
    rate_cat_err = float(row["RATE_ERR"])
    rate_corr = float(row.get("CORR_RATE", np.nan))

    star_in_ap_cat = rate_cat * frac
    star_in_ap_cat_err = math.sqrt((frac * rate_cat_err) ** 2 + (rate_cat * frac_mc_err) ** 2)

    rh_cat = RH_LARGE - star_in_ap_cat
    rh_cat_err = propagated_deblend_error(
        RH_LARGE_ERR,
        rate_cat,
        rate_cat_err,
        frac,
        frac_mc_err,
    )

    # Internal consistency of the copied large-box values.
    rh_from_counts = NET_COUNTS_LARGE / exptime
    width_est = A_AP_EFF_LARGE / AP_HEIGHT_PIX

    print("\n=== TRANSVAALIA DEBLEND ===")
    print(f"Image HDU                   : {image_hdu_idx}")
    print(f"EXPOSURE                    : {exptime:.6f} s")
    print(f"R_h copied                  : {RH_LARGE:.9f} +/- {RH_LARGE_ERR:.9f} ct/s")
    print(f"R_h from net_counts         : {rh_from_counts:.9f} ct/s")
    print(f"Approx. large-box width     : {width_est:.3f} px  (= A_eff/h)")

    print("\n--- SRCLIST contaminant ---")
    print(f"SRCNUM / SRC_ID             : {row['SRCNUM']} / {row.get('SRC_ID')}")
    print(f"RA_CORR, DEC_CORR           : {row.get('RA_CORR')}, {row.get('DEC_CORR')}")
    print(f"WCS x,y in FSIMAG           : {star_xy[0]:.4f}, {star_xy[1]:.4f}")
    print(f"RATE raw                    : {rate_cat:.9f} +/- {rate_cat_err:.9f} ct/s")
    print(f"CORR_RATE (NOT subtracted)  : {rate_corr:.9f} ct/s")
    print(f"FWHM maj,min                : {fwhm_maj:.4f}, {fwhm_min:.4f} px")
    print(f"PA                          : {pa_deg:.4f} deg")

    print("\n--- Gaussian core / large aperture ---")
    print(f"signed d_perp               : {dperp:.4f} px")
    print(f"sigma_perp                  : {sigma_perp:.4f} px")
    print(f"PSF fraction in h=13 box    : {frac:.6f}")
    print(f"MC fraction median/scatter  : {frac_mc_med:.6f} +/- {frac_mc_err:.6f}")

    print("\n--- Method A: SRCLIST RATE normalisation ---")
    print(f"star rate inside aperture   : {star_in_ap_cat:.9f} +/- {star_in_ap_cat_err:.9f} ct/s")
    print(f"R_h deblended               : {rh_cat:.9f} +/- {rh_cat_err:.9f} ct/s")
    print(f"deblended net counts        : {rh_cat * exptime:.3f} counts")

    # ---------------------------------------------------------------
    # Method B: fit star amplitude from clean image pixels.
    # ---------------------------------------------------------------
    fits_out = []
    print("\n--- Method B: clean-pixel PSF amplitude fits ---")
    for halfw in TRAIL_MASK_HALF_WIDTHS:
        fit = fit_star_from_clean_pixels(data, star_xy, row, halfw)
        star_rate_fit = fit["amplitude_counts"] / exptime
        star_rate_fit_err = fit["amplitude_counts_err"] / exptime
        star_in_ap_fit = star_rate_fit * frac
        star_in_ap_fit_err = math.sqrt(
            (frac * star_rate_fit_err) ** 2 + (star_rate_fit * frac_mc_err) ** 2
        )
        rh_fit = RH_LARGE - star_in_ap_fit
        rh_fit_err = propagated_deblend_error(
            RH_LARGE_ERR,
            star_rate_fit,
            star_rate_fit_err,
            frac,
            frac_mc_err,
        )

        fit.update({
            "star_rate_fit": float(star_rate_fit),
            "star_rate_fit_err": float(star_rate_fit_err),
            "star_rate_in_ap": float(star_in_ap_fit),
            "star_rate_in_ap_err": float(star_in_ap_fit_err),
            "rh_deblended": float(rh_fit),
            "rh_deblended_err": float(rh_fit_err),
        })
        fits_out.append(fit)

        print(
            f"mask +/-{halfw:.1f}px: "
            f"Rstar={star_rate_fit:.6f}+/-{star_rate_fit_err:.6f}, "
            f"R_h={rh_fit:.6f}+/-{rh_fit_err:.6f} ct/s, "
            f"chi2_red={fit['chi2_red']:.3f}"
        )

    # Save arrays for later visual inspection/replotting without rerunning the fit.
    # Use the central mask result for the array diagnostic.
    mid = fits_out[len(fits_out) // 2]
    np.savez_compressed(
        OUT_DIAGNOSTIC_NPZ,
        cutout=mid["cutout"],
        psf=mid["psf"],
        model=mid["model"],
        residual=mid["residual"],
        fit_mask=mid["fit_mask"],
        perpendicular_coordinate=mid["perp"],
        fraction_mc_samples=frac_samples,
    )

    result = {
        "case": {
            "target": "Transvaalia",
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
            "width_pix_estimated_from_Aeff_over_h": width_est,
            "theta_rad": TRAIL_THETA_RAD,
            "trail_anchor_xy": TRAIL_ANCHOR_XY.tolist(),
            "exptime": exptime,
        },
        "srclist": {
            "row0": int(row["_ROW0"]),
            "srcnum": int(row["SRCNUM"]),
            "src_id": int(row.get("SRC_ID", -1)),
            "rate": rate_cat,
            "rate_err": rate_cat_err,
            "corr_rate": rate_corr,
            "fwhm_maj": fwhm_maj,
            "fwhm_min": fwhm_min,
            "pa_deg": pa_deg,
            "star_xy_wcs": star_xy.tolist(),
        },
        "psf_fraction": {
            "nominal": frac,
            "mc_median": frac_mc_med,
            "mc_sigma": frac_mc_err,
            "dperp_pix": dperp,
            "sigma_perp_pix": sigma_perp,
            "assume_full_longitudinal_enclosure": ASSUME_FULL_LONGITUDINAL_ENCLOSURE,
        },
        "method_A_srclist_rate": {
            "star_rate_in_ap": star_in_ap_cat,
            "star_rate_in_ap_err": star_in_ap_cat_err,
            "rh_deblended": rh_cat,
            "rh_deblended_err": rh_cat_err,
        },
        "method_B_clean_pixel_fit": [serialisable_fit(f) for f in fits_out],
        "notes": [
            "Use raw SRCLIST RATE, not CORR_RATE, because R_h is a raw/background-subtracted rate before CoI/C2/TDS.",
            "Method A is diagnostic because the SRCLIST RATE may itself contain asteroid-trail flux when source and trail overlap.",
            "Method B avoids that circularity by fitting the star amplitude from pixels outside the masked trail core.",
            "After choosing a deblended R_h, rerun the normal nonlinear coincidence-loss step and subsequent C2/TDS calibration.",
        ],
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_DIAGNOSTIC_NPZ}")


if __name__ == "__main__":
    main()
