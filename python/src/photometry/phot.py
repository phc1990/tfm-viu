# -*- coding: utf-8 -*-
"""
phot.py — trail photometry core with units-aware handling + uncertainty
- RATE images (counts/s/pix): compute c and σ_c directly from region stats.
- COUNTS images: derive EXPTIME robustly, then compute c and σ_c from counts.
- If units ambiguous and ZP is from OBSMLI, default to RATE mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import math
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# photutils (new API first; fallback)
try:
    from photutils.aperture import (
        aperture_photometry,
        RectangularAperture,
        RectangularAnnulus,
    )
except Exception:
    from photutils import (
        aperture_photometry,
        RectangularAperture,
        RectangularAnnulus,
    )

AB_PROP_COEFF = 1.085736  # 2.5/ln(10)


@dataclass
class PhotometryResult:
    # Magnitudes and uncertainties
    mag_ab: Optional[float] = None
    mag_err: Optional[float] = None

    # Aperture-correction term in magnitudes (applied additively to mag_ab)
    # apcorr_mag = -2.5*log10(f_apcorr); if f_apcorr>1 (flux increases), apcorr_mag is negative.
    apcorr_mag: Optional[float] = None
    apcorr_mag_err: Optional[float] = None
    mag_ab_apcorr: Optional[float] = None
    mag_ab_apcorr_err: Optional[float] = None


    # Rates and uncertainties
    count_rate: Optional[float] = None
    count_rate_err: Optional[float] = None

    # Counts (filled for COUNTS images; N/A for mosaics unless derivable)
    net_counts: Optional[float] = None
    net_counts_err: Optional[float] = None
    aper_counts: Optional[float] = None  # Cap (counts or counts/s depending on unit)

    # Background / geometry
    bkg_per_pix: Optional[float] = None            # mean background per pixel (in image units)
    bkg_rms_per_pix: Optional[float] = None        # rms (std) per pixel (in image units)
    A_ap_eff: Optional[float] = None               # effective aperture pixels (weights sum)
    A_bg_eff: Optional[float] = None               # effective bg pixels (weights sum)

    # Calibration provenance
    zp_ab: Optional[float] = None
    zp_keyword: Optional[str] = None
    zp_source_file: Optional[str] = None
    zp_source_kind: Optional[str] = None
@dataclass
class RatePhotometryResult:
    # Net count-rate (ct/s) and uncertainty
    net_rate: float
    net_rate_err: Optional[float]

    # Total in aperture (source+background), in native units (counts or counts/s)
    aper_sum: float

    # Background stats per pixel, in native units (counts/pix or (counts/s)/pix)
    bkg_per_pix: float
    bkg_rms_per_pix: float

    # Effective areas (weights sum)
    A_ap_eff: float
    A_bg_eff: Optional[float]

    # Unit bookkeeping
    unit_kind: str                # "rate" or "counts"
    exptime: Optional[float]      # seconds if counts, else None
    bunit_str: str


class PhotTable:
    def __init__(self, hduw) -> None:
        self.hduw = hduw
        self.zero_point: Optional[float] = None
        self._zp_prov: Dict[str, Optional[str]] = {"keyword": None, "file": None, "kind": None}

    # ----------------- helpers -----------------

    @staticmethod
    def _filter_to_band(filter_code: str) -> str:
        f = (filter_code or "").strip().upper()
        return {"L": "UVW1", "M": "UVM2", "S": "UVW2", "V": "V", "B": "B", "U": "U"}.get(f, f)

    def _exptime_from(self) -> Optional[float]:
        """
        Robust exposure discovery across all HDUs.

        Strategy:
        1) Use attributes on the HDUW wrapper (texp / exptime) if present.
        2) Look for common exposure keywords in the header of the wrapped HDU.
        3) If the wrapper exposes a 'file' attribute, optionally open the FITS
           and look in all HDUs + derive from MJD-END/MJD-OBS or DATE-END/DATE-OBS.
        """
        # 1) Direct attributes on the wrapper (HDUW)
        for attr in ("exptime", "texp", "exposure_time"):
            try:
                if hasattr(self.hduw, attr):
                    val = getattr(self.hduw, attr)
                    if val is not None:
                        val = float(val)
                        if np.isfinite(val) and val > 0:
                            # print(f"[PHOT] Using HDUW.{attr}={val} s")
                            return val
            except Exception:
                pass

        # 2) Look into the header of the wrapped HDU first
        KEY_CANDIDATES = (
            "EXPTIME",
            "EXPOSURE",
            "ONTIME",
            "EXP_TIME",
            "EXPOSURE_TIME",
            "TELAPSE",
        )

        hdr = None
        try:
            h = getattr(self.hduw, "hdu", None)
            hdr = getattr(h, "header", None)
        except Exception:
            hdr = None

        if hdr is not None:
            for k in KEY_CANDIDATES:
                if k in hdr and hdr[k] is not None:
                    try:
                        val = float(hdr[k])
                        if np.isfinite(val) and val > 0:
                            # print(f"[PHOT] Using {k}={val} s from HDUW.hdu.header")
                            return val
                    except Exception:
                        continue

        # 3) If the wrapper knows the file path, fall back to full-file search + MJD/DATE
        fpath = getattr(self.hduw, "file", None)
        if not fpath:
            return None  # nothing more we can do

        try:
            from astropy.time import Time
            with fits.open(str(fpath), memmap=False) as hdul:
                # 3a) Search all HDUs for exposure keywords
                for h in hdul:
                    hdr = getattr(h, "header", None)
                    if not hdr:
                        continue
                    for k in KEY_CANDIDATES:
                        if k in hdr and hdr[k] is not None:
                            try:
                                val = float(hdr[k])
                                if np.isfinite(val) and val > 0:
                                    # print(f"[PHOT] Using {k}={val} s from file={fpath}")
                                    return val
                            except Exception:
                                continue

                # 3b) Derive from MJD-END / MJD-OBS or DATE-END / DATE-OBS in primary header
                prim = hdul[0].header if len(hdul) > 0 else None
                if prim is not None:
                    if all(k in prim and prim[k] is not None for k in ("MJD-END", "MJD-OBS")):
                        try:
                            dt = (float(prim["MJD-END"]) - float(prim["MJD-OBS"])) * 86400.0
                            if np.isfinite(dt) and dt > 0:
                                # print(f"[PHOT] Using MJD-END/MJD-OBS → {dt} s")
                                return dt
                        except Exception:
                            pass
                    if all(k in prim and prim[k] for k in ("DATE-END", "DATE-OBS")):
                        try:
                            t_end = Time(str(prim["DATE-END"]), format="isot", scale="utc")
                            t_obs = Time(str(prim["DATE-OBS"]), format="isot", scale="utc")
                            dt = (t_end - t_obs).sec
                            if np.isfinite(dt) and dt > 0:
                                # print(f"[PHOT] Using DATE-END/DATE-OBS → {dt} s")
                                return float(dt)
                        except Exception:
                            pass
        except Exception:
            pass

        return None


    def _effective_area_from_mask(self, ap) -> float:
        mask = ap.to_mask(method='exact')
        if isinstance(mask, (list, tuple)):
            area = 0.0
            for m in mask:
                data = m.data
                if data is not None:
                    area += np.nansum(data)
            return float(area)
        data = mask.data
        return float(np.nansum(data)) if data is not None else float(ap.area)

    def _effective_area_bg_from_mask(self, ann: Optional[RectangularAnnulus]) -> Optional[float]:
        if ann is None:
            return None
        mask = ann.to_mask(method='exact')
        if isinstance(mask, (list, tuple)):
            area = 0.0
            for m in mask:
                data = m.data
                if data is not None:
                    area += np.nansum(data)
            return float(area)
        data = mask.data
        return float(np.nansum(data)) if data is not None else None

    def _bkg_stats_from_annulus(self, data: np.ndarray, ann: Optional[RectangularAnnulus]) -> Tuple[float, float]:
        """
        Mean and std in the annulus footprint; if not enough pixels, use global sigma-clipped stats.
        Returns (mean, std) in the same units as `data` (counts or counts/s).
        """
        if ann is not None:
            mask = ann.to_mask(method='exact')
            vals: List[np.ndarray] = []
            if isinstance(mask, (list, tuple)):
                for m in mask:
                    cut = m.cutout(data)
                    w = m.data
                    if cut is not None and w is not None:
                        vals.append(cut[w > 0])
                if vals:
                    ann_values = np.concatenate(vals)
                else:
                    ann_values = np.array([])
            else:
                cut = mask.cutout(data)
                w = mask.data
                ann_values = cut[w > 0] if (cut is not None and w is not None) else np.array([])

            finite = np.isfinite(ann_values)
            if ann_values.size > 20 and finite.sum() >= 20:
                mean, med, std = sigma_clipped_stats(ann_values[finite], sigma=3.0, maxiters=5)
                return float(mean), float(std)

        # Fallback: whole image (robust)
        mean, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        return float(mean), float(std)

    def _data_unit_kind_and_bunit(self) -> (str, Optional[str]):
        """
        Inspect headers to decide if image is 'rate' or 'counts'.
        Returns ('rate'|'counts'|'unknown', bunit_string_or_none)
        """
        bunit_val: Optional[str] = None
        try:
            with fits.open(str(self.hduw.file), memmap=False) as hdul:
                cand_hdrs = []
                if len(hdul) > 1 and hasattr(hdul[1], "header"):
                    cand_hdrs.append(hdul[1].header)
                if hasattr(hdul[0], "header"):
                    cand_hdrs.append(hdul[0].header)
                cand_hdrs += [h.header for h in hdul[2:] if hasattr(h, "header")]

                for hdr in cand_hdrs:
                    for key in ("BUNIT", "BUNIT1", "BUNIT2"):
                        if key in hdr and hdr[key]:
                            bunit_val = str(hdr[key]).strip()
                            break
                    if bunit_val:
                        break
        except Exception:
            pass

        if bunit_val:
            u = bunit_val.lower().replace(" ", "")
            # common variants for rates
            if any(token in u for token in ("count/s", "counts/s", "ct/s", "cnt/s", "s^-1", "s-1")):
                return "rate", bunit_val
            # common variants for counts
            if "count" in u or "counts" in u or "cnt" in u:
                return "counts", bunit_val

        return "unknown", bunit_val

    # ----------------- calibration -----------------

    def calibrate_against_source_list(
        self,
        source_list_file: str,
        filter: str,
        source_list_kind: Optional[str] = None,
    ) -> None:
        fpath = Path(source_list_file).expanduser().resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"SRCLIST file not found: {fpath}")

        filt = (filter or "").strip().upper()
        band = self._filter_to_band(filt)

        with fits.open(str(fpath), memmap=False) as hdul:
            if (source_list_kind or "").upper() == "OBSMLI" and len(hdul) > 1:
                hdr = hdul[1].header
                key = f"ABM0{band}"
                if key in hdr and hdr[key] is not None:
                    self.zero_point = float(hdr[key])
                    self._zp_prov.update(keyword=key, file=str(fpath), kind="OBSMLI")
                    print(f"[PHOT] ZP={self.zero_point:.6f} from {fpath.name} key={key} [OBSMLI]")
                    return

            common_keys = [
                "ABMAGZP", "MAGZPT", "PHOTZP", "MAGZERO", "ZEROPOINT", "OMZP",
                f"ABMAGZP_{filt}", f"MAGZPT_{filt}", f"OMZP_{filt}", f"ZEROPOINT_{filt}",
                f"ABM0{band}",
            ]
            headers = [hdul[0].header] + [h.header for h in hdul[1:] if hasattr(h, "header")]
            for hdr in headers:
                for key in common_keys:
                    if key in hdr and hdr[key] is not None:
                        self.zero_point = float(hdr[key])
                        self._zp_prov.update(keyword=key, file=str(fpath), kind=(source_list_kind or "OMSRLI"))
                        print(f"[PHOT] ZP={self.zero_point:.6f} from {fpath.name} key={key} [{self._zp_prov['kind']}]")
                        return

            tokens = ("ZP", "ZERO", "MAGZ", "ABMAG", "PHOTZ")
            for hdr in headers:
                for k in hdr.keys():
                    uk = k.upper()
                    if any(t in uk for t in tokens):
                        try:
                            self.zero_point = float(hdr[k])
                            self._zp_prov.update(keyword=k, file=str(fpath), kind=(source_list_kind or "UNKNOWN"))
                            print(f"[PHOT] ZP={self.zero_point:.6f} from {fpath.name} key={k} [{self._zp_prov['kind']}]")
                            return
                        except Exception:
                            pass

        raise KeyError(f"AB zero point not found in {fpath} (filter={filt}, kind={source_list_kind}).")

    # ----------------- photometry (units-aware) -----------------

    def perform_trail_photometry(
        self,
        rectangular_aperture: RectangularAperture,
        rectangular_annulus: Optional[RectangularAnnulus] = None,
        debug: bool = False,
    ) -> PhotometryResult:
        if self.zero_point is None:
            raise RuntimeError("Zero point not calibrated. Call calibrate_against_source_list(...) first.")

        data = np.asarray(self.hduw.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("HDUW data must be a 2-D image.")

        # Units?
        unit_kind, bunit_str = self._data_unit_kind_and_bunit()
        # If ambiguous AND ZP came from OBSMLI, default to rate-mode
        if unit_kind == "unknown" and (self._zp_prov.get("kind") or "").upper() == "OBSMLI":
            unit_kind = "rate"

        # Aperture sum (exact) and effective areas
        phot_ap = aperture_photometry(data, rectangular_aperture, method='exact')
        Cap = float(phot_ap['aperture_sum'][0])
        A_ap_eff = self._effective_area_from_mask(rectangular_aperture)
        A_bg_eff = self._effective_area_bg_from_mask(rectangular_annulus)

        # Background stats (mean, std) from annulus (or global fallback)
        bkg_mean, bkg_std = self._bkg_stats_from_annulus(data, rectangular_annulus)

        # -------- RATE images (counts/s/pix) --------
        if unit_kind == "rate":
            # Total source rate in aperture:
            # Cap is in counts/s (sum over pixels); background contribution is mean_rate*area
            Cnet_rate = Cap - bkg_mean * A_ap_eff
            if not np.isfinite(Cnet_rate) or Cnet_rate <= 0:
                raise ValueError("Non-positive count rate (rate image); cannot compute magnitude.")

            # Uncertainty in rate:
            # Background variance in rate units:
            var_bkg_rate = np.nan
            if np.isfinite(bkg_std) and bkg_std > 0 and np.isfinite(A_ap_eff) and (A_bg_eff is not None) and np.isfinite(A_bg_eff) and A_bg_eff > 0:
                var_bkg_rate = A_ap_eff * (bkg_std ** 2) * (1.0 + (A_ap_eff / A_bg_eff))

            # Effective exposure time from local background: t_eff ≈ mean/std^2 (if > 0)
            t_eff = None
            if np.isfinite(bkg_mean) and np.isfinite(bkg_std) and bkg_std > 0:
                t_eff = max(bkg_mean / (bkg_std ** 2), 0.0)

            # Source Poisson variance in rate units:
            var_src_rate = np.nan
            if (t_eff is not None) and t_eff > 0 and np.isfinite(Cap) and Cap >= 0:
                var_src_rate = Cap / t_eff

            terms = [v for v in (var_bkg_rate, var_src_rate) if np.isfinite(v)]
            count_rate_err = float(np.sqrt(max(sum(terms), 0.0))) if terms else np.nan

            # Magnitude and uncertainty
            c = Cnet_rate
            m_ab = float(self.zero_point - 2.5 * math.log10(c))
            m_err = float(AB_PROP_COEFF * count_rate_err / c) if (np.isfinite(count_rate_err) and c > 0) else np.nan

            if debug:
                print("[TrailPhotometry DEBUG — RATE image]")
                print(f"  BUNIT                  = {bunit_str}")
                print(f"  Aperture sum Cap      = {Cap:.3f} (counts/s)")
                print(f"  bkg_mean              = {bkg_mean:.6f} (counts/s/pix)")
                print(f"  bkg_rms               = {bkg_std:.6f} (counts/s/pix)")
                print(f"  A_ap_eff              = {A_ap_eff:.3f} (pix)")
                print(f"  A_bg_eff              = {A_bg_eff if A_bg_eff is not None else np.nan:.3f} (pix)")
                print(f"  Net rate Cnet_rate    = {Cnet_rate:.6f} (counts/s)")
                print(f"  ZP used               = {self.zero_point:.6f} ({self._zp_prov})")
                print(f"  m_AB                  = {m_ab:.6f} mag")

            return PhotometryResult(
                mag_ab=m_ab,
                mag_err=m_err if np.isfinite(m_err) else None,
                count_rate=c,
                count_rate_err=count_rate_err if np.isfinite(count_rate_err) else None,
                net_counts=None,
                net_counts_err=None,
                aper_counts=None,  # Cap is RATE sum; keep None to avoid confusion
                bkg_per_pix=bkg_mean,
                bkg_rms_per_pix=bkg_std,
                A_ap_eff=A_ap_eff,
                A_bg_eff=A_bg_eff,
                zp_ab=self.zero_point,
                zp_keyword=self._zp_prov["keyword"],
                zp_source_file=self._zp_prov["file"],
                zp_source_kind=self._zp_prov["kind"],
            )

        # -------- COUNTS images --------
        exptime = self._exptime_from()
        if exptime is None or not np.isfinite(exptime) or exptime <= 0:
            raise ValueError("Invalid or missing EXPTIME.")

        Cnet = Cap - bkg_mean * A_ap_eff
        c = Cnet / exptime
        if not np.isfinite(c) or c <= 0:
            raise ValueError("Non-positive count rate; cannot compute magnitude.")
        m_ab = float(self.zero_point - 2.5 * math.log10(c))

        # Uncertainty in COUNTS domain:
        # Background variance in counts:
        var_bkg_counts = np.nan
        if np.isfinite(bkg_std) and np.isfinite(A_ap_eff) and (A_bg_eff is not None) and np.isfinite(A_bg_eff) and A_bg_eff > 0:
            var_bkg_counts = A_ap_eff * (bkg_std ** 2) * (1.0 + (A_ap_eff / A_bg_eff))

        # Source Poisson variance in counts:
        var_src_counts = Cap if np.isfinite(Cap) and Cap >= 0 else np.nan

        terms_counts = [v for v in (var_bkg_counts, var_src_counts) if np.isfinite(v)]
        Cnet_err = float(np.sqrt(max(sum(terms_counts), 0.0))) if terms_counts else np.nan
        count_rate_err = Cnet_err / exptime if np.isfinite(Cnet_err) else np.nan
        m_err = float(AB_PROP_COEFF * count_rate_err / c) if (np.isfinite(count_rate_err) and c > 0) else np.nan

        if debug:
            print("[TrailPhotometry DEBUG — COUNTS image]")
            print(f"  BUNIT                  = {bunit_str}")
            print(f"  Aperture sum Cap      = {Cap:.3f} (counts)")
            print(f"  bkg_mean              = {bkg_mean:.6f} (counts/pix)")
            print(f"  bkg_rms               = {bkg_std:.6f} (counts/pix)")
            print(f"  A_ap_eff              = {A_ap_eff:.3f} (pix)")
            print(f"  A_bg_eff              = {A_bg_eff if A_bg_eff is not None else np.nan:.3f} (pix)")
            print(f"  EXPTIME               = {exptime:.3f} (s)")
            print(f"  Net counts Cnet       = {Cnet:.3f} (counts)")
            print(f"  Cnet_err              = {Cnet_err:.6f} (counts)")
            print(f"  Count rate c          = {c:.6f} (counts/s)")
            print(f"  ZP used               = {self.zero_point:.6f} ({self._zp_prov})")
            print(f"  m_AB                  = {m_ab:.6f} mag")


        return PhotometryResult(
            mag_ab=m_ab,
            mag_err=m_err if np.isfinite(m_err) else None,
            count_rate=c,
            count_rate_err=count_rate_err if np.isfinite(count_rate_err) else None,
            net_counts=Cnet,
            net_counts_err=Cnet_err if np.isfinite(Cnet_err) else None,
            aper_counts=Cap,
            bkg_per_pix=bkg_mean,
            bkg_rms_per_pix=bkg_std,
            A_ap_eff=A_ap_eff,
            A_bg_eff=A_bg_eff,
            zp_ab=self.zero_point,
            zp_keyword=self._zp_prov["keyword"],
            zp_source_file=self._zp_prov["file"],
            zp_source_kind=self._zp_prov["kind"],
        )
    
    def perform_trail_rate_photometry(
        self,
        rectangular_aperture: RectangularAperture,
        rectangular_annulus: Optional[RectangularAnnulus] = None,
        debug: bool = False,
    ) -> RatePhotometryResult:
        """
        Rate-only trail photometry:
        - NO requires zero point
        - Works for RATE images (counts/s/pix) and COUNTS images (counts/pix) using EXPTIME
        - Returns net count-rate in ct/s
        """

        data = np.asarray(self.hduw.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("HDUW data must be a 2-D image.")

        unit_kind, bunit_str = self._data_unit_kind_and_bunit()

        # Aperture sum (exact) and effective areas
        phot_ap = aperture_photometry(data, rectangular_aperture, method="exact")
        Cap = float(phot_ap["aperture_sum"][0])  # in native units: counts OR counts/s

        A_ap_eff = self._effective_area_from_mask(rectangular_aperture)
        A_bg_eff = self._effective_area_bg_from_mask(rectangular_annulus)

        bkg_mean, bkg_std = self._bkg_stats_from_annulus(data, rectangular_annulus)  # per-pixel, native units

        # ---------------- RATE images (counts/s/pix) ----------------
        if unit_kind == "rate":
            net_rate = Cap - bkg_mean * A_ap_eff  # ct/s

            # uncertainty in rate units (same approach as your RATE branch)
            var_bkg_rate = np.nan
            if np.isfinite(bkg_std) and bkg_std > 0 and np.isfinite(A_ap_eff) and (A_bg_eff is not None) and np.isfinite(A_bg_eff) and A_bg_eff > 0:
                var_bkg_rate = A_ap_eff * (bkg_std ** 2) * (1.0 + (A_ap_eff / A_bg_eff))

            # try estimating t_eff as in your code; if not available, leave source term out
            t_eff = None
            if np.isfinite(bkg_mean) and np.isfinite(bkg_std) and bkg_std > 0:
                t_eff = max(bkg_mean / (bkg_std ** 2), 0.0)

            var_src_rate = np.nan
            if (t_eff is not None) and t_eff > 0 and np.isfinite(Cap) and Cap >= 0:
                var_src_rate = Cap / t_eff

            terms = [v for v in (var_bkg_rate, var_src_rate) if np.isfinite(v)]
            net_rate_err = float(np.sqrt(max(sum(terms), 0.0))) if terms else None

            if debug:
                print("[TrailRatePhotometry DEBUG — RATE image]")
                print(f"  BUNIT               = {bunit_str}")
                print(f"  Cap (aperture_sum)  = {Cap:.6f} (counts/s)")
                print(f"  bkg_mean            = {bkg_mean:.6e} (counts/s/pix)")
                print(f"  bkg_rms             = {bkg_std:.6e} (counts/s/pix)")
                print(f"  A_ap_eff            = {A_ap_eff:.3f} (pix)")
                print(f"  A_bg_eff            = {A_bg_eff if A_bg_eff is not None else np.nan:.3f} (pix)")
                print(f"  net_rate            = {net_rate:.6f} (counts/s)")

            return RatePhotometryResult(
                net_rate=float(net_rate),
                net_rate_err=net_rate_err,
                aper_sum=Cap,
                bkg_per_pix=bkg_mean,
                bkg_rms_per_pix=bkg_std,
                A_ap_eff=A_ap_eff,
                A_bg_eff=A_bg_eff,
                unit_kind="rate",
                exptime=None,
                bunit_str=bunit_str,
            )

        # ---------------- COUNTS images (counts/pix) ----------------
        exptime = self._exptime_from()
        if exptime is None or not np.isfinite(exptime) or exptime <= 0:
            raise ValueError("Invalid or missing EXPTIME for COUNTS image.")

        # net counts in aperture:
        net_counts = Cap - bkg_mean * A_ap_eff
        net_rate = net_counts / exptime

        # uncertainties in counts domain (your COUNTS branch)
        var_bkg_counts = np.nan
        if np.isfinite(bkg_std) and np.isfinite(A_ap_eff) and (A_bg_eff is not None) and np.isfinite(A_bg_eff) and A_bg_eff > 0:
            var_bkg_counts = A_ap_eff * (bkg_std ** 2) * (1.0 + (A_ap_eff / A_bg_eff))

        var_src_counts = Cap if np.isfinite(Cap) and Cap >= 0 else np.nan

        terms_counts = [v for v in (var_bkg_counts, var_src_counts) if np.isfinite(v)]
        net_counts_err = float(np.sqrt(max(sum(terms_counts), 0.0))) if terms_counts else None
        net_rate_err = (net_counts_err / exptime) if (net_counts_err is not None) else None

        if debug:
            print("[TrailRatePhotometry DEBUG — COUNTS image]")
            print(f"  BUNIT               = {bunit_str}")
            print(f"  Cap (aperture_sum)  = {Cap:.3f} (counts)")
            print(f"  bkg_mean            = {bkg_mean:.6e} (counts/pix)")
            print(f"  bkg_rms             = {bkg_std:.6e} (counts/pix)")
            print(f"  A_ap_eff            = {A_ap_eff:.3f} (pix)")
            print(f"  A_bg_eff            = {A_bg_eff if A_bg_eff is not None else np.nan:.3f} (pix)")
            print(f"  EXPTIME             = {exptime:.3f} (s)")
            print(f"  net_counts          = {net_counts:.3f} (counts)")
            print(f"  net_rate            = {net_rate:.6f} (counts/s)")

        return RatePhotometryResult(
            net_rate=float(net_rate),
            net_rate_err=net_rate_err,
            aper_sum=Cap,
            bkg_per_pix=bkg_mean,
            bkg_rms_per_pix=bkg_std,
            A_ap_eff=A_ap_eff,
            A_bg_eff=A_bg_eff,
            unit_kind="counts",
            exptime=float(exptime),
            bunit_str=bunit_str,
        )


# # --- phot.py (añadir imports si no están) ---
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, Any, List
# import math
# import numpy as np

# from astropy.wcs import WCS
# from photutils.aperture import RectangularAperture, RectangularAnnulus

# ------------------------------------------------------------
# Pixel scale helpers
# ------------------------------------------------------------
def _get_arcsec_per_pix_from_header(header) -> Optional[float]:
    """
    Try to infer arcsec/pix from FITS WCS keywords.
    Returns None if cannot be inferred robustly.
    """
    # 1) CDELT in degrees/pix
    for k in ("CDELT1", "CDELT2"):
        if k in header:
            try:
                val = float(header[k])
                if np.isfinite(val) and val != 0:
                    return abs(val) * 3600.0
            except Exception:
                pass

    # 2) CD matrix in degrees/pix
    # Pixel scale approx sqrt(CD1_1^2 + CD2_1^2) etc; we take CD1_1 if present
    for k in ("CD1_1", "CD2_2"):
        if k in header:
            try:
                val = float(header[k])
                if np.isfinite(val) and val != 0:
                    return abs(val) * 3600.0
            except Exception:
                pass

    # 3) try WCS (if header has enough)
    try:
        w = WCS(header)
        # proj_plane_pixel_scales gives degrees/pix in each axis
        from astropy.wcs.utils import proj_plane_pixel_scales
        sc = proj_plane_pixel_scales(w)  # degrees/pix
        if sc is not None and len(sc) >= 1 and np.isfinite(sc[0]) and sc[0] != 0:
            return float(abs(sc[0]) * 3600.0)
    except Exception:
        pass

    return None


def arcsec_to_pix(hduw, arcsec: float, fallback_arcsec_per_pix: Optional[float] = None) -> int:
    """
    Convert arcsec to pixels using header WCS if possible.
    Falls back to fallback_arcsec_per_pix if provided.
    """
    hdr = None
    try:
        hdr = hduw.hdu.header
    except Exception:
        try:
            hdr = hduw.header
        except Exception:
            hdr = None

    asp = None
    if hdr is not None:
        asp = _get_arcsec_per_pix_from_header(hdr)

    if asp is None:
        if fallback_arcsec_per_pix is None:
            raise RuntimeError("Cannot infer plate scale (arcsec/pix) from header; provide fallback.")
        asp = float(fallback_arcsec_per_pix)

    pix = int(round(float(arcsec) / asp))
    return max(pix, 1)


def build_rect_ap_ann(
    x: float, y: float,
    width_pix: float,
    height_pix: float,
    theta_rad: float,
    semi_out_pix: float,
    bg_center: Optional[Tuple[float, float]] = None,
) -> Tuple[RectangularAperture, RectangularAnnulus]:
    """
    Build rectangular aperture + annulus centered on same point (or external bg center).
    """
    centre = (float(x), float(y))
    ann_c = bg_center if bg_center is not None else centre

    ap = RectangularAperture(positions=centre, w=float(width_pix), h=float(height_pix), theta=float(theta_rad))
    an = RectangularAnnulus(
        positions=ann_c,
        w_in=float(width_pix),
        w_out=float(width_pix) + 2.0 * float(semi_out_pix),
        h_in=float(height_pix),
        h_out=float(height_pix) + 2.0 * float(semi_out_pix),
        theta=float(theta_rad),
    )
    return ap, an


@dataclass
class C2StarMeasurement:
    slot: int

    # net rates (ct/s)
    rate6: float
    err6: Optional[float]
    rate35: float
    err35: Optional[float]

    # ratio
    c2: float
    c2_err: Optional[float]

    # --- extra audit fields (6") ---
    aper_sum_6: float
    bkg_per_pix_6: float
    bkg_rms_per_pix_6: float
    A_ap_eff_6: float
    A_bg_eff_6: Optional[float]
    unit_kind_6: str
    exptime_6: Optional[float]

    # --- extra audit fields (35") ---
    aper_sum_35: float
    bkg_per_pix_35: float
    bkg_rms_per_pix_35: float
    A_ap_eff_35: float
    A_bg_eff_35: Optional[float]
    unit_kind_35: str
    exptime_35: Optional[float]


def compute_c2_for_star(
    phot: "PhotTable",
    x: float, y: float,
    width_pix: float,
    theta_rad: float,
    height6_pix: float,
    height35_pix: float,
    semi_out6_pix: float,
    semi_out35_pix: float,
    bg_center=None,
    debug: bool = False,
) -> C2StarMeasurement:

    ap6, an6 = build_rect_ap_ann(x, y, width_pix, height6_pix, theta_rad, semi_out6_pix, bg_center=bg_center)
    r6 = phot.perform_trail_rate_photometry(ap6, an6, debug=debug)
    if not np.isfinite(r6.net_rate) or r6.net_rate <= 0:
        raise ValueError(f"Invalid R6 net_rate: {r6.net_rate}")

    ap35, an35 = build_rect_ap_ann(x, y, width_pix, height35_pix, theta_rad, semi_out35_pix, bg_center=bg_center)
    r35 = phot.perform_trail_rate_photometry(ap35, an35, debug=debug)
    if not np.isfinite(r35.net_rate) or r35.net_rate <= 0:
        raise ValueError(f"Invalid R35 net_rate: {r35.net_rate}")

    c2 = float(r35.net_rate / r6.net_rate)

    # Propagación opcional si tienes net_rate_err
    c2_err = None
    if r6.net_rate_err is not None and r35.net_rate_err is not None:
        e6 = float(r6.net_rate_err)
        e35 = float(r35.net_rate_err)
        c2_err = float(c2 * np.sqrt((e35 / r35.net_rate) ** 2 + (e6 / r6.net_rate) ** 2))

    return C2StarMeasurement(
        slot=-1,
        rate6=float(r6.net_rate), err6=r6.net_rate_err,
        rate35=float(r35.net_rate), err35=r35.net_rate_err,
        c2=c2, c2_err=c2_err,

        aper_sum_6=float(r6.aper_sum),
        bkg_per_pix_6=float(r6.bkg_per_pix),
        bkg_rms_per_pix_6=float(r6.bkg_rms_per_pix),
        A_ap_eff_6=float(r6.A_ap_eff),
        A_bg_eff_6=(float(r6.A_bg_eff) if r6.A_bg_eff is not None else None),
        unit_kind_6=str(r6.unit_kind),
        exptime_6=(float(r6.exptime) if r6.exptime is not None else None),

        aper_sum_35=float(r35.aper_sum),
        bkg_per_pix_35=float(r35.bkg_per_pix),
        bkg_rms_per_pix_35=float(r35.bkg_rms_per_pix),
        A_ap_eff_35=float(r35.A_ap_eff),
        A_bg_eff_35=(float(r35.A_bg_eff) if r35.A_bg_eff is not None else None),
        unit_kind_35=str(r35.unit_kind),
        exptime_35=(float(r35.exptime) if r35.exptime is not None else None),
    )

