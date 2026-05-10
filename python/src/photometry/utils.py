from typing import Optional

import numpy as np

from python.src.screening import fits


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

