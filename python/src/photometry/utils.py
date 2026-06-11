from __future__ import annotations

from typing import Any, Optional

import numpy as np


def angle_to_rad(theta: Any) -> float:
    """
    Return theta as a plain float in radians.

    Handles:
      - plain float
      - astropy Quantity with angular units
    """
    if theta is None:
        return 0.0

    try:
        if hasattr(theta, "to_value"):
            import astropy.units as u
            return float(theta.to_value(u.rad))
    except Exception:
        pass

    try:
        return float(theta)
    except Exception:
        return 0.0


def exptime_from_hduw(hduw: Any, *, debug: bool = False) -> Optional[float]:
    """
    Return XMM-OM exposure time in seconds.

    For OM SIMAG products this is stored in the primary header keyword:
        EXPOSURE [seconds]

    Fallbacks are kept only for compatibility with the HDUW wrapper.
    """

    # 1) Prefer the FITS header value for XMM-OM SIMAG.
    try:
        hdr = getattr(getattr(hduw, "hdu", None), "header", None)
        if hdr is not None and "EXPOSURE" in hdr and hdr["EXPOSURE"] is not None:
            val = float(hdr["EXPOSURE"])
            if np.isfinite(val) and val > 0:
                if debug:
                    print(f"[UTILS][EXPTIME] Using header EXPOSURE={val} s")
                return val
    except Exception:
        pass

    # 2) Fallback: attributes already parsed by HDUW.
    for attr in ("texp", "exptime", "exposure_time"):
        try:
            val = getattr(hduw, attr, None)
            if val is not None:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    if debug:
                        print(f"[UTILS][EXPTIME] Using HDUW.{attr}={val} s")
                    return val
        except Exception:
            pass

    if debug:
        print("[UTILS][EXPTIME] WARN: no valid EXPOSURE found")

    return None