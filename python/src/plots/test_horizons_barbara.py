#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

from astropy.time import Time
from astroquery.jplhorizons import Horizons


TARGET = "234"  # (234) Barbara

T1 = Time("2016-08-28T02:36:00.000", scale="utc").jd
T2 = Time("2016-08-28T03:51:00.000", scale="utc").jd

# Test several possible observer syntaxes/codes.
# 500 / @399 are controls.
# XMM codes are intentionally exploratory.
LOCATIONS = [
    "500",        # geocenter
    "@399",       # Earth
    "-125544",    # possible spacecraft-style numeric code
    "@-125544",   # possible Horizons center syntax
    "XMM",        # possible name alias
    "XMM-Newton",
    "@XMM",
    "@XMM-Newton",
]


def get_pos(location: str, jd: float):
    obj = Horizons(
        id=TARGET,
        id_type="smallbody",
        location=location,
        epochs=jd,
    )
    eph = obj.ephemerides()

    print(f"[HORIZONS] location={location!r} columns={eph.colnames}")

    row = eph[0]
    ra = float(row["RA"])
    dec = float(row["DEC"])
    vmag = row["V"] if "V" in eph.colnames else None

    return ra, dec, vmag


def motion_params(ra1: float, dec1: float, ra2: float, dec2: float):
    mean_dec = 0.5 * (dec1 + dec2)
    dra = (ra2 - ra1) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec = (dec2 - dec1) * 3600.0
    length = math.hypot(dra, ddec)

    pa = math.degrees(math.atan2(dra, ddec))
    if pa < 0:
        pa += 360.0

    return dra, ddec, length, pa


def main():
    print(
        "location,ra1,dec1,ra2,dec2,"
        "dra_arcsec,ddec_arcsec,trail_len_arcsec,pa_deg_E_of_N,"
        "vmag1,vmag2"
    )

    for loc in LOCATIONS:
        try:
            ra1, dec1, v1 = get_pos(loc, T1)
            ra2, dec2, v2 = get_pos(loc, T2)
            dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)

            print(
                f"{loc},"
                f"{ra1:.9f},{dec1:.9f},"
                f"{ra2:.9f},{dec2:.9f},"
                f"{dra:.3f},{ddec:.3f},{length:.3f},{pa:.3f},"
                f"{v1},{v2}"
            )

        except Exception as e:
            print(f"{loc},FAILED,{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()