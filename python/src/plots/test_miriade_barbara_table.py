#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.imcce import Miriade


TARGET = "Barbara"
T1 = Time("2016-08-28T02:36:00.000", scale="utc")
T2 = Time("2016-08-28T03:51:00.000", scale="utc")
LOCATIONS = ["500", "@399", "@-10", "xmm-newton"]


def as_float(x):
    try:
        return float(getattr(x, "value", x))
    except Exception:
        return math.nan


def get_ra_dec(eph):
    row = eph[0]

    # Miriade often returns RA/DEC in degrees, but column names can vary.
    candidates = [
        ("RA", "DEC"),
        ("RAJ2000", "DECJ2000"),
        ("RA_ICRS", "DEC_ICRS"),
        ("ra", "dec"),
    ]

    for ra_col, dec_col in candidates:
        if ra_col in eph.colnames and dec_col in eph.colnames:
            return as_float(row[ra_col]), as_float(row[dec_col]), ra_col, dec_col

    raise KeyError(f"No RA/DEC columns found. Columns={eph.colnames}")


def eph_at(location, epoch):
    eph = Miriade.get_ephemerides(
        TARGET,
        objtype="asteroid",
        epoch=epoch,
        location=location,
        epoch_nsteps=1,
    )
    ra, dec, ra_col, dec_col = get_ra_dec(eph)
    return ra, dec, ra_col, dec_col


def motion_params(ra1, dec1, ra2, dec2):
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
        "ra_col,dec_col"
    )

    for loc in LOCATIONS:
        try:
            ra1, dec1, ra_col, dec_col = eph_at(loc, T1)
            ra2, dec2, _, _ = eph_at(loc, T2)
            dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)

            print(
                f"{loc},"
                f"{ra1:.9f},{dec1:.9f},"
                f"{ra2:.9f},{dec2:.9f},"
                f"{dra:.3f},{ddec:.3f},{length:.3f},{pa:.3f},"
                f"{ra_col},{dec_col}"
            )
        except Exception as e:
            print(f"{loc},FAILED,{repr(e)}")


if __name__ == "__main__":
    main()