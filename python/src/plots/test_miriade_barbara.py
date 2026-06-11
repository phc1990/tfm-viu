#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.time import Time
from astroquery.imcce import Miriade


TARGET = "Barbara"

TIMES = [
    "2016-08-28T02:36:00.000",
    "2016-08-28T03:51:00.000",
]

LOCATIONS = [
    "500",
    "@399",
    "@-10",
    "xmm-newton",
]


def print_eph(location: str, isot: str):
    print("\n" + "=" * 80)
    print(f"target={TARGET}  location={location}  epoch={isot}")

    eph = Miriade.get_ephemerides(
        TARGET,
        objtype="asteroid",
        epoch=Time(isot, scale="utc"),
        location=location,
        epoch_nsteps=1,
    )

    print("columns:", eph.colnames)

    row = eph[0]

    # Print all columns first, because Miriade column names vary by astroquery version.
    for c in eph.colnames:
        print(f"{c:20s} = {row[c]}")

    # Try common RA/DEC columns.
    for ra_col, dec_col in [
        ("RA", "DEC"),
        ("RAJ2000", "DECJ2000"),
        ("RA_ICRS", "DEC_ICRS"),
        ("ra", "dec"),
    ]:
        if ra_col in eph.colnames and dec_col in eph.colnames:
            print(f"\nBEST_GUESS {ra_col},{dec_col}: {row[ra_col]}, {row[dec_col]}")


def main():
    for loc in LOCATIONS:
        for t in TIMES:
            try:
                print_eph(loc, t)
            except Exception as e:
                print("\n" + "=" * 80)
                print(f"FAILED target={TARGET} location={loc} epoch={t}")
                print(repr(e))


if __name__ == "__main__":
    main()