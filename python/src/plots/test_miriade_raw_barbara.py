#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from astropy.io.votable import parse_single_table
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u


BASE_URL = "https://ssp.imcce.fr/webservices/miriade/ephemcc_query.php"

TARGET = "234"  # (234) Barbara
T1 = Time("2016-08-28T02:36:00.000", scale="utc")
T2 = Time("2016-08-28T03:51:00.000", scale="utc")

LOCATIONS = [
    "500",
    "@399",
    "@-10",
    "xmm-newton",
]


def safe_observer_name(observer: str) -> str:
    return (
        str(observer)
        .replace("@", "at")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "_")
    )


def scalar_to_str(x) -> str:
    """
    Robust scalar -> string conversion for astropy table cells, masked values,
    numpy scalars, bytes, and plain Python objects.
    """
    try:
        if hasattr(x, "mask") and bool(x.mask):
            return ""
    except Exception:
        pass

    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()

    return str(x).strip()


def parse_ra_value(x) -> float:
    """
    Parse RA from Miriade.

    Miriade may return RA either as:
      - numeric degrees
      - sexagesimal hourangle, e.g. '03 20 14.93'
      - sexagesimal with colons, e.g. '03:20:14.93'
    """
    s = scalar_to_str(x)
    if not s:
        return math.nan

    # Direct numeric degrees
    try:
        return float(s)
    except Exception:
        pass

    # Sexagesimal hourangle
    try:
        return Angle(s, unit=u.hourangle).degree
    except Exception:
        pass

    # Fallback: sexagesimal degrees
    try:
        return Angle(s, unit=u.deg).degree
    except Exception:
        pass

    print(f"[PARSE][RA] could not parse {s!r}")
    return math.nan


def parse_dec_value(x) -> float:
    """
    Parse DEC from Miriade.

    Miriade may return DEC either as:
      - numeric degrees
      - sexagesimal degrees, e.g. '+00 24 54.1'
      - sexagesimal with colons, e.g. '+00:24:54.1'
    """
    s = scalar_to_str(x)
    if not s:
        return math.nan

    # Direct numeric degrees
    try:
        return float(s)
    except Exception:
        pass

    # Sexagesimal degrees
    try:
        return Angle(s, unit=u.deg).degree
    except Exception:
        pass

    print(f"[PARSE][DEC] could not parse {s!r}")
    return math.nan


def miriade_raw(target: str, epoch: Time, observer: str):
    params = {
        "-name": target,
        "-type": "Asteroid",
        "-ep": str(epoch.jd),
        "-step": "1d",
        "-nbd": "1",
        "-observer": observer,
        "-output": "--jul",
        "-tscale": "UTC",
        "-theory": "INPOP",
        "-teph": "1",
        "-tcoor": "1",
        "-rplane": "1",
        "-oscelem": "ASTORB",
        "-mime": "votable",
    }

    url = BASE_URL + "?" + urlencode(params)
    print(f"[MIRIADE][URL] {url}")

    req = Request(
        url,
        headers={
            "User-Agent": "tfm-viu-miriade-test/1.0",
        },
    )

    try:
        with urlopen(req, timeout=60) as r:
            data = r.read()

    except HTTPError as e:
        body = e.read()
        tmp = (
            Path(tempfile.gettempdir())
            / f"miriade_ERROR_{target}_{safe_observer_name(observer)}.txt"
        )
        tmp.write_bytes(body)

        print(f"[MIRIADE][HTTPERROR] code={e.code} reason={e.reason}")
        print(f"[MIRIADE][HTTPERROR] body saved {tmp} size={len(body)} bytes")
        print("[MIRIADE][HTTPERROR] body preview:")
        print(body[:2000].decode("utf-8", errors="replace"))
        raise

    except URLError as e:
        print(f"[MIRIADE][URLERROR] {repr(e)}")
        raise

    # Save raw response in case parsing fails
    tmp = (
        Path(tempfile.gettempdir())
        / f"miriade_{target}_{safe_observer_name(observer)}.xml"
    )
    tmp.write_bytes(data)
    print(f"[MIRIADE][RAW] saved {tmp} size={len(data)} bytes")

    table = parse_single_table(str(tmp)).to_table()
    return table


def get_ra_dec(tab):
    print("[MIRIADE] columns:", tab.colnames)

    if len(tab) == 0:
        raise ValueError("Miriade returned an empty table")

    row = tab[0]

    candidates = [
        ("RA", "DEC"),
        ("RAJ2000", "DECJ2000"),
        ("RA_ICRS", "DEC_ICRS"),
        ("ra", "dec"),
        ("_RAJ2000", "_DEJ2000"),
        ("alpha", "delta"),
    ]

    for ra_col, dec_col in candidates:
        if ra_col in tab.colnames and dec_col in tab.colnames:
            raw_ra = row[ra_col]
            raw_dec = row[dec_col]

            print(f"[MIRIADE] raw {ra_col}={raw_ra!r} {dec_col}={raw_dec!r}")

            ra = parse_ra_value(raw_ra)
            dec = parse_dec_value(raw_dec)

            if not math.isfinite(ra) or not math.isfinite(dec):
                print("[MIRIADE] first row for debugging:")
                for c in tab.colnames:
                    print(f"  {c:24s} = {row[c]!r}")

            return ra, dec, ra_col, dec_col

    print("[MIRIADE] first row:")
    for c in tab.colnames:
        print(f"  {c:24s} = {row[c]!r}")

    raise KeyError("No obvious RA/DEC columns found")


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
        "ra_col,dec_col"
    )

    for loc in LOCATIONS:
        try:
            tab1 = miriade_raw(TARGET, T1, loc)
            ra1, dec1, ra_col, dec_col = get_ra_dec(tab1)

            tab2 = miriade_raw(TARGET, T2, loc)
            ra2, dec2, _, _ = get_ra_dec(tab2)

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