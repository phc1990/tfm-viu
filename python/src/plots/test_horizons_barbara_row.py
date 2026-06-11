#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.time import Time
from astroquery.jplhorizons import Horizons
import math

TARGET = "234"  # Barbara
TARGET_NAME = "Barbara"
OBSID = "0781040101"
FITS_NAME = "P0781040101OMS008FSIMAGL000.FTZ"
FILTER = "L"
LOCATION = "@XMM"

T1 = Time("2016-08-28T02:36:00.000", scale="utc")
T2 = Time("2016-08-28T03:51:00.000", scale="utc")


def get_pos(target, t):
    obj = Horizons(
        id=target,
        id_type="smallbody",
        location=LOCATION,
        epochs=t.jd,
    )
    eph = obj.ephemerides()
    row = eph[0]
    return float(row["RA"]), float(row["DEC"]), float(row["V"])


def motion_params(ra1, dec1, ra2, dec2):
    mean_dec = 0.5 * (dec1 + dec2)
    dra = (ra2 - ra1) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec = (dec2 - dec1) * 3600.0
    length = math.hypot(dra, ddec)

    pa = math.degrees(math.atan2(dra, ddec))
    if pa < 0:
        pa += 360.0

    return dra, ddec, length, pa


ra1, dec1, v1 = get_pos(TARGET, T1)
ra2, dec2, v2 = get_pos(TARGET, T2)

dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)

print(
    "observation_id,sso_name,ra_deg_1,dec_deg_1,ra_deg_2,dec_deg_2,"
    "filter,xmatch_type,v_mag_1,v_mag_1_corrected,mlim_obs,FITS_FILE,DECISION,"
    "dra_arcsec,ddec_arcsec,trail_len_arcsec,pa_deg_E_of_N,"
    "frame_t1,frame_t2,location"
)

print(
    f"{OBSID},{TARGET_NAME},"
    f"{ra1:.9f},{dec1:.9f},"
    f"{ra2:.9f},{dec2:.9f},"
    f"{FILTER},2,"
    f"{v1:.3f},{v1:.3f},20.696785,"
    f"{FITS_NAME},D,"
    f"{dra:.3f},{ddec:.3f},{length:.3f},{pa:.3f},"
    f"{T1.isot},{T2.isot},{LOCATION}"
)