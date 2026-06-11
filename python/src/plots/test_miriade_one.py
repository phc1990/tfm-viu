#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.time import Time
from astroquery.imcce import Miriade

Miriade.Conf.ephemcc_server.set(
    "https://ssp.imcce.fr/webservices/miriade/ephemcc_query.php"
)

TARGET = "234"
EPOCH = Time("2016-08-28T02:36:00.000", scale="utc")

for loc in ["500", "@-10"]:
    print("\n" + "=" * 80)
    print(f"TARGET={TARGET} LOC={loc}")

    try:
        eph = Miriade.get_ephemerides(
            TARGET,
            objtype="asteroid",
            epoch=EPOCH,
            location=loc,
            epoch_nsteps=1,
            cache=False,
        )
        print("URI:", getattr(Miriade, "uri", None))
        print(eph.colnames)
        print(eph)
    except Exception as e:
        print("FAILED")
        print(repr(e))