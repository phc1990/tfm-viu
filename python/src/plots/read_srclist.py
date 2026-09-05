SRCLIST="/Users/eracero/workspace/tfm-viu/temp/0821871601/L/P0821871601OMS006SWSRLIL000.FTZ"

from astropy.io import fits
import numpy as np
import sys

fn = sys.argv[1]

ra0 = 342.035325
dec0 = -22.211951

with fits.open(fn, memmap=False) as hdul:

    for i, hdu in enumerate(hdul):
        if getattr(hdu, "columns", None) is not None:
            tab = hdu.data
            ihdu = i
            break

    names = tab.columns.names
    cols = {name.upper(): name for name in names}

    ra_col = cols.get("RA_CORR") or cols.get("RA")
    dec_col = cols.get("DEC_CORR") or cols.get("DEC")

    ra = np.asarray(tab[ra_col], dtype=float)
    dec = np.asarray(tab[dec_col], dtype=float)

    dra = (ra - ra0) * np.cos(np.deg2rad(dec0))
    ddec = dec - dec0

    dist2 = dra**2 + ddec**2
    j = int(np.nanargmin(dist2))

    sep_arcsec = 3600.0 * np.sqrt(dist2[j])

    print("HDU =", ihdu)
    print("FITS row 0-based =", j)
    print("FITS row 1-based =", j + 1)
    print("Separation =", sep_arcsec, "arcsec")
    print()

    for name in names:
        print(f"{name:25s} = {tab[name][j]!r}")