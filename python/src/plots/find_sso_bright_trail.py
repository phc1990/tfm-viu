from __future__ import annotations

from pathlib import Path
import math

from astroquery.imcce import Skybot
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.table import Table


# ---------------------------------------------------------------------
# Observation/frame metadata
# ---------------------------------------------------------------------
OBS_ID = "0802200801"
FITS_FILE = "P0802200801OMS403SIMAGE2000.FTZ"
FILTER = "L"  # UVW1 in your screening notation

DATE_OBS = "2017-11-28T05:59:35.000"
EXPOSURE_S = 2980.0

# XMM-Newton spacecraft point of view.
# Use the one that worked in your current SkyBoT script.
LOCATION = "xmm-newton"
# LOCATION = "@-10"   # alternative if needed

t1 = Time(DATE_OBS, scale="utc")
t2 = t1 + EXPOSURE_S * u.s
tmid = t1 + 0.5 * EXPOSURE_S * u.s

# ---------------------------------------------------------------------
# Search region
# ---------------------------------------------------------------------
# Approximate bright trail midpoint from your PNG.
# Adjust if you measure a better centroid.
center = SkyCoord(
    ra="11h00m34.5s",
    dec="+09d57m20s",
    frame="icrs",
)

# Use generous radius so the same candidate is caught at t1/t2.
radius = 45 * u.arcmin


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def as_str(x) -> str:
    """Robust conversion for masked astropy table values."""
    try:
        if getattr(x, "mask", False):
            return ""
    except Exception:
        pass
    return str(x).strip()


# def as_float(x) -> float:
#     """
#     Robust conversion for astropy Table scalar values, including Quantity,
#     MaskedColumn elements, strings, numpy scalars, etc.
#     """
#     import math
#     import numpy as np
#     import astropy.units as u

#     try:
#         # Masked value
#         if getattr(x, "mask", False):
#             return math.nan
#     except Exception:
#         pass

#     try:
#         # Astropy Quantity, e.g. 165 deg
#         if hasattr(x, "unit") and hasattr(x, "value"):
#             try:
#                 return float(x.to_value(u.deg))
#             except Exception:
#                 return float(x.value)
#     except Exception:
#         pass

#     try:
#         # Astropy scalar wrappers
#         if hasattr(x, "value"):
#             return float(x.value)
#     except Exception:
#         pass

#     try:
#         # Numpy scalar
#         if isinstance(x, np.generic):
#             return float(x)
#     except Exception:
#         pass

#     try:
#         return float(str(x).strip())
#     except Exception:
#         return math.nan
def as_float(x) -> float:
    import math
    import numpy as np

    try:
        if getattr(x, "mask", False):
            return math.nan
    except Exception:
        pass

    # Astropy Quantity or Column scalar with unit
    try:
        if hasattr(x, "value"):
            x = x.value
    except Exception:
        pass

    # Sometimes value is a 0-d array
    try:
        if isinstance(x, np.ndarray):
            x = x.item()
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        pass

    # Last resort: parse first numeric token from string like "165.12 deg"
    try:
        import re
        s = str(x)
        m = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", s)
        if m:
            return float(m.group(0))
    except Exception:
        pass

    return math.nan

# def object_key(row) -> str:
#     """
#     Prefer numbered designation when available; otherwise use Name.
#     SkyBoT uses Number='--' for unnumbered objects.
#     """
#     number = as_str(row.get("Number", ""))
#     name = as_str(row.get("Name", ""))

#     if number and number not in {"--", "None", "nan"}:
#         return f"N:{number}"

#     return f"NAME:{name}"
def object_key(row) -> str:
    name = as_str(row["Name"])
    return name.replace(" ", "_")


def query_skybot(epoch: Time, label: str) -> Table:
    print(f"[SKYBOT] Query {label}: {epoch.isot} location={LOCATION}")
    tab = Skybot.cone_search(
        center,
        radius,
        epoch,
        location=LOCATION,
    )
    print(f"[SKYBOT] {label}: {len(tab)} rows")
    return tab


def table_to_map(tab: Table) -> dict[str, object]:
    out = {}
    for row in tab:
        k = object_key(row)
        if k:
            out[k] = row
    return out

# def get_ra_dec(row, tab=None) -> tuple[float, float]:
#     """
#     Prefer _raj2000/_decj2000 in decimal degrees.
#     Falls back to RA/DEC.
#     """
#     colnames = set(tab.colnames) if tab is not None else set(row.colnames) if hasattr(row, "colnames") else set()

#     if "_raj2000" in colnames and "_decj2000" in colnames:
#         ra = as_float(row["_raj2000"])
#         dec = as_float(row["_decj2000"])
#     else:
#         ra = as_float(row["RA"])
#         dec = as_float(row["DEC"])

#     return ra, dec
def get_ra_dec(row) -> tuple[float, float]:
    """
    Extract J2000 RA/Dec in decimal degrees from SkyBoT row.
    """
    ra = as_float(row["_raj2000"])
    dec = as_float(row["_decj2000"])

    if not math.isfinite(ra) or not math.isfinite(dec):
        ra = as_float(row["RA"])
        dec = as_float(row["DEC"])

    return ra, dec

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
tab1 = query_skybot(t1, "START")
tab2 = query_skybot(t2, "END")
tabm = query_skybot(tmid, "MID")

print("\n[RAW START TABLE]")
print(tab1)
print(tab1.colnames)
tab1[:5].pprint_all()

print("\n[RAW END TABLE]")
print(tab2)
print(tab2.colnames)
tab2[:5].pprint_all()

print("\n[RAW MID TABLE]")
print(tabm)
print(tabm.colnames)
tabm[:5].pprint_all()

m1 = table_to_map(tab1)
m2 = table_to_map(tab2)
mm = table_to_map(tabm)

common_keys = sorted(set(m1) & set(m2))

print(f"[DBG] START keys: {len(m1)}")
print(f"[DBG] END keys:   {len(m2)}")
print(f"[DBG] COMMON:     {len(common_keys)}")

if not common_keys:
    print("[DBG] First START keys:", list(m1.keys())[:10])
    print("[DBG] First END keys:", list(m2.keys())[:10])

rows = []
for k in common_keys:
    r1 = m1[k]
    r2 = m2[k]
    rm = mm.get(k, r1)

    name = as_str(rm["Name"])
    number = as_str(rm["Number"])
    obj_type = as_str(rm["Type"]) if "Type" in rm.colnames else ""

    # Magnitude: use MID if available, otherwise START
    if "V" in rm.colnames:
        vmag = as_float(rm["V"])
    elif "V" in r1.colnames:
        vmag = as_float(r1["V"])
    else:
        vmag = math.nan


    ra1, dec1 = get_ra_dec(r1)
    ra2, dec2 = get_ra_dec(r2)

    if not all(math.isfinite(v) for v in [ra1, dec1, ra2, dec2]):
        print(
            "[WARN] bad RA/Dec",
            as_str(rm["Name"]),
            "START raw=", r1["_raj2000"], r1["_decj2000"],
            "END raw=", r2["_raj2000"], r2["_decj2000"],
            "parsed=", ra1, dec1, ra2, dec2,
        )
        continue

     # Motion over exposure.
    # RA term corrected by cos(dec).
    mean_dec = 0.5 * (dec1 + dec2)
    dra_arcsec = (ra2 - ra1) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec_arcsec = (dec2 - dec1) * 3600.0

    trail_len_arcsec = math.hypot(dra_arcsec, ddec_arcsec)

    # Position angle, degrees East of North.
    # atan2(East component, North component)
    pa_deg = math.degrees(math.atan2(dra_arcsec, ddec_arcsec))
    if pa_deg < 0:
        pa_deg += 360.0

    rows.append(
        {
            "number": number if number not in {"--", "None", "nan"} else "",
            "name": name,
            "type": obj_type,
            "V_mid": vmag,
            "ra_deg_1": ra1,
            "dec_deg_1": dec1,
            "ra_deg_2": ra2,
            "dec_deg_2": dec2,
            "trail_len_arcsec": trail_len_arcsec,
            "pa_deg_E_of_N": pa_deg,
        }
    )

print(f"[DBG] rows built: {len(rows)}")

# Sort by V magnitude, brightest first
rows.sort(key=lambda r: (math.inf if math.isnan(r["V_mid"]) else r["V_mid"]))

print()
print("# Candidate start/end positions from XMM-Newton SkyBoT")
print(f"# t1 = {t1.isot}")
print(f"# t2 = {t2.isot}")
print(f"# center = {center.to_string('hmsdms')}")
print()

print(
    "number,name,type,V_mid,"
    "ra_deg_1,dec_deg_1,ra_deg_2,dec_deg_2,"
    "trail_len_arcsec,pa_deg_E_of_N"
)

for r in rows:
    print(
        f"{r['number']},{r['name']},{r['type']},{r['V_mid']:.2f},"
        f"{r['ra_deg_1']:.9f},{r['dec_deg_1']:.9f},"
        f"{r['ra_deg_2']:.9f},{r['dec_deg_2']:.9f},"
        f"{r['trail_len_arcsec']:.2f},{r['pa_deg_E_of_N']:.2f}"
    )

print()
print("# Screening-like rows")
print(
    "observation_id,sso_name,ra_deg_1,dec_deg_1,ra_deg_2,dec_deg_2,"
    "filter,xmatch_type,v_mag_1,v_mag_1_corrected,mlim_obs,FITS_FILE,DECISION"
)

for r in rows:
    # v_mag_1_corrected: apply your UVW1 SVO correction convention if desired.
    # In your current L batch, v_mag_1_corrected = v_mag_1 + 1.4842.
    v = r["V_mid"]
    v_corr = v + 1.4842 if math.isfinite(v) else math.nan

    print(
        f"{OBS_ID},{r['name']},"
        f"{r['ra_deg_1']:.9f},{r['dec_deg_1']:.9f},"
        f"{r['ra_deg_2']:.9f},{r['dec_deg_2']:.9f},"
        f"{FILTER},2,"
        f"{v:.2f},{v_corr:.4f},,"
        f"{FITS_FILE},D"
    )