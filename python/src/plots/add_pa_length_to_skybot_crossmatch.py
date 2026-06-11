#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
inp = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/input_super_bright_source_2.csv")
out = inp.with_name(inp.stem + "_with_pa_length.csv")


# ---------------------------------------------------------------------
# Observed bright trail measured in DS9
# pos1 = 11:00:33.9929, +09:56:56.735
# pos2 = 11:00:35.0002, +09:57:56.200
# ---------------------------------------------------------------------
OBS_RA1 = 165.141637083
OBS_DEC1 = 9.949093056
OBS_RA2 = 165.145834167
OBS_DEC2 = 9.965611111


def to_float(x):
    try:
        s = str(x or "").strip()
        if not s:
            return math.nan
        return float(s)
    except Exception:
        return math.nan


def motion_params(ra1, dec1, ra2, dec2):
    """
    Return approximate tangent-plane motion parameters:
    dra_arcsec is East-positive, ddec_arcsec is North-positive.
    PA is degrees East of North.
    """
    if not all(math.isfinite(v) for v in [ra1, dec1, ra2, dec2]):
        return math.nan, math.nan, math.nan, math.nan

    mean_dec = 0.5 * (dec1 + dec2)

    dra_arcsec = (ra2 - ra1) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec_arcsec = (dec2 - dec1) * 3600.0

    length = math.hypot(dra_arcsec, ddec_arcsec)

    pa = math.degrees(math.atan2(dra_arcsec, ddec_arcsec))
    if pa < 0:
        pa += 360.0

    return dra_arcsec, ddec_arcsec, length, pa


def angle_diff_deg(a, b):
    """
    Smallest absolute difference between two position angles.
    Result in [0, 180].
    """
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    return abs((a - b + 180.0) % 360.0 - 180.0)


obs_dra, obs_ddec, obs_len, obs_pa = motion_params(
    OBS_RA1, OBS_DEC1, OBS_RA2, OBS_DEC2
)

print(f"[OBS] dra_arcsec={obs_dra:.2f}")
print(f"[OBS] ddec_arcsec={obs_ddec:.2f}")
print(f"[OBS] trail_len_arcsec={obs_len:.2f}")
print(f"[OBS] pa_deg_E_of_N={obs_pa:.2f}")


extra_fields = [
    "dra_arcsec",
    "ddec_arcsec",
    "trail_len_arcsec",
    "pa_deg_E_of_N",
    "obs_len_diff_arcsec",
    "obs_pa_diff_deg",
    "obs_mid_dist_arcsec",
    "score_simple",
]


def midpoint(ra1, dec1, ra2, dec2):
    if not all(math.isfinite(v) for v in [ra1, dec1, ra2, dec2]):
        return math.nan, math.nan
    return 0.5 * (ra1 + ra2), 0.5 * (dec1 + dec2)


def sky_distance_arcsec(ra_a, dec_a, ra_b, dec_b):
    """
    Small-angle sky distance in arcsec.
    Good enough for this local field.
    """
    if not all(math.isfinite(v) for v in [ra_a, dec_a, ra_b, dec_b]):
        return math.nan

    mean_dec = 0.5 * (dec_a + dec_b)
    dra = (ra_b - ra_a) * math.cos(math.radians(mean_dec)) * 3600.0
    ddec = (dec_b - dec_a) * 3600.0
    return math.hypot(dra, ddec)


obs_mid_ra, obs_mid_dec = midpoint(OBS_RA1, OBS_DEC1, OBS_RA2, OBS_DEC2)

rows = []

with inp.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
    fieldnames = list(reader.fieldnames or [])

    for row in reader:
        ra1 = to_float(row.get("ra_deg_1"))
        dec1 = to_float(row.get("dec_deg_1"))
        ra2 = to_float(row.get("ra_deg_2"))
        dec2 = to_float(row.get("dec_deg_2"))

        dra, ddec, length, pa = motion_params(ra1, dec1, ra2, dec2)

        mid_ra, mid_dec = midpoint(ra1, dec1, ra2, dec2)

        len_diff = abs(length - obs_len) if math.isfinite(length) else math.nan
        pa_diff = angle_diff_deg(pa, obs_pa)
        mid_dist = sky_distance_arcsec(obs_mid_ra, obs_mid_dec, mid_ra, mid_dec)

        # Simple heuristic: lower is better.
        # PA is weighted strongly because direction was the main mismatch.
        if all(math.isfinite(v) for v in [len_diff, pa_diff, mid_dist]):
            score = mid_dist + 2.0 * len_diff + 5.0 * pa_diff
        else:
            score = math.nan

        row["dra_arcsec"] = f"{dra:.3f}" if math.isfinite(dra) else ""
        row["ddec_arcsec"] = f"{ddec:.3f}" if math.isfinite(ddec) else ""
        row["trail_len_arcsec"] = f"{length:.3f}" if math.isfinite(length) else ""
        row["pa_deg_E_of_N"] = f"{pa:.3f}" if math.isfinite(pa) else ""
        row["obs_len_diff_arcsec"] = f"{len_diff:.3f}" if math.isfinite(len_diff) else ""
        row["obs_pa_diff_deg"] = f"{pa_diff:.3f}" if math.isfinite(pa_diff) else ""
        row["obs_mid_dist_arcsec"] = f"{mid_dist:.3f}" if math.isfinite(mid_dist) else ""
        row["score_simple"] = f"{score:.3f}" if math.isfinite(score) else ""

        rows.append(row)

for col in extra_fields:
    if col not in fieldnames:
        fieldnames.append(col)

# Sort rows by score, best candidates first
def sort_key(row):
    s = to_float(row.get("score_simple"))
    return s if math.isfinite(s) else math.inf

rows.sort(key=sort_key)

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] Read rows: {len(rows)}")
print(f"[OK] Saved: {out}")

print()
print("[TOP 20 by score_simple]")
cols = [
    "sso_name",
    "v_mag_1",
    "trail_len_arcsec",
    "pa_deg_E_of_N",
    "obs_mid_dist_arcsec",
    "obs_len_diff_arcsec",
    "obs_pa_diff_deg",
    "score_simple",
]

print(",".join(cols))
for row in rows[:20]:
    print(",".join(str(row.get(c, "")) for c in cols))