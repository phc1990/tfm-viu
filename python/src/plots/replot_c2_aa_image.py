from pathlib import Path
import pandas as pd

from src.photometry.hdu import HDUW
from src.photometry.ui import UI
from src.photometry.phot import arcsec_to_pix


# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
CSV = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/c2_details.csv")
FTZ = Path("/Users/eracero/workspace/tfm-viu/temp/0781040101/L/P0781040101OMS010FSIMAGL000.FTZ")
OUT = Path("/Users/eracero/workspace/tfm-viu/results/batch_L/elena/P0781040101OMS010FSIMAGL000_c2_clean.png")

N_STARS = 15

# Geometría C2 final
R6 = 6.0
R35 = 35.0
ANN6 = 6.0
ANN35 = 10.0


# ------------------------------------------------------------------
# Recover final saved selections
# ------------------------------------------------------------------
df = pd.read_csv(CSV)

rows = df[
    df["ftz"].apply(lambda p: Path(str(p)).name) == FTZ.name
].tail(N_STARS).copy()

if len(rows) != N_STARS:
    raise RuntimeError(
        f"Expected {N_STARS} C2 stars for {FTZ.name}, found {len(rows)}"
    )

print(rows[["slot", "x", "y", "snr35", "c2"]])


# ------------------------------------------------------------------
# Load image and reproduce C2 overlays
# ------------------------------------------------------------------
hduw = HDUW(file=str(FTZ))
ui = UI(hduw)

height6_pix = float(arcsec_to_pix(hduw, 2.0 * R6))
height35_pix = float(arcsec_to_pix(hduw, 2.0 * R35))
semi6_pix = float(arcsec_to_pix(hduw, ANN6))
semi35_pix = float(arcsec_to_pix(hduw, ANN35))

for _, row in rows.iterrows():
    ui._draw_c2_overlays_at(
        x=float(row["x"]),
        y=float(row["y"]),
        width=float(row["width_pix"]),
        theta=float(row["theta_rad"]),
        height6=height6_pix,
        semi6=semi6_pix,
        height35=height35_pix,
        semi35=semi35_pix,
    )

    # Optional: number the calibration stars
    ui.ax.text(
        float(row["x"]),
        float(row["y"]),
        f'{int(row["slot"])}',
        fontsize=7,
        ha="center",
        va="center",
    )


# ------------------------------------------------------------------
# Clean figure for the paper
# ------------------------------------------------------------------
ui.fig.suptitle("")
ui.ax.set_title("")

# remove interactive helper text if any
if getattr(ui, "_hint", None) is not None:
    try:
        ui._hint.remove()
    except Exception:
        pass

ui.fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)

print(f"Saved: {OUT}")