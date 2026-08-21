#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch


# ---------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------
C1_CSV = Path(
    "/Users/eracero/workspace/tfm-viu/results/batch_B/elena/c1corr/"
    "c1_details_Lucubratio_P0110980101OMS402SIMAGE1000.csv"
)

FITS_PATH = Path(
    "/Users/eracero/workspace/tfm-viu/temp/0110980101/B/"
    "P0110980101OMS402SIMAGE1000.FTZ"
)

OUT_PNG = Path(
    "/Users/eracero/workspace/tfm-viu/results/batch_B/elena/screenshots/"
    "P0110980101OMS402SIMAGE1000__Lucubratio_C1_boxes.png"
)


def draw_rotated_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    theta_rad: float,
    *,
    edgecolor: str,
    linewidth: float = 1.4,
    linestyle: str = "-",
    label: str | None = None,
):
    """
    Draw a rotated rectangle centred at (x, y).

    width  = trail length direction
    height = trail/cross-trail aperture height
    theta_rad = angle of width axis, radians
    """
    rect = Rectangle(
        (-width / 2.0, -height / 2.0),
        width,
        height,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )

    trans = Affine2D().rotate(theta_rad).translate(x, y) + ax.transData
    rect.set_transform(trans)
    ax.add_patch(rect)
    return rect


def main():
    df = pd.read_csv(C1_CSV)

    if not FITS_PATH.exists():
        raise FileNotFoundError(f"FITS not found: {FITS_PATH}")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(FITS_PATH) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)

    data = np.where(np.isfinite(data), data, np.nan)

    norm = ImageNormalize(
        data,
        interval=ZScaleInterval(),
        stretch=AsinhStretch(),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        norm=norm,
        interpolation="nearest",
    )

    # Draw C1 boxes.
    # Small box: selected trail height h.
    # Large box: standard 6-arcsec-equivalent height.
    first_small = True
    first_large = True

    for _, r in df.iterrows():
        slot = int(r["slot"])
        x = float(r["x"])
        y = float(r["y"])
        width = float(r["width_pix"])
        theta = float(r["theta_rad"])
        h_small = float(r["height_h_pix"])
        h6 = float(r["height6_pix"])

        draw_rotated_box(
            ax,
            x,
            y,
            width,
            h6,
            theta,
            edgecolor="tab:orange",
            linewidth=1.4,
            linestyle="--",
            label="6 arcsec-equivalent box" if first_large else None,
        )
        first_large = False

        draw_rotated_box(
            ax,
            x,
            y,
            width,
            h_small,
            theta,
            edgecolor="tab:cyan",
            linewidth=1.8,
            linestyle="-",
            label="Selected trail-height box" if first_small else None,
        )
        first_small = False

        ax.plot(x, y, marker="+", color="tab:red", markersize=8, mew=1.5)
        ax.text(
            x + 5,
            y + 5,
            f"C1-{slot}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
        )

    target = str(df["target"].iloc[0]) if "target" in df.columns else "Lucubratio"
    obs_id = str(df["obs_id"].iloc[0]).zfill(10) if "obs_id" in df.columns else "0110980101"
    filt = str(df["filter"].iloc[0]) if "filter" in df.columns else "B"
    fits_name = str(df["fits_name"].iloc[0]) if "fits_name" in df.columns else FITS_PATH.name

    c1_med = np.nanmedian(df["c1"].astype(float))
    h_small = float(df["height_h_pix"].iloc[0])
    h6 = float(df["height6_pix"].iloc[0])

    ax.set_title(
        f"{target} | OBSID {obs_id} | {filt} | {fits_name}\n"
        rf"$C_1$ calibration stars: h={h_small:.1f} pix, h$_{{6''}}$={h6:.1f} pix, "
        rf"median $C_1$={c1_med:.3f}",
        fontsize=10,
    )

    ax.set_xlabel("X [pix]")
    ax.set_ylabel("Y [pix]")
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(-0.5, data.shape[0] - 0.5)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()