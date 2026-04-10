#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
For photometry
'''
import numpy as np
import pandas as pd
from scipy.stats import sigmaclip
import sep
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


def phot_dcam_xy(f, coo, wi, ann_gap, ann_width,
              radius=None, radius_list=None, rad_nominal=None, rad_factor=1.5,
              label=None, fltr=None, out=None):
    """
    Perform photometry at input coordinates, with top/right count profiles.

    Parameters
    ----------
    f : str
        FITS file
    coo : tuple
        x and y coordinate
    wi : int
        size of cutout
    ann_gap : int
        gap between annulus and aperture
    ann_width : float
        radial length of annulus
    radius : float, optional
        aperture radius in pix
    radius_list : array-like, optional
        aperture radii in pix
    rad_factor : float
        Used if radius and radius_list are not given
    label : str
        label in the figure
    fltr : str
        color map indicator
    out : str
        output file name

    Returns
    -------
    df : pandas.DataFrame
        dataframe with radius, flux, fluxerr, mag, magerr
    """

    src = fits.open(f)
    hdr0 = src[0].header

    # Seeing FWHM
    seeing_fwhm = hdr0["FWHM"]
    print(f"  Seeing FWHM = {seeing_fwhm:.1f} pix")

    if radius_list is not None:
        radii = radius_list
    else:
        if radius is None:
            radius = seeing_fwhm * rad_factor
        radii = [radius]
 
    # Background annulus fixed outside max radius
    max_r = max(radii)
    bkgann = (max_r + ann_gap, max_r + ann_gap + ann_width)

    # Zeropoint
    magzpt = hdr0["MAGZPT"]
    t_exp = hdr0["EXPTIME"]
    UTC0 = hdr0["UTC0"]
    print(f"  Exposure time: {t_exp} s")
    print(f"  MAGZPT = {magzpt:.2f} mag")
    print(f"  UTC0 = {UTC0}")

    gaina = hdr0["GAINA"]
    gainb = hdr0["GAINB"]
    gain = (gaina + gainb) / 2.

    img = src[0].data.byteswap().newbyteorder()
    x0, y0 = coo
    half_wi = wi / 2.0

    xmin, xmax = int(np.floor(x0 - half_wi)), int(np.floor(x0 + half_wi))
    ymin, ymax = int(np.floor(y0 - half_wi)), int(np.floor(y0 + half_wi))

    # Original size
    ny, nx = img.shape

    x_start, x_end = max(0, xmin), min(nx, xmax)
    y_start, y_end = max(0, ymin), min(ny, ymax)

    # Extract existing pixels
    img_temp = img[y_start:y_end, x_start:x_end].astype(np.float32)

    pad_y_before = max(0, -ymin)
    pad_y_after  = max(0, ymax - ny)
    pad_x_before = max(0, -xmin)
    pad_x_after  = max(0, xmax - nx)


    img_cut = np.pad(img_temp,
                     ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
                     mode='constant',
                     constant_values=0)

    xc, yc = half_wi, half_wi

    sig = seeing_fwhm / 2.355  
    x_new, y_new, flag = sep.winpos(img_cut, [xc], [yc], sig)

    xc, yc = x_new[0], y_new[0]

    # Background noise
    bkg = sep.Background(img)
    bgerr_pix = bkg.globalrms


    # --- lists for photometry ---
    flux_list, fluxerr_list = [], []
    mag_list, magerr_list = [], []

    print()
    for r in radii:
        flux, fluxerr, eflag = sep.sum_circle(
            img_cut, [xc], [yc], r=r, err=bgerr_pix, gain=None, bkgann=bkgann)
        flux, fluxerr = float(flux[0]), float(fluxerr[0])

        Perr_target = (flux * gain)**0.5 / gain
        fluxerr = (fluxerr**2 + Perr_target**2)**0.5

        flux_sec = flux / t_exp
        fluxerr_sec = fluxerr / t_exp

        if flux_sec > 0 and fluxerr > 0:
            mag = magzpt - 2.5 * np.log10(flux_sec)
            magerr = 2.5 / np.log(10) * (fluxerr_sec / flux_sec)
        else:
            mag, magerr = np.nan, np.nan

        flux_list.append(flux)
        fluxerr_list.append(fluxerr)
        mag_list.append(mag)
        magerr_list.append(magerr)
        print(f"  r={r:.2f}  flux={flux:.1f}  mag={mag:.3f}")

        # Extract nominal value
        if r == rad_nominal:
            mag_nominal = mag
            magerr_nominal = magerr
            print("    -> This is nominal value in the figure")

    # --- Plot main image with top/right profiles ---
    fig = plt.figure(figsize=(6, 6))
    # main image axis
    ax_img = fig.add_axes([0.17, 0.15, 0.63, 0.63])
    ax_img.set_xlabel("x [pix]")
    ax_img.set_ylabel("y [pix]")
    # top profile
    ax_top = fig.add_axes([0.17, 0.82, 0.63, 0.1], sharex=ax_img)
    # right profile
    ax_right = fig.add_axes([0.82, 0.15, 0.1, 0.63], sharey=ax_img)


    sigma = 5
    _, vmin, vmax = sigmaclip(img_cut, sigma, sigma)

    if fltr is None:
        cmap = "inferno"
    elif fltr == "g":
        cmap = "Greens"
    elif fltr == "r":
        cmap = "Reds"
    elif fltr == "i":
        cmap = "PuRd"
    elif fltr == "z":
        cmap = "Purples"

    ax_img.imshow(img_cut, vmin=vmin, vmax=vmax, cmap=cmap)
    if rad_nominal is not None:
        label=f"{label} {mag_nominal:.2f} ± {magerr_nominal:.2f}"
    else:
        label=f"{label}"
    ax_img.scatter(
        xc, yc, color="black", s=200, lw=2, marker="x", alpha=1, label=label)
    ax_img.grid(True, linestyle=':', alpha=0.6)

    # Draw apertures
    col_app = "black"
    col_ann = "black"


    if rad_nominal is not None:
        # Show only nominal radius
        radii_plot = [rad_nominal]
    else:
        radii_plot = radii

    for r_plot in radii_plot:
        ax_img.add_collection(PatchCollection([Circle((xc, yc), r_plot)],
                                              color=col_app, ls="dashed", lw=1, facecolor="None"))

    ax_img.add_collection(PatchCollection([Circle((xc, yc), max_r + ann_gap)],
                                          color=col_ann, ls="solid", lw=1, facecolor="None"))
    ax_img.add_collection(PatchCollection([Circle((xc, yc), max_r + ann_gap + ann_width)],
                                          color=col_ann, ls="solid", lw=1, facecolor="None"))
    ax_img.legend().get_frame().set_alpha(1)

    # --- Top profile (average of 3 rows centered on yc) ---
    row_indices = np.arange(int(yc)-1, int(yc)+2)
    row_profile = img_cut[row_indices, :].mean(axis=0)
    x = np.arange(img_cut.shape[1])
    ax_top.plot(x, row_profile, color='gray')
    ax_top.set_ylabel("Counts")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, linestyle=':', alpha=0.6)

    # --- Right profile (average of 3 columns centered on xc) ---
    col_indices = np.arange(int(xc)-1, int(xc)+2)
    col_profile = img_cut[:, col_indices].mean(axis=1)
    y = np.arange(img_cut.shape[0])
    ax_right.plot(col_profile, y, color='gray')
    ax_right.set_xlabel("Counts")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle=':', alpha=0.6)
    ax_right.invert_yaxis()


    # --- Overlay aperture/annulus lines on top/right profiles symmetrically ---
    colors = ['black', 'black', 'black']  # main aperture, inner annulus, outer annulus
    linestyles = ['dashed', 'solid', 'solid']


    # Top profile (x-direction)
    for r, c, ls in zip(radii_plot, colors, linestyles):
        ax_top.vlines([xc - r, xc + r], ymin=row_profile.min(), ymax=row_profile.max(),
                    color=c, linestyle=ls, lw=1)

    # Annulus edges (outer)
    ax_top.vlines([xc - (radii[-1] + ann_gap),
                xc + (radii[-1] + ann_gap),
                xc - (radii[-1] + ann_gap + ann_width),
                xc + (radii[-1] + ann_gap + ann_width)],
                ymin=row_profile.min(), ymax=row_profile.max(),
                color=col_ann, linestyle='solid', lw=1)

    # Right profile (y-direction)
    for r, c, ls in zip(radii_plot, colors, linestyles):
        ax_right.hlines([yc - r, yc + r], xmin=col_profile.min(), xmax=col_profile.max(),
                        color=c, linestyle=ls, lw=1)

    # Annulus edges (outer)
    ax_right.hlines([yc - (radii[-1] + ann_gap),
                    yc + (radii[-1] + ann_gap),
                    yc - (radii[-1] + ann_gap + ann_width),
                    yc + (radii[-1] + ann_gap + ann_width)],
                    xmin=col_profile.min(), xmax=col_profile.max(),
                    color=col_ann, linestyle='solid', lw=1)
    if out:
        print(f"Save figure as {out}")
        plt.savefig(out)
    print()

    # --- DataFrame ---
    df = pd.DataFrame({
        "radius": radii,
        "flux": flux_list,
        "fluxerr": fluxerr_list,
        "mag": mag_list,
        "magerr": magerr_list
    })
    df["utc"] = UTC0

    return df


def plot_curve_of_growth(df, label="Target", show_flux=True):
    """
    Plot curve of growth.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flux, fluxerr, mag, magerr, and radius
    label : str
        Label in legend
    show_flux : bool
        Whether show flux or not (i.e., mag)
    """
    fig, ax = plt.subplots(figsize=(6,5))

    if show_flux:
        flux = df["flux"].values
        fluxerr = df["fluxerr"].values
        ax.errorbar(df["radius"], flux, yerr=fluxerr, fmt='o-', label=label)
        ax.set_ylabel("Flux [counts]")
        ax.set_yscale('log')
    else:
        mag = df["mag"].values
        magerr = df["magerr"].values
        ax.errorbar(df["radius"], mag, yerr=magerr, fmt='o-', label=label)
        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()

    ax.set_xlabel("Aperture radius [pix]")

    ax.minorticks_on()
    ax.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)

    ax.legend()
    plt.show()


def plot_color_curve_of_growth_mix(df, colors=["g-i", "g-z", "i-z"], label_prefix=""):
    """
    Plot color growth curves based on radius, even if bands were observed on different dates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'date', 'radius', 'band', 'mag', and 'magerr'.
    colors : list of str
        List of color indices to plot (e.g., ["g-i", "r-z"]).
    label_prefix : str
        Prefix for the legend labels.
    """
    # 1. Collect unique dates per band for the legend
    # Mapping each band to its observation date(s)
    date_info = df.groupby("band")["date"].unique().to_dict()

    # 2. Create pivot tables
    # Removing "date" from the index to align different bands by "radius" only.
    # If multiple dates exist for the same band/radius, they are averaged.
    pivot_mag = df.pivot_table(
        index="radius", columns="band", values="mag", aggfunc="mean"
    )
    pivot_err = df.pivot_table(
        index="radius", columns="band", values="magerr", aggfunc="mean"
    )

    n_colors = len(colors)
    fig, axes = plt.subplots(n_colors, 1, figsize=(7, 4 * n_colors), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, color_str in enumerate(colors):
        ax = axes[i]
        try:
            b1, b2 = color_str.split('-')
        except ValueError:
            print(f"Skipping invalid format: {color_str}")
            continue

        # Check if both required bands exist in the dataset
        if b1 not in pivot_mag.columns or b2 not in pivot_mag.columns:
            print(f"Skipping {color_str}: One or both bands not found in DataFrame.")
            continue

        # Calculate color index and propagated error
        # Subtraction is performed row-wise based on aligned radius
        color_val = pivot_mag[b1] - pivot_mag[b2]
        color_err = np.sqrt(pivot_err[b1]**2 + pivot_err[b2]**2)

        # Create a descriptive label (e.g., g(2019-02-14) - i(2019-02-27))
        d1 = date_info[b1][0] if b1 in date_info else "N/A"
        d2 = date_info[b2][0] if b2 in date_info else "N/A"
        custom_label = f"{label_prefix}{b1}({d1}) - {b2}({d2})"

        # Plotting the curve
        ax.errorbar(
            color_val.index,
            color_val.values,
            yerr=color_err.values,
            marker="o",
            markersize=5,
            capsize=3,
            label=custom_label
        )

        ax.set_ylabel(f"{color_str} [mag]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(f"Color Growth Curve: {color_str}")

    axes[-1].set_xlabel("Aperture radius [pix]")
    plt.tight_layout()
    plt.show()


def plot_color_curve_of_growth(df, colors=["g-i", "g-z", "i-z"], label_prefix=""):
    """
    Plot color growth curves for bands observed on the same date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'date', 'radius', 'band', 'mag', and 'magerr'.
    colors : list of str
        List of color indices to plot (e.g., ["g-r", "g-i"]).
    label_prefix : str
        Prefix for the legend labels.
    """
    # 1. Create pivot tables indexed by date and radius
    # This ensures that subtraction only occurs for data points sharing the exact same date
    pivot_mag = df.pivot_table(
        index=["date", "radius"], columns="band", values="mag"
    )
    pivot_err = df.pivot_table(
        index=["date", "radius"], columns="band", values="magerr"
    )

    n_colors = len(colors)
    fig, axes = plt.subplots(n_colors, 1, figsize=(7, 4 * n_colors), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, color_str in enumerate(colors):
        ax = axes[i]
        try:
            b1, b2 = color_str.split('-')
        except ValueError:
            print(f"Skipping invalid format: {color_str}")
            continue

        # Check if both required bands are available in the pivoted data
        if b1 not in pivot_mag.columns or b2 not in pivot_mag.columns:
            print(f"Skipping {color_str}: Bands not found in dataset.")
            continue

        # Calculate color index and propagated error
        # Resulting Series will still have (date, radius) as MultiIndex
        color_val = pivot_mag[b1] - pivot_mag[b2]
        color_err = np.sqrt(pivot_err[b1]**2 + pivot_err[b2]**2)

        # Combine values and reset index for easier grouping by date during plotting
        plot_df = color_val.to_frame(name="val").join(color_err.to_frame(name="err")).reset_index()

        # Group by date to plot a separate curve for each observation night
        for date, group in plot_df.groupby("date"):
            # Drop NaN rows to ensure a clean curve (in case a date is missing one of the bands)
            group = group.dropna(subset=["val"])
            if group.empty:
                continue
                
            ax.errorbar(
                group["radius"],
                group["val"],
                yerr=group["err"],
                marker="o",
                markersize=5,
                capsize=3,
                label=f"{label_prefix}{date}"
            )

        ax.set_ylabel(f"{color_str} [mag]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize='small', loc='best')
        ax.set_title(f"Same-Date Color Growth Curve: {color_str}")

    axes[-1].set_xlabel("Aperture radius [pix]")
    plt.tight_layout()
    plt.show()
