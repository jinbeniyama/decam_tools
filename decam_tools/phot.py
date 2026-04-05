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
              radius=None, radius_list=None, rad_factor=1.5,
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
    print(f"  Seeing FWHM = {seeing_fwhm:.1f} arcsec")

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
    print(f"  Exposure time: {t_exp} s")
    print(f"  MAGZPT = {magzpt:.2f} mag")

    gaina = hdr0["GAINA"]
    gainb = hdr0["GAINB"]
    gain = (gaina + gainb) / 2.

    img = src[0].data.byteswap().newbyteorder()
    x0, y0 = coo

    # Background noise
    bkg = sep.Background(img)
    bgerr_pix = bkg.globalrms

    wi = wi / 2.0
    xmin, xmax = int(x0 - wi - 1), int(x0 + wi)
    ymin, ymax = int(y0 - wi - 1), int(y0 + wi)
    img_cut = img[ymin:ymax, xmin:xmax].astype(np.float32)
    xc, yc = wi, wi

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

    # --- Plot main image with top/right profiles ---
    fig = plt.figure(figsize=(6, 6))
    # main image axis
    ax_img = fig.add_axes([0.15, 0.15, 0.65, 0.65])
    ax_img.set_xlabel("x [pix]")
    ax_img.set_ylabel("y [pix]")
    # top profile
    ax_top = fig.add_axes([0.15, 0.82, 0.65, 0.1], sharex=ax_img)
    # right profile
    ax_right = fig.add_axes([0.82, 0.15, 0.1, 0.65], sharey=ax_img)

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
    ax_img.scatter(
        xc, yc, color="black", s=200, lw=2, marker="x", alpha=1, 
        label=f"{label} {mag_list[-1]:.2f} ± {magerr_list[-1]:.2f}")
    ax_img.grid(True, linestyle=':', alpha=0.6)

    # Draw apertures
    col_app = "black"
    col_ann = "black"
    for r_plot in radii:
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
    ax_right.plot(col_profile[::-1], y, color='gray')  # flip counts to match imshow
    ax_right.set_xlabel("Counts")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle=':', alpha=0.6)
    ax_right.invert_yaxis()


    # --- Overlay aperture/annulus lines on top/right profiles symmetrically ---
    colors = ['black', 'black', 'black']  # main aperture, inner annulus, outer annulus
    linestyles = ['dashed', 'solid', 'solid']

    # Top profile (x-direction)
    for r, c, ls in zip(radius_list, colors, linestyles):
        ax_top.vlines([xc - r, xc + r], ymin=row_profile.min(), ymax=row_profile.max(),
                    color=c, linestyle=ls, lw=1)

    # Annulus edges (outer)
    ax_top.vlines([xc - (radius_list[-1] + ann_gap),
                xc + (radius_list[-1] + ann_gap),
                xc - (radius_list[-1] + ann_gap + ann_width),
                xc + (radius_list[-1] + ann_gap + ann_width)],
                ymin=row_profile.min(), ymax=row_profile.max(),
                color=col_ann, linestyle='solid', lw=1)

    # Right profile (y-direction)
    for r, c, ls in zip(radius_list, colors, linestyles):
        ax_right.hlines([yc - r, yc + r], xmin=col_profile.min(), xmax=col_profile.max(),
                        color=c, linestyle=ls, lw=1)

    # Annulus edges (outer)
    ax_right.hlines([yc - (radius_list[-1] + ann_gap),
                    yc + (radius_list[-1] + ann_gap),
                    yc - (radius_list[-1] + ann_gap + ann_width),
                    yc + (radius_list[-1] + ann_gap + ann_width)],
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


def plot_color_curve_of_growth(df, label_prefix=""):
    """
    Plot g-r color curve of growth.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flux, fluxerr, mag, magerr, and radius
    label_prefix : str
        Label in legend
    """

    df_mag = df.pivot_table(
        index=["date", "radius"],
        columns="band",
        values="mag"
    ).reset_index()

    df_err = df.pivot_table(
        index=["date", "radius"],
        columns="band",
        values="magerr"
    ).reset_index()

    df_mag["g-r"] = df_mag["g"] - df_mag["r"]
    df_mag["g-r_err"] = np.sqrt(df_err["g"]**2 + df_err["r"]**2)

    # --- plot ---
    plt.figure(figsize=(6,5))

    for date, group in df_mag.groupby("date"):
        plt.errorbar(
            group["radius"],
            group["g-r"],
            yerr=group["g-r_err"],
            marker="o",
            label=f"{label_prefix}{date}"
        )

    plt.xlabel("Aperture radius [pix]")
    plt.ylabel("g-r")
    plt.grid(True)
    plt.legend()
    plt.show()
