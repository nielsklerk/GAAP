from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import warnings
import pandas as pd
import gc
import glob
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['image.origin'] = "lower"
plt.rcParams['image.cmap'] = 'magma'


def plot_colors(filters: list, all_fluxes, ax, point_like_cutoff=0.9, gaap_error_cutoff=0.1, bins=500,
                plot_extended=True,
                plot_pointlike=True):
    # Compute colors from master_table fluxes
    x_color = -2.5 * \
        np.log10(all_fluxes[filters[0]] / all_fluxes[filters[1]])
    y_color = -2.5 * \
        np.log10(all_fluxes[filters[2]] / all_fluxes[filters[3]])

    # Compute errors from sigma dictionary
    xerr = 2.5 / np.log(10) * np.sqrt(
        (all_fluxes[f'{filters[0]}_sigma'] / all_fluxes[filters[0]])**2 +
        (all_fluxes[f'{filters[1]}_sigma'] / all_fluxes[filters[1]])**2
    )
    yerr = 2.5 / np.log(10) * np.sqrt(
        (all_fluxes[f'{filters[2]}_sigma'] / all_fluxes[filters[2]])**2 +
        (all_fluxes[f'{filters[3]}_sigma'] / all_fluxes[filters[3]])**2
    )

    gaap_error = xerr**2 + yerr**2

    base_mask = (np.isfinite(x_color)) & (np.isfinite(
        y_color)) & (gaap_error < gaap_error_cutoff**2)

    images = {}
    if plot_pointlike:
        mask = (all_fluxes['point_like_prob_mer']
                >= point_like_cutoff) & base_mask
        H, xedges, yedges = np.histogram2d(
            x_color[mask], y_color[mask], bins=bins)
        im = ax.imshow(
            H.T,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto',
            norm=LogNorm()
        )

        print(np.sum(mask))
        images['pointlike_im'] = (im, np.sum(mask))

    if plot_extended:
        mask = (all_fluxes['point_like_prob_mer']
                < point_like_cutoff) & base_mask
        H, xedges, yedges = np.histogram2d(
            x_color[mask], y_color[mask], bins=bins)
        im = ax.imshow(
            H.T,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto',
            norm=LogNorm(),
            cmap='viridis'
        )
        print(np.sum(mask))
        images['extended_im'] = (im, np.sum(mask))

    ax.set_xlabel(f'{filters[0]} - {filters[1]}')
    ax.set_ylabel(f'{filters[2]} - {filters[3]}')
    return images


def plot_gaap(filters: list, ax, point_like_cutoff=0.9, gaap_error_cutoff=0.1, bins=500,
              plot_extended=True,
              plot_pointlike=True,
              storage_folder='/net/vdesk/data2/deklerk/GAAP_data/flux_files'):
    all_fluxes = pd.read_pickle(f'{storage_folder}/all_fluxes.pkl')
    images = plot_colors(filters, all_fluxes, ax, point_like_cutoff, gaap_error_cutoff, bins,
                         plot_extended,
                         plot_pointlike)
    return images


def plot_catalog(filters: list, ax, FWHM='4', stop_index=10, point_like_cutoff=0.9, gaap_error_cutoff=0.1, bins=500,
                 plot_extended=True,
                 plot_pointlike=True,
                 processed_file="/net/vdesk/data2/deklerk/GAAP_data/processed.txt", catalog_folder='/net/vdesk/data2/deklerk/GAAP_data/catalog_files'):
    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)

    filter_to_catalog_name = {
        'CFIS-U': 'U_EXT_MEGACAM',
        'WISHES-G': 'G_EXT_HSC',
        'CFIS-R': 'R_EXT_MEGACAM',
        'PANSTARRS-I': 'I_EXT_PANSTARRS',
        'WISHES-Z': 'Z_EXT_HSC',
        'DES-G': 'G_EXT_DECAM',
        'DES-R': 'R_EXT_DECAM',
        'DES-I': 'I_EXT_DECAM',
        'DES-Z': 'Z_EXT_DECAM',
        'NIR-Y': 'Y',
        'NIR-J': 'J',
        'NIR-H': 'H',
        'VIS': 'VIS'
    }

    all_fluxes = None
    for i, tile_index in enumerate(processed):
        df = {}
        if i > stop_index:
            break
        gc.collect()

        catalog_file = glob.glob(
            f'{catalog_folder}/EUC_MER_FINAL-CAT_TILE{tile_index}*.fits')[0]
        with fits.open(catalog_file, memmap=True) as hdul:
            cat = Table(hdul[1].data)

        for filter in filters:
            df[filter] = np.array(
                cat[f'FLUX_{filter_to_catalog_name[filter]}_{FWHM}FWHM_APER'].data, dtype='<f8')
            df[filter + '_sigma'] = np.array(
                cat[f'FLUXERR_{filter_to_catalog_name[filter]}_{FWHM}FWHM_APER'].data, dtype='<f8')
        df['point_like_prob_mer'] = np.array(
            cat['POINT_LIKE_PROB'].data, dtype='<f8')

        if all_fluxes is None:
            all_fluxes = pd.DataFrame(df)
        else:
            fluxes = pd.DataFrame(df)
            all_fluxes = pd.concat([all_fluxes, fluxes], ignore_index=True)

    images = plot_colors(filters, all_fluxes, ax, point_like_cutoff, gaap_error_cutoff, bins,
                         plot_extended,
                         plot_pointlike)
    return images


def main():
    filter_1, filter_2, filter_3, filter_4 = 'DES-G', 'NIR-J', 'NIR-J', 'NIR-H'
    xmin, xmax, ymin, ymax = -1, 6, -0.5, 1
    x_bins = np.linspace(xmin, xmax, 501)
    y_bins = np.linspace(ymin, ymax, 501)
    bins = (x_bins, y_bins)
    gaap_error_cutoff = 0.1
    stop_index = 400
    plot_extended = True
    plot_pointlike = True
    PLOT_GAAP = True
    PLOT_CATALOG = True
    if PLOT_GAAP and PLOT_CATALOG:
        fig, axs = plt.subplots(1, 2,
                                figsize=(12, 6),
                                sharex=True,
                                sharey=True,
                                gridspec_kw={'wspace': 0, 'hspace': 0})
        images = plot_gaap([filter_1, filter_2, filter_3,
                            filter_4], ax=axs[0], gaap_error_cutoff=gaap_error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)

        images = plot_catalog([filter_1, filter_2, filter_3,
                               filter_4], ax=axs[1], stop_index=stop_index, gaap_error_cutoff=gaap_error_cutoff, plot_pointlike=plot_pointlike, plot_extended=plot_extended, bins=bins)
        axs[1].set_xlim(xmin, xmax)
        axs[1].set_ylim(ymin, ymax)

        # Labels only on outer axes
        axs[1].set_ylabel('')
        axs[1].tick_params(labelleft=False)
        for ax in axs:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

        axs[0].set_title('GAAP')
        axs[1].set_title('MER')
        fig.tight_layout()
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_gaap_catalog_cutoff_{gaap_error_cutoff}.pdf',
                    bbox_inches='tight')
    elif PLOT_GAAP:
        fig, ax = plt.subplots()
        images = plot_gaap([filter_1, filter_2, filter_3,
                            filter_4], ax=ax, gaap_error_cutoff=gaap_error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_title('GAAP')
        fig.tight_layout()
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_gaap_cutoff_{gaap_error_cutoff}.pdf',
                    bbox_inches='tight')
    elif PLOT_GAAP:
        fig, ax = plt.subplots()
        images = plot_catalog([filter_1, filter_2, filter_3,
                               filter_4], ax=ax, stop_index=stop_index, gaap_error_cutoff=gaap_error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_title('MER')
        fig.tight_layout()
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_catalog_cutoff_{gaap_error_cutoff}.pdf',
                    bbox_inches='tight')


if __name__ == '__main__':
    main()
