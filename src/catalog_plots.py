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


def plot_colors(filters: list, all_fluxes, ax, point_like_cutoff=0.9, error_cutoff=0.1, bins=500,
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
        y_color)) & (gaap_error < error_cutoff**2)

    images = {}
    if plot_pointlike:
        mask = (all_fluxes['point_like_prob_mer']
                >= point_like_cutoff) & base_mask
        H, xedges, yedges = np.histogram2d(
            x_color[mask], y_color[mask], bins=bins)
        im = ax.pcolormesh(
            xedges,
            yedges,
            H.T,
            # origin='lower',
            # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            shading='auto',
            norm=LogNorm(),
            rasterized=True
        )

        print(np.sum(mask))
        images['pointlike_im'] = (im, np.sum(mask))

    if plot_extended:
        mask = (all_fluxes['point_like_prob_mer']
                < point_like_cutoff) & base_mask
        H, xedges, yedges = np.histogram2d(
            x_color[mask], y_color[mask], bins=bins)
        im = ax.pcolormesh(
            xedges,
            yedges,
            H.T,
            # origin='lower',
            # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            shading='auto',
            norm=LogNorm(),
            rasterized=True,
            cmap='viridis'
        )
        print(np.sum(mask))
        images['extended_im'] = (im, np.sum(mask))

    ax.set_xlabel(f'{filters[0]} - {filters[1]}')
    ax.set_ylabel(f'{filters[2]} - {filters[3]}')
    return images


def plot_gaap(filters: list, ax, point_like_cutoff=0.9, error_cutoff=0.1, bins=500,
              plot_extended=True,
              plot_pointlike=True,
              storage_folder='/net/vdesk/data2/deklerk/GAAP_data/flux_files'):
    all_fluxes = pd.read_pickle(f'{storage_folder}/all_fluxes.pkl')
    images = plot_colors(filters, all_fluxes, ax, point_like_cutoff, error_cutoff, bins,
                         plot_extended,
                         plot_pointlike)
    return images


def plot_catalog(filters: list, ax, FWHM='4', stop_index=10, point_like_cutoff=0.9, error_cutoff=0.1, bins=500,
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
        print(i)
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

    images = plot_colors(filters, all_fluxes, ax, point_like_cutoff, error_cutoff, bins,
                         plot_extended,
                         plot_pointlike)
    return images


def plot_color_color_gaap_vs_mer(filters, plot_limits, n_bins, error_cutoff=0.1,
                                 stop_index=400,
                                 plot_extended=True,
                                 plot_pointlike=True,
                                 PLOT_GAAP=True,
                                 PLOT_CATALOG=True,):
    filter_1, filter_2, filter_3, filter_4 = filters
    xmin, xmax, ymin, ymax = plot_limits
    x_bins = np.linspace(xmin, xmax, n_bins+1)
    y_bins = np.linspace(ymin, ymax, n_bins+1)
    bins = (x_bins, y_bins)
    if PLOT_GAAP and PLOT_CATALOG:
        fig, axs = plt.subplots(1, 2,
                                figsize=(12, 6),
                                sharex=True,
                                sharey=True,
                                gridspec_kw={'wspace': 0, 'hspace': 0})
        images = plot_gaap([filter_1, filter_2, filter_3,
                            filter_4], ax=axs[0], error_cutoff=error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)

        images = plot_catalog([filter_1, filter_2, filter_3,
                               filter_4], ax=axs[1], stop_index=stop_index, error_cutoff=error_cutoff, plot_pointlike=plot_pointlike, plot_extended=plot_extended, bins=bins)
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
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_gaap_catalog_cutoff_{error_cutoff}.pdf',
                    bbox_inches='tight')
    elif PLOT_GAAP:
        fig, ax = plt.subplots()
        images = plot_gaap([filter_1, filter_2, filter_3,
                            filter_4], ax=ax, error_cutoff=error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_title('GAAP')
        fig.tight_layout()
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_gaap_cutoff_{error_cutoff}.pdf',
                    bbox_inches='tight')
    elif PLOT_CATALOG:
        fig, ax = plt.subplots()
        images = plot_catalog([filter_1, filter_2, filter_3,
                               filter_4], ax=ax, stop_index=stop_index, error_cutoff=error_cutoff, plot_extended=plot_extended, plot_pointlike=plot_pointlike, bins=bins)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_title('MER')
        fig.tight_layout()
        plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_3[-1]}{filter_4[-1]}_catalog_cutoff_{error_cutoff}.pdf',
                    bbox_inches='tight')


def plot_snr_comparison():
    storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files'
    catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'
    processed_file = "/net/vdesk/data2/deklerk/GAAP_data/processed.txt"

    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)

    all_fluxes = None
    stop_index = 3
    total = 0

    fig, axs = plt.subplots(
        3, 5,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )
    FWHM = '4'
    cutoff_snr = 1e-3
    DES_PLOTTED = False
    OTHER_PLOTTED = False
    processed = ['102022978', '102158274']
    for outer_i, tile_index in enumerate(processed):
        gc.collect()
        print(tile_index)

        catalog_file = glob.glob(
            f'{catalog_folder}/EUC_MER_FINAL-CAT_TILE{tile_index}*.fits')[0]
        with fits.open(catalog_file, memmap=True) as hdul:
            cat = Table(hdul[1].data)

        if DES_PLOTTED and OTHER_PLOTTED:
            break

        fluxes = pd.read_csv(f'{storage_folder}/{tile_index}_fluxes.csv')

        try:
            for j, flt in enumerate(['G', 'R', 'I', 'Z']):
                snr_meer = cat[f'FLUX_{flt}_EXT_DECAM_{FWHM}FWHM_APER'] / \
                    cat[f'FLUXERR_{flt}_EXT_DECAM_{FWHM}FWHM_APER']
                snr_gaap_euclid = fluxes[f'DES-{flt}'] / \
                    fluxes[f'DES-{flt}_sigma']

                mask = (snr_meer + snr_gaap_euclid > cutoff_snr) & \
                    np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

                ax = axs[1, j]
                ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                           s=.1, rasterized=True, c='b')
                ax.plot([cutoff_snr*0.1 + 1e-6, 1e6],
                        [cutoff_snr*0.1 + 1e-6, 1e6], 'r--')
                ax.text(0.05, 0.95, f'DES-{flt}', transform=ax.transAxes,
                        ha='left', va='top', fontsize=15,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                print('des, ', tile_index)
                DES_PLOTTED = True
        except:
            columns = {'U_EXT_MEGACAM': 'CFIS-U', 'G_EXT_HSC': 'WISHES-G',
                       'R_EXT_MEGACAM': 'CFIS-R', 'I_EXT_PANSTARRS': 'PANSTARRS-I', 'Z_EXT_HSC': 'WISHES-Z'}
            for j, flt in enumerate(['U_EXT_MEGACAM', 'G_EXT_HSC', 'R_EXT_MEGACAM', 'I_EXT_PANSTARRS', 'Z_EXT_HSC']):
                snr_meer = cat[f'FLUX_{flt}_{FWHM}FWHM_APER'] / \
                    cat[f'FLUXERR_{flt}_{FWHM}FWHM_APER']
                snr_gaap_euclid = fluxes[f'{columns[flt]}'] / \
                    fluxes[f'{columns[flt]}_sigma']

                mask = (snr_meer + snr_gaap_euclid > cutoff_snr) & \
                    np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

                ax = axs[0, j]
                ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                           s=.1, rasterized=True, c='b')
                ax.plot([cutoff_snr*0.1 + 1e-6, 1e6],
                        [cutoff_snr*0.1 + 1e-6, 1e6], 'r--')
                ax.text(0.05, 0.95, f'{columns[flt]}', transform=ax.transAxes,
                        ha='left', va='top', fontsize=15,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                print('other, ', tile_index)
                OTHER_PLOTTED = True

            # NIR filters
            for j, flt in enumerate(['H', 'J', 'Y']):
                snr_meer = cat[f'FLUX_{flt}_{FWHM}FWHM_APER'] / \
                    cat[f'FLUXERR_{flt}_{FWHM}FWHM_APER']
                snr_gaap_euclid = fluxes[f'NIR-{flt}'] / \
                    fluxes[f'NIR-{flt}_sigma']

                mask = (snr_meer + snr_gaap_euclid > cutoff_snr) & \
                    np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

                ax = axs[2, j]
                ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                           s=.1, rasterized=True, c='b')
                ax.plot([cutoff_snr*0.1 + 1e-6, 1e6],
                        [cutoff_snr*0.1 + 1e-6, 1e6], 'r--')
                ax.text(0.05, 0.95, f'NIR-{flt}', transform=ax.transAxes,
                        ha='left', va='top', fontsize=15,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # VIS
            snr_meer = cat[f'FLUX_VIS_{FWHM}FWHM_APER'] / \
                cat[f'FLUXERR_VIS_{FWHM}FWHM_APER']
            snr_gaap_euclid = fluxes['VIS'] / fluxes['VIS_sigma']

            mask = (snr_meer + snr_gaap_euclid > cutoff_snr) & \
                np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

            ax = axs[2, 3]
            ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                       s=.1, rasterized=True, c='b')
            ax.plot([cutoff_snr*0.1 + 1e-16, 1e6],
                    [cutoff_snr*0.1 + 1e-16, 1e6], 'r--')
            ax.text(0.05, 0.95, f'VIS', transform=ax.transAxes,
                    ha='left', va='top', fontsize=15,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(cutoff_snr*0.1 + 1e-6, 1e6)
            ax.set_ylim(cutoff_snr*0.1 + 1e-6, 1e6)

    # Labels only on outer axes
    for ax in axs[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axs[:, 1:].flatten():
        ax.tick_params(labelleft=False)

    # Remove individual axis labels
    for ax in axs.flat:
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Single shared labels centered on the figure
    fig.supxlabel('MER SNR')
    fig.supylabel('GAAP SNR')

    # Ensure zero spacing (in case tight_layout reintroduces gaps)
    fig.subplots_adjust(wspace=0, hspace=0)

    # plt.show()
    plt.savefig(
        f'/home/deklerk/GAAP/results/figures/analysis/snr_comparison.pdf')


def plot_residual(filters):
    filter_1, filter_2 = filters
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
    storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files'
    catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'
    processed_file = "/net/vdesk/data2/deklerk/GAAP_data/processed.txt"

    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)

    fig, ax = plt.subplots()
    FWHM = '4'
    cutoff_snr = 1e-3
    color_vs_color = True
    OTHER_PLOTTED = False
    processed = ['102022978', '102158274']
    for outer_i, tile_index in enumerate(processed):
        gc.collect()
        print(tile_index)

        catalog_file = glob.glob(
            f'{catalog_folder}/EUC_MER_FINAL-CAT_TILE{tile_index}*.fits')[0]
        with fits.open(catalog_file, memmap=True) as hdul:
            cat = Table(hdul[1].data)

        fluxes = pd.read_csv(f'{storage_folder}/{tile_index}_fluxes.csv')

        try:
            gaap_color = -2.5 * \
                np.log10(fluxes[f'{filter_1}']/fluxes[f'{filter_2}'])
            mer_color = -2.5 * np.log10(np.array(cat[f'FLUX_{filter_to_catalog_name[filter_1]}_{FWHM}FWHM_APER'].data, dtype='<f8')/np.array(
                cat[f'FLUX_{filter_to_catalog_name[filter_2]}_{FWHM}FWHM_APER'].data, dtype='<f8'))
            x = np.array(cat[f'FLUX_VIS_{FWHM}FWHM_APER'].data, dtype='<f8')
            y = gaap_color - mer_color
            x = mer_color
            y = gaap_color
            mask = np.isfinite(x) & np.isfinite(y)
            xmin, xmax, ymin, ymax = 1e-8, 1e5, -15, 15
            n_bins = 500
            x_bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins+1)
            y_bins = np.linspace(ymin, ymax, n_bins+1)
            xmin, xmax, ymin, ymax = -5, 5, -5, 5
            n_bins = 100
            x_bins = np.linspace(xmin, xmax, n_bins+1)
            y_bins = np.linspace(ymin, ymax, n_bins+1)
            bins = (x_bins, y_bins)
            H, xedges, yedges = np.histogram2d(
                x[mask], y[mask], bins=bins)
            im = ax.pcolormesh(
                xedges,
                yedges,
                H.T,
                # origin='lower',
                # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                shading='auto',
                norm=LogNorm(),
                rasterized=True
            )
            ax.plot([xmin, xmax], [xmin, xmax], 'r--')
            break
            # ax.scatter(x[mask], y[mask], rasterized=True, s=.1)
        except Exception as e:
            print(e)
            continue

    # ax.set_xscale('log')
    ax.set_xlabel('VIS Flux [uJy] (MER)')
    ax.set_ylabel(f'({filter_1} - {filter_2})' + r'$_\mathrm{GAAP}$' +
                  f'-({filter_1} - {filter_2})' + r'$_\mathrm{MER}$')
    # ax.axhline(0, c='r', linestyle='--')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # plt.show()
    plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/test.pdf')


def main():
    # plot_color_color_gaap_vs_mer(
    #     ['DES-G', 'DES-R', 'DES-R', 'DES-I'], [-.5, 2.5, -0.5, 2.5], 500, plot_extended=False)
    # plot_color_color_gaap_vs_mer(
    #     ['DES-G', 'NIR-J', 'NIR-J', 'NIR-H'], [-1, 6, -0.5, 1], 500)
    # plot_snr_comparison()
    plot_residual(['NIR-J', 'NIR-H'])


if __name__ == '__main__':
    main()
