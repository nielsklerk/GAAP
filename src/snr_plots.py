from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import warnings
import pandas as pd
import gc
import glob
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['image.origin'] = "lower"
plt.rcParams['image.cmap'] = 'magma'

storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files'
catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'
processed_file = "/net/vdesk/data2/deklerk/GAAP_data/processed.txt"

with open(processed_file, "r") as f:
    processed = set(line.strip() for line in f)

all_fluxes = None
stop_index = 1
total = 0

fig, axs = plt.subplots(
    2, 4,
    figsize=(12, 6),
    sharex=True,
    sharey=True,
    gridspec_kw={'wspace': 0, 'hspace': 0}
)

for outer_i, tile_index in enumerate(processed):
    gc.collect()
    print(outer_i)

    catalog_file = glob.glob(
        f'{catalog_folder}/EUC_MER_FINAL-CAT_TILE{tile_index}*.fits')[0]
    with fits.open(catalog_file, memmap=True) as hdul:
        cat = Table(hdul[1].data)

    if outer_i > stop_index:
        break

    fluxes = pd.read_csv(f'{storage_folder}/{tile_index}_fluxes.csv')
    FWHM = '1'
    cutoff_snr = 1

    # Optical filters
    for j, flt in enumerate(['G', 'R', 'I']):
        snr_meer = cat[f'FLUX_{flt}_EXT_DECAM_{FWHM}FWHM_APER'] / \
            cat[f'FLUXERR_{flt}_EXT_DECAM_{FWHM}FWHM_APER']
        snr_gaap_euclid = fluxes[f'DES-{flt}'] / fluxes[f'DES-{flt}_sigma']

        mask = (snr_meer > cutoff_snr) & (snr_gaap_euclid > cutoff_snr) & \
            np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

        ax = axs[0, j]
        ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                   s=.1, rasterized=True, c='b')
        ax.plot([cutoff_snr*0.1 + 1e-16, 1e6],
                [cutoff_snr*0.1 + 1e-16, 1e6], 'r--')
        ax.text(0.05, 0.95, f'DES-{flt}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(cutoff_snr*0.1 + 1, 5e5)
        ax.set_ylim(cutoff_snr*0.1 + 1, 5e5)

    # NIR filters
    for j, flt in enumerate(['H', 'J', 'Y']):
        snr_meer = cat[f'FLUX_{flt}_{FWHM}FWHM_APER'] / \
            cat[f'FLUXERR_{flt}_{FWHM}FWHM_APER']
        snr_gaap_euclid = fluxes[f'NIR-{flt}'] / fluxes[f'NIR-{flt}_sigma']

        mask = (snr_meer > cutoff_snr) & (snr_gaap_euclid > cutoff_snr) & \
            np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

        ax = axs[1, j]
        ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
                   s=.1, rasterized=True, c='b')
        ax.plot([cutoff_snr*0.1 + 1e-16, 1e6],
                [cutoff_snr*0.1 + 1e-16, 1e6], 'r--')
        ax.text(0.05, 0.95, f'NIR-{flt}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(cutoff_snr*0.1 + 1, 1e4)
        ax.set_ylim(cutoff_snr*0.1 + 1, 1e4)

    # VIS
    snr_meer = cat[f'FLUX_VIS_{FWHM}FWHM_APER'] / \
        cat[f'FLUXERR_VIS_{FWHM}FWHM_APER']
    snr_gaap_euclid = fluxes['VIS'] / fluxes['VIS_sigma']

    mask = (snr_meer > cutoff_snr) & (snr_gaap_euclid > cutoff_snr) & \
        np.isfinite(snr_gaap_euclid) & np.isfinite(snr_meer)

    ax = axs[1, 3]
    ax.scatter(snr_meer[mask], snr_gaap_euclid[mask],
               s=.1, rasterized=True, c='b')
    ax.plot([cutoff_snr*0.1 + 1e-16, 1e6],
            [cutoff_snr*0.1 + 1e-16, 1e6], 'r--')
    ax.text(0.05, 0.95, 'VIS', transform=ax.transAxes,
            ha='left', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(cutoff_snr*0.1 + 1, 1e4)
    ax.set_ylim(cutoff_snr*0.1 + 1, 1e4)

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
plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/test.pdf')
