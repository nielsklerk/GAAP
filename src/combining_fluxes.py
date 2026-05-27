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

MAKE_NEW_COMBINED_FILE = False

storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files'
catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'
processed_file = "/net/vdesk/data2/deklerk/GAAP_data/processed.txt"
filename_file = '/home/deklerk/GAAP/src/EUCLID_ARCHIVE_files.pkl'

filter = 'DES-G'
if MAKE_NEW_COMBINED_FILE:
    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)

    all_fluxes = None
    stop_index = 400
    total = 0
    filenames = pd.read_pickle(filename_file)
    tile_indeces = filenames.index.tolist()
    for i, tile_index in enumerate(processed):
        if filter not in filenames.loc[tile_index]['FILTER']:
            continue
        print(i)
        if i > stop_index:
            break
        if all_fluxes is None:
            all_fluxes = pd.read_csv(
                f'{storage_folder}/{tile_index}_fluxes.csv')
        else:
            fluxes = pd.read_csv(f'{storage_folder}/{tile_index}_fluxes.csv')
            all_fluxes = pd.concat([all_fluxes, fluxes], ignore_index=True)
        gc.collect()
    all_fluxes.to_pickle(f'{storage_folder}/all_fluxes_DES.pkl')
else:
    all_fluxes = pd.read_pickle(f'{storage_folder}/all_fluxes_DES.pkl')
    # gc.collect()
    print(len(all_fluxes['tile_index']))

# filter_1 = 'DES-G'   # e.g., master_table column name
# filter_2 = 'DES-R'
# filter_3 = 'DES-R'
# filter_4 = 'DES-I'

# plot_error = True

# # Compute colors from master_table fluxes
# x_color = -2.5 * np.log10(all_fluxes[filter_1] / all_fluxes[filter_2])
# y_color = -2.5 * np.log10(all_fluxes[filter_3] / all_fluxes[filter_4])

# # Compute errors from sigma dictionary
# xerr = 2.5 / np.log(10) * np.sqrt(
#     (all_fluxes[f'{filter_1}_sigma'] / all_fluxes[filter_1])**2 +
#     (all_fluxes[f'{filter_2}_sigma'] / all_fluxes[filter_2])**2
# )
# yerr = 2.5 / np.log(10) * np.sqrt(
#     (all_fluxes[f'{filter_3}_sigma'] / all_fluxes[filter_3])**2 +
#     (all_fluxes[f'{filter_4}_sigma'] / all_fluxes[filter_4])**2
# )

# gaap_error = xerr**2 + yerr**2
# point_like_cutoff = 0.9
# gaap_error_cutoff = 0.5
# bins = 500
# plot_extended = False
# plot_pointlike = True
# left, right, bottom, top = -.5, 2.5, -.5, 2.5

# base_mask = (np.isfinite(x_color)) & (np.isfinite(y_color)) & (gaap_error < gaap_error_cutoff**2) & (
#     x_color > left) & (x_color < right) & (y_color > bottom) & (y_color < top)

# plt.figure(figsize=(10, 6))
# if plot_pointlike:
#     mask = (all_fluxes['point_source_probability_mer'] >= point_like_cutoff) & base_mask
#     H, xedges, yedges = np.histogram2d(x_color[mask], y_color[mask], bins=bins)
#     im = plt.imshow(
#         H.T,
#         origin='lower',
#         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#         aspect='auto',
#         norm=LogNorm()
#     )

#     print(np.sum(mask))
#     cbar = plt.colorbar(im)
#     cbar.set_label("Number of Point-Like Sources")

# if plot_extended:
#     mask = (all_fluxes['point_source_probability_mer'] < point_like_cutoff) & base_mask
#     H, xedges, yedges = np.histogram2d(x_color[mask], y_color[mask], bins=bins)
#     im = plt.imshow(
#         H.T,
#         origin='lower',
#         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#         aspect='auto',
#         norm=LogNorm(),
#         cmap='viridis'
#     )
#     print(np.sum(mask))
#     cbar = plt.colorbar(im)
#     cbar.set_label("Number of Extended Sources")
# plt.xlabel(f'{filter_1} - {filter_2}')
# plt.ylabel(f'{filter_3} - {filter_4}')
# plt.xlim(left, right)
# plt.ylim(bottom, top)
# plt.tight_layout()
# plt.show()
# plt.savefig(f'/home/deklerk/GAAP/results/figures/analysis/{filter_1[-1]}{filter_2[-1]}{filter_4[-1]}.pdf',
#             bbox_inches='tight')
