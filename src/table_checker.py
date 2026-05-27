import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import pandas as pd
warnings.filterwarnings("ignore")
#('OBJECT_ID',
# 'CONFIGURATION',
# 'PHZ_STAR_PROB',
# 'PHZ_GAL_PROB',
# 'PHZ_QSO_PROB',
# 'PHZ_GLOB_CL_PROB',
# 'PHZ_CLASSIFICATION',
# 'VIS_DET',
# 'MISSING_BAND',
# 'STAR_THRESHOLD',
# 'GALAXY_THRESHOLD',
# 'QSO_THRESHOLD',
# 'GLOBC_THRESHOLD')

catalog_file = glob.glob(f'/net/ketelmeer/data2/kuijken/GAAP/PHZ/EUC_PHZ_CLASSCAT_*.fits')[0]
with fits.open(catalog_file, memmap=True) as hdul:
    cat = Table(hdul[1].data)
column = cat['PHZ_QSO_PROB']
print(cat['OBJECT_ID'][np.argsort(column)[::-1]][:16])

# snr_cutoff = 1
# FWHM = 2
# point_source_probability_cutoff_lower = 0.0
# point_source_probability_cutoff_upper = 1


# fluxes = pd.read_csv(
#     f'/net/vdesk/data2/deklerk/GAAP_data/flux_files/102018667_fluxes.csv')

# filter_1 = 'DES-G'   # e.g., master_table column name
# filter_2 = 'DES-Z'
# filter_3 = 'NIR-Y'
# filter_4 = 'NIR-H'

# plot_error = False

# # Compute colors from master_table fluxes
# x_color = -2.5 * np.log10(fluxes[f'FLUX_{filter_1}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_2}_{FWHM}FWHM'])
# y_color = -2.5 * np.log10(fluxes[f'FLUX_{filter_3}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_4}_{FWHM}FWHM'])

# # Compute errors from sigma dictionary
# xerr = 2.5 / np.log(10) * np.sqrt(
#     (fluxes[f'FLUXERR_{filter_1}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_1}_{FWHM}FWHM'])**2 +
#     (fluxes[f'FLUXERR_{filter_2}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_2}_{FWHM}FWHM'])**2
# )
# yerr = 2.5 / np.log(10) * np.sqrt(
#     (fluxes[f'FLUXERR_{filter_3}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_3}_{FWHM}FWHM'])**2 +
#     (fluxes[f'FLUXERR_{filter_4}_{FWHM}FWHM'] / fluxes[f'FLUX_{filter_4}_{FWHM}FWHM'])**2
# )

# # Select star based on MER catalog
# mask = (fluxes[f'FLUX_{filter_1}_{FWHM}FWHM']/fluxes[f'FLUXERR_{filter_1}_{FWHM}FWHM'] > snr_cutoff) & (fluxes[f'FLUX_{filter_2}_{FWHM}FWHM']/fluxes[f'FLUXERR_{filter_2}_{FWHM}FWHM'] > snr_cutoff) & (
#     fluxes[f'FLUX_{filter_3}_{FWHM}FWHM']/fluxes[f'FLUXERR_{filter_3}_{FWHM}FWHM'] > snr_cutoff) & (fluxes[f'FLUX_{filter_4}_{FWHM}FWHM']/fluxes[f'FLUXERR_{filter_4}_{FWHM}FWHM'] > snr_cutoff) & (fluxes['point_source_probability_mer'] >= point_source_probability_cutoff_lower) & (fluxes['point_source_probability_mer'] < point_source_probability_cutoff_upper)

# print(np.sum(mask))

# plt.errorbar(
#     x_color[mask], y_color[mask],
#     xerr=xerr[mask] * plot_error, yerr=yerr[mask] * plot_error,
#     fmt='o', c='b', ms=1, elinewidth=0.5, alpha=0.8, label='GAAP'
# )
# plt.xlabel(f'{filter_1} - {filter_2}')
# plt.ylabel(f'{filter_3} - {filter_4}')
# # plt.xlim(-.5, 2.5)
# # plt.ylim(-.5, 2.5)
# plt.grid(True)
# plt.show()

# probability_cutoff = 0.7
# FWHM1 = 1
# FWHM2 = 2
# print(np.nanmin(fluxes['FWHM']))
# mask = (fluxes['point_source_probability_mer'] < probability_cutoff)
# plt.axhline(0, c='grey', lw=.5)
# plt.scatter(fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask], np.abs((fluxes[f'FLUX_VIS_{FWHM2}FWHM'][mask]- fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask])/fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask]), label='Extended', s=1)
# # print([np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask])], [np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])])
# # plt.plot([np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])], [np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])], 'r--')
# # plt.xscale('log')
# # plt.yscale('symlog')
# # plt.axhline(0, c='r')
# # plt.title('Extended')
# # plt.show()
# mask = (fluxes['point_source_probability_mer'] >= probability_cutoff)
# plt.scatter(fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask], np.abs((fluxes[f'FLUX_VIS_{FWHM2}FWHM'][mask]- fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask])/fluxes[f'FLUX_VIS_{FWHM1}FWHM'][mask]), label='Point-Source', s=1)
# # print([np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask])], [np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])])
# # plt.plot([np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])], [np.nanmin(fluxes['FLUX_VIS_1FWHM'][mask]), np.nanmax(fluxes['FLUX_VIS_1FWHM'][mask])], 'r--')
# plt.xscale('log')
# plt.yscale('symlog')

# plt.legend()
# plt.ylim(-1e-1, 2e2)
# plt.show()