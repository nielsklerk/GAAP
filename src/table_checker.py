from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

fluxes = pd.read_csv(
    f'/net/vdesk/data2/deklerk/GAAP_data/flux_files_new/102018213_fluxes.csv')
# print(np.mean(fluxes['size']))
# plt.hist(fluxes['size'])
# plt.show()
# print(np.sum(np.isnan(fluxes['DES-G_sigma'])))
# print(fluxes['weight_size'])

filter_1 = 'DES-G'   # e.g., master_table column name
filter_2 = 'DES-R'
filter_3 = 'DES-R'
filter_4 = 'DES-I'

plot_error = False

# Compute colors from master_table fluxes
x_color = -2.5 * np.log10(fluxes[filter_1] / fluxes[filter_2])
y_color = -2.5 * np.log10(fluxes[filter_3] / fluxes[filter_4])

# Compute errors from sigma dictionary
xerr = 2.5 / np.log(10) * np.sqrt(
    (fluxes[f'{filter_1}_sigma'] / fluxes[filter_1])**2 +
    (fluxes[f'{filter_2}_sigma'] / fluxes[filter_2])**2
)
yerr = 2.5 / np.log(10) * np.sqrt(
    (fluxes[f'{filter_3}_sigma'] / fluxes[filter_3])**2 +
    (fluxes[f'{filter_4}_sigma'] / fluxes[filter_4])**2
)

gaap_error = xerr**2 + yerr**2
# Select star based on MER catalog
snr_cutoff = 5
mask = (fluxes[filter_1]/fluxes[f'{filter_1}_sigma'] > snr_cutoff) & (fluxes[filter_2]/fluxes[f'{filter_2}_sigma'] > snr_cutoff) & (
    fluxes[filter_3]/fluxes[f'{filter_3}_sigma'] > snr_cutoff) & (fluxes[filter_4]/fluxes[f'{filter_4}_sigma'] > snr_cutoff)
print(np.sum(mask))
plt.errorbar(
    x_color[mask], y_color[mask],
    xerr=xerr[mask] * plot_error, yerr=yerr[mask] * plot_error,
    fmt='o', c='b', ms=1, elinewidth=0.5, alpha=0.8, label='GAAP'
)

plt.xlabel(f'{filter_1} - {filter_2}')
plt.ylabel(f'{filter_3} - {filter_4}')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.grid(True)
plt.show()
