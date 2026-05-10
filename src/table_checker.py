import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

snr_cutoff = 3
point_source_probability_cutoff_lower = 0.0
point_source_probability_cutoff_upper = 1


fluxes = pd.read_csv(
    f'/net/vdesk/data2/deklerk/GAAP_data/flux_files_new/102044185_fluxes.csv')

filter_1 = 'DES-G'   # e.g., master_table column name
filter_2 = 'NIR-J'
filter_3 = 'NIR-J'
filter_4 = 'NIR-H'

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

# Select star based on MER catalog
mask = (fluxes[filter_1]/fluxes[f'{filter_1}_sigma'] > snr_cutoff) & (fluxes[filter_2]/fluxes[f'{filter_2}_sigma'] > snr_cutoff) & (
    fluxes[filter_3]/fluxes[f'{filter_3}_sigma'] > snr_cutoff) & (fluxes[filter_4]/fluxes[f'{filter_4}_sigma'] > snr_cutoff) #& (fluxes['point_source_probability_mer'] >= point_source_probability_cutoff_lower) & (fluxes['point_source_probability_mer'] < point_source_probability_cutoff_upper)

print(np.sum(mask))

plt.errorbar(
    x_color[mask], y_color[mask],
    xerr=xerr[mask] * plot_error, yerr=yerr[mask] * plot_error,
    fmt='o', c='b', ms=1, elinewidth=0.5, alpha=0.8, label='GAAP'
)
plt.xlabel(f'{filter_1} - {filter_2}')
plt.ylabel(f'{filter_3} - {filter_4}')
plt.xlim(-.5, 6)
plt.ylim(-.5, 1)
plt.grid(True)
plt.show()
