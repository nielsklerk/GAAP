from utils import download_archive_files, create_psf_from_psf_grid
import numpy as np
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from GAAP import gaap_flux


def process_filter(filter_):

    # --- Image ---
    file = glob.glob(f'{data_folder}/EUC_MER_BGSUB-MOSAIC-{filter_}_*.fits')[0]
    with fits.open(file, memmap=True) as hdul:
        hdu = hdul[0]
        zeropoint = hdu.header["MAGZERO"]
        image_conversion_factor = 10 ** ((8.90 - zeropoint) / 2.5) * 1e9
        image = hdu.data
        wcs = WCS(hdu.header)

    centers = wcs.wcs_world2pix(
        fluxes['ra'], fluxes['dec'], 0, ra_dec_order=True)

    # --- RMS ---
    file = glob.glob(f'{data_folder}/EUC_MER_MOSAIC-{filter_}-RMS_*.fits')[0]
    with fits.open(file, memmap=True) as hdul:
        hdu = hdul[0]
        zeropoint = hdu.header["MAGZERO"]
        rms_conversion_factor = 10 ** ((8.90 - zeropoint) / 2.5) * 1e9
        rms = hdu.data

    # --- PSF ---
    file = glob.glob(f'{data_folder}/EUC_MER_CATALOG-PSF-{filter_}_*.fits')[0]
    with fits.open(file, memmap=True) as hdul:
        psf_grid = hdul[1].data
    psf = create_psf_from_psf_grid(
        psf_grid, psf_size_dictionary[filter_], 100)

    # --- Measurements ---
    measured_fluxes, variances = gaap_flux(image, psf, np.asarray(centers).T, sigma_binned/PIXEL_SCALE_EUCLID, rms,
                                           image_conversion_factor=image_conversion_factor, rms_conversion_factor=rms_conversion_factor)
    return filter_, measured_fluxes, np.sqrt(variances)


psf_size_dictionary = {'CFIS-R': 47, 'CFIS-U': 49, 'PANSTARRS-I': 47, 'WISHES-G': 47, 'WISHES-Z': 49,
                       'NIR-Y': 33, 'NIR-J': 33, 'NIR-H': 33, 'DES-G': 49, 'DES-R': 49, 'DES-I': 49, 'DES-Z': 49, 'VIS': 21}

data_folder = '/net/vdesk/data2/deklerk/GAAP_data/temp'
storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files_new'
filename_file = '/home/deklerk/GAAP/src/EUCLID_ARCHIVE_files.pkl'
processed_file = "/net/vdesk/data2/deklerk/GAAP_data/processed.txt"

processed = set()

# Load progress at start
try:
    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)
except FileNotFoundError:
    pass


PIXEL_SCALE_EUCLID = .1  # arcsec per pixel
size = 128  # cutout size
max_workers_download = 6
max_workers = 4

filenames = pd.read_pickle(filename_file)
tile_indeces = filenames.index.tolist()
# tile_indeces = ['102070144']
for tile_index in tile_indeces:
    # try:
    tile_index_str = str(tile_index)

    if tile_index_str in processed:
        continue
    """
    Dowloading files
    """
    filters = download_archive_files(
        tile_index, filename_file=filename_file, data_folder=data_folder, max_workers=max_workers_download)
    if len(filters) >= 9:
        max_workers = 5
    else:
        max_workers = 4
    filters = filenames.loc[tile_index]['FILTER']
    # filters = ['DES-G', 'DES-R', 'DES-I']
    """
    Load catalog for coordinates
    """
    # Load catalog file for coordinates
    catalog_file = glob.glob(f'{data_folder}/EUC_MER_FINAL-CAT_*.fits')[0]
    with fits.open(catalog_file, memmap=True) as hdul:
        cat = Table(hdul[1].data)

    fluxes = {}
    fluxes['id'] = cat['OBJECT_ID']
    fluxes['tile_index'] = tile_index
    fluxes['ra'] = cat['RIGHT_ASCENSION']
    fluxes['dec'] = cat['DECLINATION']
    fluxes['FWHM'] = cat['FWHM']
    fluxes['point_source_probability_mer'] = cat['POINT_LIKE_PROB']

    """
    Fitting the Gaussians to sources
    """
    weight_size = cat['FWHM']
    print(weight_size)

    """
    Binning the fitted sizes above the maximum PSF size
    """
    bins = np.arange(.3, np.nanmax(weight_size), 0.01)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # prepare array
    sigma_binned = np.full_like(weight_size, np.nan, dtype=float)

    # mask valid (non-NaN) entries
    mask = ~np.isnan(weight_size)

    # digitize only valid entries
    bin_idx = np.digitize(weight_size[mask], bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)

    # assign binned values
    sigma_binned[mask] = bin_centers[bin_idx]

    fluxes['weight_size'] = sigma_binned

    """
    Calculate flux for each filter
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_filter, f) for f in filters]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing filters"):
            results.append(future.result())

    for filter_, flux, sigma in results:
        fluxes[f'{filter_}'] = flux
        fluxes[f'{filter_}_sigma'] = sigma

    """
    Store the data into a csv file
    """
    df = pd.DataFrame(fluxes)
    df.to_csv(f'{storage_folder}/{tile_index}_fluxes.csv', index=False)

    with open(processed_file, "a") as f:
        f.write(tile_index_str + "\n")

    processed.add(tile_index_str)
    # except Exception as e:
    #     print(f"Error on {tile_index}: {e}")
    #     continue
    gc.collect()
