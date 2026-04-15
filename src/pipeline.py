from utils import download_archive_files, gaussian_2d, create_psf_from_psf_grid
import numpy as np
import glob
from numpy.polynomial import Polynomial
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.optimize import curve_fit
from tqdm import tqdm
import gc
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from class_implementation import NoiseModel, PSFDeconvolver, GAAPPhotometry
from tqdm import tqdm


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

    psf = create_psf_from_psf_grid(psf_grid, psf_size_dictionary[filter_], 40)
    deconvolver = PSFDeconvolver(psf)
    deconvolver.prepare([size, size])

    noise = NoiseModel(
        image=image,
        rms=rms,
        image_conversion_factor=image_conversion_factor,
        rms_conversion_factor=rms_conversion_factor,
    )
    noise.find_noise_square(100)
    noise.set_noise_covariance(lag)

    phot = GAAPPhotometry(
        image=image,
        centers=np.asarray(centers),
        sigmas=sigma_binned,
        pixel_scale=PIXEL_SCALE_EUCLID,
        image_conversion_factor=image_conversion_factor,
    )

    phot.measure(size, lag, noise, deconvolver,
                 uncorrelated=False, show_progress=False)

    return filter_, phot.flux, np.sqrt(phot.variance)

coeffs = [-5.90661928e-05,  2.18985870e-03, -3.19578589e-02,  2.41491336e-01, -1.25279768e-01,  4.11357033e+00]
find_best_size = Polynomial(coeffs[::-1])
psf_size_dictionary = {'CFIS-R': 47, 'CFIS-U': 49, 'PANSTARRS-I': 47, 'WISHES-G': 47, 'WISHES-Z': 49,
                       'NIR-Y': 33, 'NIR-J': 33, 'NIR-H': 33, 'DES-G': 49, 'DES-R': 49, 'DES-I': 49, 'DES-Z': 49, 'VIS': 21}

data_folder = '/net/vdesk/data2/deklerk/GAAP_data/temp'
storage_folder = '/net/vdesk/data2/deklerk/GAAP_data/flux_files'
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
lag = 8  # max pixel distance in covariance
max_workers_download = 6
max_workers = 4

filenames = pd.read_pickle(filename_file)
tile_indeces = filenames.index.tolist()

for tile_index in tile_indeces:
    try:
        tile_index_str = str(tile_index)

        if tile_index_str in processed:
            continue
        """
        Dowloading files
        """
        filters = download_archive_files(tile_index, filename_file=filename_file, data_folder=data_folder, max_workers=max_workers_download)
        #filters = filenames.loc[tile_index]['FILTER']
        """
        Load catalog for coordinates
        """
        # Load catalog file for coordinates
        catalog_file = glob.glob(f'{data_folder}/EUC_MER_FINAL-CAT_*.fits')[0]
        with fits.open(catalog_file, memmap=True) as hdul:
            cat = Table(hdul[1].data)

        fluxes = {}
        fluxes['id'] = cat['OBJECT_ID']
        fluxes['ra'] = cat['RIGHT_ASCENSION']
        fluxes['dec'] = cat['DECLINATION']

        """
        Fitting the Gaussians to sources
        """
        # Load VIS image for fitting galaxy shapes
        file = glob.glob(f'{data_folder}/EUC_MER_BGSUB-MOSAIC-VIS_*.fits')[0]
        with fits.open(file, memmap=True) as hdul:
            hdu = hdul[0]
            image = hdu.data
            wcs = WCS(hdu.header)
            nx = hdu.header["NAXIS1"]
            ny = hdu.header["NAXIS2"]

        # Pixel coordinates
        x_c, y_c = wcs.wcs_world2pix(
            fluxes['ra'], fluxes['dec'], 0, ra_dec_order=True)

        # Initial guess for fitting
        initial_guess = [np.mean(image[:1000, :1000]), .2]
        aperture_size = np.full(len(x_c), np.nan)

        # Fitting each object
        for i, (x_center, y_center) in enumerate(tqdm(zip(x_c, y_c), total=len(x_c), desc='Fitting Sources')):
            x0 = int(round(x_center))
            y0 = int(round(y_center))

            # bounds (clip to image)
            x_min = max(0, x0 - size//2)
            x_max = min(nx, x0 + size//2 + 1)
            y_min = max(0, y0 - size//2)
            y_max = min(ny, y0 + size//2 + 1)

            cutout = image[y_min:y_max, x_min:x_max]

            x_center_cutout = x_center - x_min
            y_center_cutout = y_center - y_min

            # local coordinate grid
            y, x = np.mgrid[y_min:y_max, x_min:x_max]
            try:
                popt, _ = curve_fit(
                    lambda xy, A, s: gaussian_2d(
                        xy, A, s, x_center_cutout, y_center_cutout),
                    (x, y),
                    cutout.ravel(),
                    p0=initial_guess
                )
                aperture_size[i] = find_best_size(
                    np.abs(popt[1])) * PIXEL_SCALE_EUCLID  # convert to arcsec
                if i % 5000 == 0:
                    gc.collect()
            except RuntimeError:
                aperture_size[i] = np.inf

        """
        Binning the fitted sizes to prevent unstable deconvolution
        """
        bins = np.linspace(.7, size * 0.09 * PIXEL_SCALE_EUCLID,
                           100)  # Bin between max PSF size and 9% cutout size
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # prepare array
        sigma_binned = np.full_like(aperture_size, np.nan, dtype=float)

        # mask valid (non-NaN) entries
        mask = ~np.isnan(aperture_size)

        # digitize only valid entries
        bin_idx = np.digitize(aperture_size[mask], bins) - 1
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
    except Exception as e:
        print(f"Error on {tile_index}: {e}")
        continue
