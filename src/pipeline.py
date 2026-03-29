from utils import download_archive_files, gaussian_2d, create_psf_from_psf_grid
import numpy as np
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.optimize import curve_fit
from tqdm import tqdm
import gc
import pandas as pd
from class_implementation import NoiseModel, PSFDeconvolver, GAAPPhotometry

def find_best_size(x):
    return 0.06 * x * x + 4.3

psf_size_dictionary = {'CFIS-R':47, 'CFIS-U':49, 'PANSTARRS-I':47, 'WISHES-G':49, 'WISHES-Z':49, 'NIR-Y': 33, 'NIR-J': 33, 'NIR-H':33, 'DES-G': 49, 'DES-R': 49, 'DES-I': 49, 'DES-Z': 49, 'VIS':21}
pixel_scale_euclid = .1     #arcsec per pixel
size = 128                  #cutout size
lag = 8                     #max pixel distance in covariance
data_folder = 'data'
storage_folder = 'fluxes'

tile_indeces = ['102159780']
for tile_index in tile_indeces:
    """
    Dowloading files
    """
    filters = download_archive_files(tile_index, data_folder=data_folder)


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
    x_c, y_c = wcs.wcs_world2pix(fluxes['ra'], fluxes['dec'], 0, ra_dec_order=True)

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
                lambda xy, A, s: gaussian_2d(xy, A, s, x_center_cutout, y_center_cutout),
                (x, y),
                cutout.ravel(),
                p0=initial_guess
            )
            aperture_size[i] = find_best_size(np.abs(popt[1])) * pixel_scale_euclid # convert to arcsec
            if i % 5000 == 0:
                gc.collect()
        except RuntimeError:
            aperture_size[i] = np.inf

    """
    Binning the fitted sizes to prevent unstable deconvolution
    """
    bins = np.linspace(.7, 1.1, 100)
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

    """
    Calculate flux for each filter
    """
    for filter in filters:
        print(f'Analyzing: {filter}')
        file = glob.glob(f'{data_folder}/EUC_MER_BGSUB-MOSAIC-{filter}_*.fits')[0]
        with fits.open(file, memmap=True) as hdul:
            hdu = hdul[0]
            zeropoint = hdu.header["MAGZERO"]
            image_conversion_factor = 10 ** ((8.90 - zeropoint) / 2.5) * 1e9
            image = hdu.data
            wcs = WCS(hdu.header)

        centers = wcs.wcs_world2pix(
                fluxes['ra'], fluxes['dec'], 0, ra_dec_order=True
                )
        
        file = glob.glob(f'{data_folder}/EUC_MER_MOSAIC-{filter}-RMS_*.fits')[0]
        with fits.open(file, memmap=True) as hdul:
            hdu = hdul[0]
            zeropoint = hdu.header["MAGZERO"]
            rms_conversion_factor = 10 ** ((8.90 - zeropoint) / 2.5) * 1e9
            rms = hdu.data

        file = glob.glob(f'{data_folder}/EUC_MER_CATALOG-PSF-{filter}_*.fits')[0]
        with fits.open(file, memmap=True) as hdul:
            psf_grid = hdul[1].data

        psf = create_psf_from_psf_grid(psf_grid, psf_size_dictionary[filter], 40)
        deconvolver = PSFDeconvolver(psf)
        deconvolver.prepare([size, size])

        noise = NoiseModel(
            image=image,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
        )
        noise.find_noise_square(80, image[:3400, :3400])
        noise.set_noise_covariance(lag)

        phot = GAAPPhotometry(
            image=image,
            centers=np.asarray(centers),
            sigmas=sigma_binned,
            pixel_scale=pixel_scale_euclid,
            image_conversion_factor=image_conversion_factor,
        )

        phot.measure(size, lag, noise, deconvolver, uncorrelated=False)
        fluxes[f'{filter}'] = phot.flux
        fluxes[f'{filter}_sigma'] = np.sqrt(phot.variance)

    """
    Store the data into a csv file
    """
    df = pd.DataFrame(fluxes)
    df.to_csv(f'{storage_folder}/{tile_index}_fluxes.csv', index=False)
        