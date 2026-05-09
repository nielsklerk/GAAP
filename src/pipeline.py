from utils import download_archive_files, create_psf_from_psf_grid
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from utils import gaussian_2d as gauss_2d
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
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.signal import fftconvolve
from numba import njit
from tqdm import tqdm
from joblib import Parallel, delayed
import pyfftw.interfaces.numpy_fft as fft
import pyfftw
cutout_size = 128
pyfftw.interfaces.cache.enable()
H = W = cutout_size

in_array = pyfftw.empty_aligned((H, W//2 + 1), dtype='complex128')
out_array = pyfftw.empty_aligned((H, W), dtype='float64')

ifft = pyfftw.FFTW(
    in_array,
    out_array,
    direction='FFTW_BACKWARD',
    axes=(0, 1),
    threads=1
)


# @njit(fastmath=True)
def padded_cutout_with_center(image: np.ndarray, cx: float, cy: float, size: int, cutout: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Extract a fixed-size cutout centered on (cy, cx).
    Pads with zeros when the cutout extends beyond the image.

    Parameters
    ----------
    image: np.ndarray
        Image from which the cutout is extracted
    cx: float
        x coordinate of the center of the cutout in the origal image
    cy: float
        y coordinate of the center of the cutout in the origal image
    size: int
        Size of the cutout

    Returns
    -------
    np.ndarray:
        Cutout
    tuple[float, float]:
        Center corresponding to (cx, cy) in the cutout

    Raises
    ------
    ValueError
        Description of when this error is raised
    """

    h, w = image.shape
    half = size // 2

    # Integer anchor
    iy = np.int64(cy)
    ix = np.int64(cx)

    # Desired bounds in image coordinates
    y0 = iy - half
    x0 = ix - half
    y1 = y0 + size
    x1 = x0 + size

    # Overlap with image
    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(h, y1)
    ix1 = min(w, x1)

    # Corresponding region in cutout coordinates
    cy0 = iy0 - y0
    cx0 = ix0 - x0
    cy1 = cy0 + (iy1 - iy0)
    cx1 = cx0 + (ix1 - ix0)

    # Allocate cutout
    if cutout is None:
        cutout = np.zeros((size, size))
    else:
        cutout.fill(0)

    # Insert image data
    cutout[cy0:cy1, cx0:cx1] = image[iy0:iy1, ix0:ix1]

    # Center in cutout coordinates
    cy_c = cy - y0
    cx_c = cx - x0

    return cutout, (cx_c, cy_c)


@njit(fastmath=True)
def gaussian_2d(x, y, x0=0, y0=0, sigma_x=1, sigma_y=1, theta=0, A=1):
    dx = x - x0
    dy = y - y0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate coordinates into Gaussian principal-axis frame
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy

    return A * np.exp(
        -0.5 * (
            (x_rot / sigma_x) ** 2 +
            (y_rot / sigma_y) ** 2
        )
    )


@njit(fastmath=True)
def calc_flux(W_shifted, cutout):
    h, w = W_shifted.shape
    s = 0.0
    for i in range(h):
        for j in range(w):
            s += W_shifted[i, j] * cutout[i, j]
    return s


@njit(fastmath=True)
def fourier_gaussian_2d(x, y, sigma_x=1.0, sigma_y=1.0, theta=0.0, A=1.0):

    c = np.cos(theta)
    s = np.sin(theta)

    sx2 = sigma_x * sigma_x
    sy2 = sigma_y * sigma_y

    norm = A * 2.0 * np.pi * sigma_x * sigma_y

    out = np.empty_like(x)

    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]

        for j in range(x.shape[1]):

            xr = c * xi[j] + s * yi[j]
            yr = -s * xi[j] + c * yi[j]

            r2 = sx2 * xr * xr + sy2 * yr * yr
            out[i, j] = norm * np.exp(-0.5 * r2)

    return out


@njit(fastmath=True)
def compute_phase(kx, ky, dx, dy):

    twopi = 2.0 * np.pi
    return np.exp(-1j * twopi * (ky * dy + kx * dx))


@njit
def bilinear_shift(weight: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Creates a shifted version of the weight function
    that captures the bilinear interpolation

    Parameters
    ----------
    weight: np.ndarray
        Weight function that needs to be shifted
    dx: float
        Fractional part of the x coordinate (0 <= dx <= 1)
    dy: float
        Fractional part of the y coordinate (0 <= dy <= 1)

    Returns
    -------
    np.ndarray
        Bilinear interpolated weight function

    Raises
    ------
    ValueError
        Description of when this error is raised
    """
    W_shifted = (1 - dx) * (1 - dy) * weight[:-1, :-1] \
        + dx * (1 - dy) * weight[:-1, 1:] \
        + (1 - dx) * dy * weight[1:, :-1] \
        + dx * dy * weight[1:, 1:]
    return W_shifted


def flux(image, centers, psfdeconvolver, weight_sizes,
         noise_model=None, cutout_size=128, image_conversion_factor=1, bilinear_interpolation=False):

    centers = np.atleast_2d(centers)
    ws = np.asarray(weight_sizes)

    Nc = centers.shape[0]

    # ----------------------------
    # Normalize weight input
    # ----------------------------
    if ws.ndim == 0:
        ws = np.array([[ws.item(), ws.item(), 0.0]])
    elif ws.ndim == 1:
        if ws.size == 3:
            ws = ws.reshape(1, 3)
        else:
            ws = np.column_stack([ws, ws, np.zeros_like(ws)])
    elif ws.ndim == 2 and ws.shape[1] != 3:
        raise ValueError("weight_sizes must have shape (N,) or (N,3)")

    Nw = ws.shape[0]

    # ----------------------------
    # Define iteration mode
    # ----------------------------
    if Nw == 1:
        mode = "scalar_weight"
    elif Nc == 1:
        mode = "scalar_center"
    elif Nc == Nw:
        mode = "paired"
    else:
        raise ValueError(
            "Mismatch: centers and weights must match or be scalar-expanded")

    # ----------------------------
    # FFT grid
    # ----------------------------
    H = W = cutout_size
    ky = fft.fftfreq(H)[:, None]
    kx = fft.rfftfreq(W)[None, :]

    fluxes = np.empty(max(Nc, Nw))
    variances = None if noise_model is None else np.empty(max(Nc, Nw))

    last_weight = None
    weight_fft = None
    cutout_buffer = np.empty((H, W))

    # ----------------------------
    # Main loop
    # ----------------------------
    if Nw > 1:
        sort_idx = np.argsort(ws[:, 0])
    else:
        sort_idx = np.array([0])
    for j in tqdm(range(max(Nc, Nw)), desc='Measuring FLux'):

        i_w = sort_idx[j] if j < Nw else sort_idx[0]

        # ---- select center ----
        if mode == "scalar_center":
            x_c, y_c = centers[0]
            out_idx = i_w
        elif mode == "paired":
            x_c, y_c = centers[i_w]
            out_idx = i_w
        else:  # scalar_weight
            x_c, y_c = centers[j]
            out_idx = j

        # ---- select weight ----
        if mode == "scalar_weight":
            sx, sy, th = ws[0]
        else:
            sx, sy, th = ws[i_w]

        # ---- recompute PSF-weight FFT if needed ----
        current_weight = (sx, sy, th)

        if current_weight != last_weight:
            FT = fourier_gaussian_2d(
                psfdeconvolver.KX,
                psfdeconvolver.KY,
                sx, sy, th
            )
            weight_fft = psfdeconvolver.psf_prefactor * FT
            last_weight = current_weight
            if bilinear_interpolation:
                in_array[:] = weight_fft
                ifft()
                weight_rescale_unshifted = out_array

        # ---- extract cutout ----
        cutout, (cx_cut, cy_cut) = padded_cutout_with_center(
            image, x_c, y_c, cutout_size, cutout_buffer
        )
        cutout *= image_conversion_factor

        # ---- subpixel shift ----
        ix = int(cx_cut)
        iy = int(cy_cut)
        dx = cx_cut - ix
        dy = cy_cut - iy

        if bilinear_interpolation:
            weight_rescale = bilinear_shift(weight_rescale_unshifted, dx, dy)
            cutout = cutout[:-1, :-1]
        else:
            phase = compute_phase(kx, ky, dx, dy)
            #in_array[:] = weight_fft * phase
            #ifft()
            #weight_rescale = np.abs(out_array)
            weight_rescale = fft.irfft2(weight_fft * phase)

        # ---- flux ----
        fluxes[out_idx] = calc_flux(weight_rescale, cutout)

        # ---- variance ----
        if noise_model is not None:
            if noise_model.rms is None:
                wf = fft.irfft2(weight_fft)
            else:
                wf = weight_rescale

            variances[out_idx] = noise_model.calc_error(
                wf, x_c, y_c, cutout_size
            )

    return fluxes if noise_model is None else (fluxes, variances)


def GAAP_flux(image, psf, centers, weight_sizes, rms=None, calculate_noise=True, cutout_size=128, noise_square_size=128, image_conversion_factor=1, rms_conversion_factor=1):
    psfdecon = PSFDeconvolver(psf)
    psfdecon.prepare(cutout_size)
    if calculate_noise:
        noise = NoiseModel(
            image=image,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
        )
        noise.find_noise_square(noise_square_size)
        noise.set_noise_covariance(cutout_size//2)
    else:
        noise = None
    return flux(image, centers, psfdecon, weight_sizes, noise, cutout_size=cutout_size, image_conversion_factor=image_conversion_factor)


class NoiseModel:
    def __init__(
        self,
        image: np.ndarray | None = None,
        rms: np.ndarray | None = None,
        image_conversion_factor: float = 1.0,
        rms_conversion_factor: float = 1.0,
        uncorrelated=False
    ) -> None:

        self.image = image
        self.rms = rms
        self.image_conversion_factor = image_conversion_factor
        self.rms_conversion_factor = rms_conversion_factor
        self.uncorrelated = uncorrelated

        self.noise_square = None
        self.noise_covariance = None
        self.poisson_image = None
        self.ac = None
        self.kernel = None

    def find_noise_square(self,
                          box_size: int = 100,
                          image: np.ndarray | None = None
                          ) -> None:
        """
        Find a sourceless square from an image and store it.

        Parameters
        ----------
        box_size: int, optional
            Size of the noise square
        image: np.ndarray | None, optional
            Image from which a noise square is extracted
            If None the class image is used
        """
        image = self.image if image is None else image

        ny, nx = image.shape

        best_std = np.inf
        best_square = None

        # define smaller square size and step
        step = box_size  # or smaller (e.g. box_size // 2 for overlap)

        for y in range(0, ny - box_size + 1, step):
            for x in range(0, nx - box_size + 1, step):

                square = image[y:y+box_size, x:x+box_size]

                # compute statistics for this square
                local_mean = np.mean(square)
                local_var = np.var(square)
                local_std = np.sqrt(local_var)

                signal_threshold = np.percentile(square, 10)
                nonzero_fraction = np.count_nonzero(square) / square.size

                # reject "bad" squares
                # if local_mean >= signal_threshold:
                #     continue
                if nonzero_fraction < 0.5:
                    continue

                # keep the best (lowest noise)
                if local_std < best_std:
                    best_std = local_std
                    best_square = square

        # fallback if nothing found
        if best_square is None:
            raise ValueError("No suitable noise square found")

        self.noise_square = best_square

    def set_noise_square(self, noise_square):
        self.noise_square = noise_square

    def set_noise_covariance(self, maxlag: int) -> None:
        """
        Extract a fixed-size cutout centered on (cy, cx).
        Pads with zeros when the cutout extends beyond the image.

        Parameters
        ----------
        maxlag: int
            Maximum pixel difference where the pixel are still correlated
        """

        image = self.noise_square * self.image_conversion_factor
        self.noise_covariance = self._covariance_fft2d(image, maxlag)

    def _covariance_fft2d(self, noise_image: np.ndarray, maxlag: int) -> np.ndarray:
        """
        Calculate the local covariance matrix from the noise square

        Parameters
        ----------
        noise_image: np.ndarray
            Description of param1
        maxlag: int
            Maximum pixel difference where the pixel are still correlated

        Returns
        -------
        np.ndarray
            Local covariance matrix
        """

        img = noise_image.copy()
        h, w = img.shape
        img -= np.mean(img)

        self.ac = fftconvolve(img, img[::-1, ::-1], mode="same")
        self.ac /= (h * w)

        cy, cx = h // 2, w // 2

        return self.ac[
            cy - maxlag: cy + maxlag + 1,
            cx - maxlag: cx + maxlag + 1,
        ]

    def calc_error(self, weight, xc, yc, size):
        if self.rms is None:
            return self.background_error(weight)
        else:
            return self.rms_error(weight, xc, yc, size)

    def background_error(self, weight):
        if self.uncorrelated:
            negative_pixels = self.noise_square[self.noise_square < 0]

            background_variance = (
                np.sum(negative_pixels**2) / len(negative_pixels)
            ) * self.image_conversion_factor**2
            return background_variance * np.sum(weight**2)
        autocorr_weight = fftconvolve(weight, weight[::-1, ::-1], mode='same')
        noise_covariance = self.noise_covariance[:weight.shape[0],
                                                 :weight.shape[1]]
        return np.sum(noise_covariance * autocorr_weight)

    def rms_error(self, weight, xc, yc, size):
        if self.kernel is None:
            self.kernel = self.ac / np.max(self.ac)

        rms_cutout, _ = padded_cutout_with_center(self.rms, xc, yc, size)
        rms_cutout = rms_cutout[:weight.shape[0], :weight.shape[1]]
        weight_prime = rms_cutout * weight * self.rms_conversion_factor
        if self.uncorrelated:
            return np.sum(weight_prime * weight_prime)
        conv = fftconvolve(weight_prime, self.kernel, mode='same')
        return np.sum(weight_prime * conv)


class PSFDeconvolver:
    def __init__(self, psf: np.ndarray):
        self.psf = psf
        self.psf_cache = None
        self._fft_buffer = None
        self.KX, self.KY = None, None
        self.psf_prefactor = None

    def prepare(self, cutout_size, eps: float = 1e-8) -> dict[str, float | tuple[int, int]]:
        """
        Prepares the PSF factor in the deconvolution.

        Parameters
        ----------
        image_shape: tuple[int, int]
            Shape of the cutout that is used for the photometry
        K: float = 1e-16
            Factor for numerical stability

        Returns
        -------
        dict[str, float | tuple[int, int]]
            Dictionary with:
            - "PSF_prefactor": PSF factor in the deconvolution (float).
            - "pad_shape": dimension of the padded shape (tuple).
        """

        ky = fft.fftfreq(cutout_size) * 2*np.pi
        kx = fft.rfftfreq(cutout_size) * 2*np.pi

        self.KX, self.KY = np.meshgrid(kx, ky)

        psf_padded, _ = padded_cutout_with_center(
            self.psf, self.psf.shape[0]/2, self.psf.shape[1]/2, cutout_size)
        ft_psf = fft.rfft2(psf_padded[::-1, ::-1])
        self.psf_prefactor = np.conj(ft_psf) / (np.abs(ft_psf)**2 + eps)


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
    measured_fluxes, variances = GAAP_flux(image, psf, np.asarray(centers).T, sigma_binned/PIXEL_SCALE_EUCLID, rms,
                                           image_conversion_factor=image_conversion_factor, rms_conversion_factor=rms_conversion_factor)

    return filter_, measured_fluxes, np.sqrt(variances)


coeffs = [5.31180363e-05, -2.61813743e-03,
          3.95507479e-02,  2.11945955e-02, 5.80616717e+00]

find_best_size = Polynomial(coeffs[::-1])
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
tile_indeces = ['102018213']
for tile_index in tile_indeces:
    try:
        tile_index_str = str(tile_index)

        if tile_index_str in processed:
            continue
        """
        Dowloading files
        """
        # filters = download_archive_files(
        #     tile_index, filename_file=filename_file, data_folder=data_folder, max_workers=max_workers_download)
        filters = filenames.loc[tile_index]['FILTER']
        filters = ['DES-G', 'DES-R', 'DES-I']
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
        fluxes['point_source_probability_mer'] = cat['POINT_LIKE_PROB']

        """
        Fitting the Gaussians to sources
        """
        # Load VIS image for fitting galaxy shapes
        file = glob.glob(f'{data_folder}/EUC_MER_BGSUB-MOSAIC-VIS_*.fits')[0]
        fits.open(file).verify()
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
        # initial_guess = [10, .2]
        # aperture_size = np.full(len(x_c), np.nan)

        # # Fitting each object
        # def model(coords, A, sigma):
        #     x, y = coords
        #     r2 = (x - x_center)**2 + (y - y_center)**2
        #     return (A * np.exp(-r2 / (2*sigma**2))).ravel()

        # for i, (x_center, y_center) in enumerate(tqdm(zip(x_c, y_c), total=len(x_c), desc='Fitting Sources', disable=False)):
        #     x0 = int(x_center)
        #     y0 = int(y_center)

        #     # bounds (clip to image)
        #     x_min = max(0, x0 - size//2)
        #     x_max = min(nx, x0 + size//2 + 1)
        #     y_min = max(0, y0 - size//2)
        #     y_max = min(ny, y0 + size//2 + 1)

        #     cutout = image[y_min:y_max, x_min:x_max]

        #     x_center_cutout = x_center-x_min
        #     y_center_cutout = y_center-y_min

        #     # local coordinate grid
        #     y, x = np.mgrid[y_min:y_max, x_min:x_max]
        #     try:
        #         p0 = [
        #             cutout.max(),
        #             1.0
        #         ]

        #         popt, _ = curve_fit(
        #             model,
        #             (x, y),
        #             cutout.ravel(),
        #             p0=p0,
        #             bounds=(
        #                 [0, 0.2],
        #                 [np.inf, size]
        #             )
        #         )
        #         # print(popt[1])
        #         aperture_size[i] = np.abs(
        #             popt[1])  # convert to arcsec
        #         if i % 5000 == 0:
        #             gc.collect()
        #     except RuntimeError:
        #         continue

        # Store size
        # fluxes['size'] = aperture_size * PIXEL_SCALE_EUCLID
        weight_size = cat['FWHM']

        # weight_size = np.zeros_like(x_c)

        """
        Binning the fitted sizes
        """
        bins = np.linspace(.7, 10 * PIXEL_SCALE_EUCLID,
                           100)  # Bin between max PSF size and 9% cutout size
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

        # with open(processed_file, "a") as f:
        #     f.write(tile_index_str + "\n")

        # processed.add(tile_index_str)
    except Exception as e:
        print(f"Error on {tile_index}: {e}")
        continue
    break
