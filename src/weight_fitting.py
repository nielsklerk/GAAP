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
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from class_implementation import NoiseModel, PSFDeconvolver, GAAPPhotometry
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

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['image.origin'] = "lower"
plt.rcParams['image.cmap'] = 'magma'


@njit(fastmath=True)
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


def flux(image, centers, psfdeconvolver, weight_sizes,
         noise_model=None, cutout_size=128, image_conversion_factor=1):

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
    for j in range(max(Nc, Nw)):

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

        phase = compute_phase(kx, ky, dx, dy)
        in_array[:] = weight_fft * phase
        ifft()
        weight_rescale = out_array

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


def GAAP_flux(image, psf, centers, weight_sizes, rms=None, calculate_noise=True, cutout_size=128, noise_square_size=128, image_conversion_factor=1, rms_conversion_factor=1, noise_square=None):
    psfdecon = PSFDeconvolver(psf)
    psfdecon.prepare(cutout_size)
    if calculate_noise:
        noise = NoiseModel(
            image=image,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
        )
        if noise_square is None:
            noise.find_noise_square(noise_square_size)
        else:
            noise.set_noise_square(noise_square)
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


psf_size_dictionary = {'CFIS-R': 49, 'CFIS-U': 47, 'PANSTARRS-I': 47, 'WISHES-G': 49, 'WISHES-Z': 49,
                       'NIR-Y': 33, 'NIR-J': 33, 'NIR-H': 33, 'DES-G': 49, 'DES-R': 49, 'DES-I': 49, 'DES-Z': 49, 'VIS': 21}
psf_size_dictionary = {'VIS': 21}
PIXEL_SCALE_EUCLID = .1
data_folder = '/net/vdesk/data2/deklerk/GAAP_data/temp'
psf_dict = {}
for filter, psf_size in psf_size_dictionary.items():
    file = glob.glob(f'{data_folder}/EUC_MER_CATALOG-PSF-{filter}_*.fits')[0]
    with fits.open(file, memmap=True) as hdul:
        psf_grid = hdul[1].data
    psf = create_psf_from_psf_grid(psf_grid, psf_size_dictionary[filter], 50)
    psf_dict[filter] = psf.copy()
    initial_model = Gaussian2D(
        amplitude=psf.max(),
        x_mean=psf.shape[1]/2,
        y_mean=psf.shape[0]/2,
        x_stddev=2,
        y_stddev=2
    )
    fitter = LevMarLSQFitter()
    y, x = np.mgrid[:psf.shape[0], :psf.shape[1]]

    fit = fitter(initial_model, x, y, psf)
    sigma_x = fit.x_stddev.value
    sigma_y = fit.y_stddev.value
    psf_sigma = (sigma_x + sigma_y) / 2
    print(
        f'The {filter} PSF has a scale parameter of: {psf_sigma*PIXEL_SCALE_EUCLID:.2f} arcsec')

size = 128
noise_image = np.random.normal(0, .1, (size, size))

weights = np.linspace(.1, 40, 500)
centers = np.full((len(weights), 2), size/2)
sigmas = np.logspace(-1, np.log10(128), 200)
# sigmas = np.array([4, 6])
best_weight = np.zeros_like(sigmas)
x = np.arange(0, size, 1) - size/2
y = np.arange(0, size, 1) - size/2
X, Y = np.meshgrid(x, y)

# plot_weight_snr = True
# for i, sigma in enumerate(tqdm(sigmas)):
#     total_snr = 0
#     image_intrinsic = gaussian_2d(X, Y, 0, 0, sigma, sigma)
#     image_intrinsic /= np.sum(image_intrinsic)
#     image_intrinsic *= 1000
#     # Convolve the galaxy with the PSF
#     for filter, psf in psf_dict.items():
#         observation = fftconvolve(
#             image_intrinsic, psf, mode='same') + noise_image
#         fluxes, error = GAAP_flux(observation, psf, centers,
#                                   weights, noise_square=noise_image)
#         snr = fluxes/error
#         snr /= np.max(snr)
#         total_snr += snr
#         if i == len(sigmas)-1 and plot_weight_snr:
#             plt.plot(weights, snr, label=filter)
#     if i == len(sigmas)-1 and plot_weight_snr:
#         plt.title(r'$\sigma_\mathrm{Source}$' + f'={sigma} pix')
#         plt.xlabel(r'$\sigma_\mathrm{weight}$ [pix]')
#         plt.ylabel('SNR [a.u.]')
#         plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
#         plt.xlim(0, np.max(weights))
#         plt.savefig(
#             '/home/deklerk/GAAP/results/figures/observation/weight_snr_test.pdf', bbox_inches="tight", pad_inches=0)
#     best_weight[i] = weights[np.argmax(total_snr)]
plt.figure()
best_weight = np.load('best_weight.npy')
convolved_sigma = np.sqrt(sigmas**2 + psf_sigma**2)
mask = convolved_sigma < 20
n = 4
coeffs = np.polyfit(np.abs(convolved_sigma[mask]), best_weight[mask], n)
print(coeffs)
f = Polynomial(coeffs[::-1])  # reverse order
# Evaluate
x_new = np.linspace(0, np.max(convolved_sigma[mask]), 100)
y_fit = f(x_new)
plt.scatter(np.abs(convolved_sigma[mask]), best_weight[mask], color='blue')
plt.plot(x_new, y_fit, color='red')
plt.xlim(0, None)
plt.ylim(0, None)
plt.xlabel('Source size in VIS')
plt.ylabel(r'Size maximizing SNR')
plt.grid()
plt.tight_layout()
plt.savefig(
    '/home/deklerk/GAAP/results/figures/observation/best_weight_snr.pdf', )
