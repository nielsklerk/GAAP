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

class NoiseModel:
    def __init__(
        self,
        image: np.ndarray | None = None,
        rms: np.ndarray | None = None,
        image_conversion_factor: float = 1.0,
        rms_conversion_factor: float = 1.0,
        uncorrelated = False
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
        noise_covariance = self.noise_covariance[:weight.shape[0], :weight.shape[1]]
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

        psf_padded, _ = padded_cutout_with_center(self.psf, self.psf.shape[0]/2, self.psf.shape[1]/2, cutout_size)
        ft_psf = fft.rfft2(psf_padded[::-1, ::-1])
        self.psf_prefactor = np.conj(ft_psf) / (np.abs(ft_psf)**2 + eps)

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
         noise_model=None, cutout_size=128):

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
        raise ValueError("Mismatch: centers and weights must match or be scalar-expanded")

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
    for i in range(max(Nc, Nw)):

        # ---- select center ----
        if mode == "scalar_center":
            x_c, y_c = centers[0]
        else:
            x_c, y_c = centers[i]

        # ---- select weight ----
        if mode == "scalar_weight":
            sx, sy, th = ws[0]
        else:
            sx, sy, th = ws[i]

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
        fluxes[i] = calc_flux(weight_rescale, cutout)

        # ---- variance ----
        if noise_model is not None:
            if noise_model.rms is None:
                wf = fft.irfft2(weight_fft)
            else:
                wf = weight_rescale

            variances[i] = noise_model.calc_error(
                wf, x_c, y_c, cutout_size
            )

    return fluxes if noise_model is None else (fluxes, variances)

def compute_psf_column(j, psf_size, weight_sizes, X, Y, star,
                       noise_images, N_trials, N):

    psf = gaussian_2d(X, Y, 0, 0, psf_size, psf_size, 1)
    psf /= np.sum(psf)

    psfdecon = PSFDeconvolver(psf)
    psfdecon.prepare(N)

    observation = fftconvolve(star, psf, mode='same')
    noisy_stack = observation[None, :, :] + noise_images

    col_fluxes = np.empty((N_trials, len(weight_sizes)))
    col_variances = np.empty((N_trials, len(weight_sizes)))

    for k in range(N_trials):

        noise_model = NoiseModel(noise_images[k])
        noise_model.set_noise_square(noise_images[k])
        noise_model.set_noise_covariance(N // 2)

        f, v = flux(
            noisy_stack[k],
            (N / 2, N / 2),
            psfdecon,
            weight_sizes=weight_sizes,
            noise_model=noise_model,
            cutout_size=N
        )

        col_fluxes[k] = f
        col_variances[k] = v

    std_flux = np.std(col_fluxes, axis=0)
    mean_sqrt_var = np.mean(np.sqrt(col_variances), axis=0)

    col = (mean_sqrt_var - std_flux) / std_flux

    return j, col

def compute_psf_column_rms(j, psf_size, weight_sizes, X, Y, star,
                       noise_images, N_trials, N):

    psf = gaussian_2d(X, Y, 0, 0, psf_size, psf_size, 1)
    psf /= np.sum(psf)

    psfdecon = PSFDeconvolver(psf)
    psfdecon.prepare(N)

    observation = fftconvolve(star, psf, mode='same')
    noisy_stack = observation[None, :, :] + noise_images

    col_fluxes = np.empty((N_trials, len(weight_sizes)))
    col_variances = np.empty((N_trials, len(weight_sizes)))

    for k in range(N_trials):
        noise_model = NoiseModel(noise_images[k], np.ones_like(noise_images[k]) * .1)
        noise_model.set_noise_square(noise_images[k])
        noise_model.set_noise_covariance(N // 2)

        f, v = flux(
            noisy_stack[k],
            (N / 2, N / 2),
            psfdecon,
            weight_sizes=weight_sizes,
            noise_model=noise_model,
            cutout_size=N
        )

        col_fluxes[k] = f
        col_variances[k] = v

    std_flux = np.std(col_fluxes, axis=0)
    mean_sqrt_var = np.mean(np.sqrt(col_variances), axis=0)

    col = (mean_sqrt_var - std_flux) / std_flux

    return j, col