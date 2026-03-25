import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from scipy.fft import rfft2, irfft2, ifftshift
from numba import njit
import gc
import warnings
from astropy.table import Table
from astropy.nddata import Cutout2D
from tqdm import tqdm
warnings.filterwarnings("ignore")


@njit
def gaussian_weight(height: int, width: int, xc: float = 0, yc: float = 0, sigma: float = 1) -> np.ndarray:
    """
    Calculate Gaussian weight function

    Parameters
    ----------
    height: int
        Height of the weight function in pixels
    width: int
        Width of the weight function in pixels
    xc: float, optional
        x coordinate of the center of the weight function in pixels
    yc: float, optional
        y coordinate of the center of the weight function in pixels
    sigma: float, optional
        Scale parameter of the weight function

    Returns
    -------
    np.ndarray:
        Weight function
    """

    # Create an array for the x and y direction
    x = np.arange(width)
    y = np.arange(height)

    # Calculate the Gaussian in the x and y direction
    gx = np.exp(-0.5 * ((x - xc) / sigma) ** 2)
    gy = np.exp(-0.5 * ((y - yc) / sigma) ** 2)

    # Combining the x and y direction to make a 2D Gaussian
    return gy[:, None] * gx[None, :]


def padded_cutout_with_center(image: np.ndarray, cx: float, cy: float, size: int) -> tuple[np.ndarray, tuple[float, float]]:
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
    iy = np.int64(np.floor(cy))
    ix = np.int64(np.floor(cx))

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
    cutout = np.zeros((size, size))

    # Insert image data
    cutout[cy0:cy1, cx0:cx1] = image[iy0:iy1, ix0:ix1]

    # Center in cutout coordinates
    cy_c = cy - y0
    cx_c = cx - x0

    return cutout, (cx_c, cy_c)


def create_psf(
    image: np.ndarray,
    catalog: Table,
    psf_size: int,
    window_size: float = 0.1,
    lower_percentile: float = 98.0,
    upper_percentile: float = 99.9,
    increase_window_factor: float = 2,
    minimum_log_flux=8,
    plot_chimney: bool = False,
    plot_psf: bool = False,
) -> np.ndarray:
    """
    Extract a fixed-size cutout centered on (cy, cx).
    Pads with zeros when the cutout extends beyond the image.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value

    Raises
    ------
    ValueError
        Description of when this error is raised
    """

    # Opening the flux and flux radius
    log_flux_radius = np.log(catalog["FLUX_RADIUS"])
    log_flux = np.log(catalog["FLUX_AUTO"])
    mask = np.isfinite(log_flux_radius) & np.isfinite(log_flux)

    # Calculate the flux radius that has the highest total flux within a window size
    maximum_flux = -np.inf
    center_maximum = -np.inf
    for window_center in np.linspace(min(log_flux_radius[mask]), max(log_flux_radius[mask]), 100):
        # Select region around window center
        new_mask = (
            np.isfinite(log_flux_radius)
            & np.isfinite(log_flux)
            & (log_flux_radius > window_center - window_size)
            & (log_flux_radius < window_center + window_size)
            & (log_flux > minimum_log_flux)
        )

        # Check if the total flux in region is maximum
        if np.sum(log_flux[new_mask]) > maximum_flux:
            maximum_flux = np.sum(log_flux[new_mask])
            center_maximum = window_center

    # Select all the sources in the found chimney
    selection_mask = (
        np.isfinite(log_flux_radius)
        & np.isfinite(log_flux)
        & (log_flux_radius > center_maximum - increase_window_factor * window_size)
        & (log_flux_radius < center_maximum + increase_window_factor * window_size)
    )

    # Make a selection of the sources in the chinmey
    percentiles = np.percentile(log_flux[selection_mask], [
                                lower_percentile, upper_percentile])
    selection_mask = (
        np.isfinite(log_flux_radius)
        & np.isfinite(log_flux)
        & (log_flux_radius > center_maximum - increase_window_factor * window_size)
        & (log_flux_radius < center_maximum + increase_window_factor * window_size)
        & (log_flux > percentiles[0])
        & (log_flux < percentiles[1])
    )

    # Make cutouts of the selected sources
    positions = catalog[selection_mask][["X_IMAGE", "Y_IMAGE"]]
    n_cutouts = len(positions)
    cutouts = np.empty((n_cutouts, psf_size, psf_size), dtype=image.dtype)
    for i, (x, y) in enumerate(positions):
        cutout = Cutout2D(image, (x, y), psf_size,
                          mode="partial", fill_value=np.nan)
        cutouts[i] = cutout.data

    # Average the cutouts to create the PSF
    psf = np.nanmean(cutouts, axis=0)

    # Plot the flux, flux radius plot with the selected sources highlighted
    if plot_chimney and plot_psf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

        # Left plot: chimney
        axes[0].scatter(log_flux_radius[mask], log_flux[mask],
                        color="b", s=1, alpha=0.5, label="Sources")
        axes[0].scatter(
            log_flux_radius[selection_mask],
            log_flux[selection_mask],
            s=1,
            alpha=0.5,
            color="r",
            label="Selected Sources for PSF",
        )
        axes[0].set_xlabel("log(flux radius)")
        axes[0].set_ylabel("log flux")
        axes[0].legend()
        axes[0].set_title("Chimney Plot")

        # Right plot: PSF
        im = axes[1].imshow(np.log(psf), cmap="gray")
        axes[1].set_title("PSF")
        fig.colorbar(im, ax=axes[1], fraction=0.046,
                     pad=0.04)  # optional colorbar

        plt.tight_layout()
        plt.show()

    else:
        # Individual plots as before
        if plot_chimney:
            plt.scatter(log_flux_radius[mask], log_flux[mask],
                        color="b", s=1, alpha=0.5, label="Sources")
            plt.scatter(
                log_flux_radius[selection_mask],
                log_flux[selection_mask],
                s=1,
                alpha=0.5,
                color="r",
                label="Selected Sources for PSF",
            )
            plt.xlabel("log(flux radius)")
            plt.ylabel("log flux")
            plt.legend()
            plt.show()

        if plot_psf:
            plt.imshow(np.log(psf), cmap="gray")
            plt.show()

    # Normalize the PSF
    psf /= np.sum(psf)

    return psf


@njit
def weighted_variance_lag(weight: np.ndarray, C_local: np.ndarray, max_lag: int) -> float:
    """
    Calculate the variance for the correlated noise for GAAP

    Parameters
    ----------
    weight: np.ndarray
        Weight function
    C_local: np.ndarray
        Local covariance matrix 
    max_lag: int
        Maximum pixel difference where the pixel are still correlated

    Returns
    -------
    float
        Variance in the flux from the correlated background noise

    Raises
    ------
    ValueError
        Description of when this error is raised
    """
    H, W = weight.shape
    V = 0.0
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            y0 = max(0, -dy)
            y1 = min(H, H - dy)
            x0 = max(0, -dx)
            x1 = min(W, W - dx)
            s1 = weight[y0:y1, x0:x1]
            s2 = weight[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            V += np.sum(s1 * s2) * C_local[dy + max_lag, dx + max_lag]
    return V


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


class NoiseModel:
    def __init__(
        self,
        image: np.ndarray | None = None,
        rms: np.ndarray | None = None,
        image_conversion_factor: float = 1.0,
        rms_conversion_factor: float = 1.0,
    ) -> None:

        self.image = image
        self.rms = rms
        self.image_conversion_factor = image_conversion_factor
        self.rms_conversion_factor = rms_conversion_factor

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

        local_mean = uniform_filter(image, size=box_size)
        local_var = uniform_filter(image**2, size=box_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))

        half = box_size // 2
        local_std[:half, :] = np.inf
        local_std[-half:, :] = np.inf
        local_std[:, :half] = np.inf
        local_std[:, -half:] = np.inf

        cy, cx = np.unravel_index(
            np.nanargmin(local_std),
            local_std.shape,
        )

        ny, nx = image.shape
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(ny, y0 + box_size)
        x1 = min(nx, x0 + box_size)

        self.noise_square = image[y0:y1, x0:x1]

    def create_poisson_image(self) -> None:
        """
        Extract the Poisson noise image from the RMS and store in
        """
        if self.rms is None:
            self.poisson_image = np.zeros_like(self.image)
        else:
            negative_pixels = self.image[self.image < 0]

            background_variance = (
                np.sum(negative_pixels**2) / len(negative_pixels)
            ) * self.image_conversion_factor**2

            self.poisson_image = np.clip(
                (self.rms * self.rms_conversion_factor) ** 2
                - background_variance,
                0,
                None,
            )

    def set_noise_square(self, noise_square):
        self.noise_square = noise_square
        self.poisson_image = np.zeros_like(noise_square)

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
        self.noise_covariance = self.covariance_fft2d(image, maxlag)

    def covariance_fft2d(self, noise_image: np.ndarray, maxlag: int) -> np.ndarray:
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

    def check_maxlag(self, percentage: float) -> float:
        """
        Calculate the smallest cutout of the covariance matrix that 
        sums to more than [percentage] of the sum of the total covariance matrix

        Parameters
        ----------
        percentage: float
            Minimum percentage
        maxlag: int
            Maximum pixel difference where the pixel are still correlated

        Returns
        -------
        float
            Local covariance matrix
        """
        h, w = self.ac.shape
        cy, cx = h // 2, w // 2
        total = np.sum(self.ac)
        for i in range(cy):
            sub_total = np.sum(self.ac[cy - i: cy + i + 1,
                                       cx - i: cx + i + 1,])
            if sub_total / total >= percentage:
                return i
        return cy

    def rms_error(self, weight, xc, yc, size, uncorrelated=False):
        if self.kernel is None:
            self.kernel = self.ac / np.max(self.ac)

        rms_cutout, _ = padded_cutout_with_center(self.rms, xc, yc, size)
        rms_cutout = rms_cutout[:weight.shape[0], :weight.shape[1]]
        weight_prime = rms_cutout * weight * self.rms_conversion_factor
        if uncorrelated:
            return np.sum(weight_prime * weight_prime)
        conv = fftconvolve(weight_prime, self.kernel, mode='same')
        return np.sum(weight_prime * conv)


class PSFDeconvolver:
    def __init__(self, psf: np.ndarray):
        self.psf = psf
        self.psf_cache = None
        self._fft_buffer = None

    def prepare(self, image_shape: tuple[int, int], K: float = 1e-16) -> dict[str, float | tuple[int, int]]:
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

        psf = self.psf[::-1, ::-1]

        pad_shape = (
            image_shape[0] + psf.shape[0] - 1,
            image_shape[1] + psf.shape[1] - 1,
        )

        psf_padded = np.zeros(pad_shape, dtype=np.float64)

        y0 = pad_shape[0] // 2 - psf.shape[0] // 2
        x0 = pad_shape[1] // 2 - psf.shape[1] // 2

        psf_padded[
            y0:y0 + psf.shape[0],
            x0:x0 + psf.shape[1],
        ] = psf

        H = rfft2(ifftshift(psf_padded))
        H_conj = np.conj(H)
        denom = (H * H_conj) + K

        self.psf_cache = {
            "PSF_prefactor": H_conj / denom,
            "pad_shape": pad_shape,
        }

    def deconvolve(self, weight: np.ndarray, size: int) -> np.ndarray:
        """
        Deconvolve the weight function by the stored PSF

        Parameters
        ----------
        weight: np.ndarray
            Weight function that is deconvolved
        size: int
            Size of the cutout

        Returns
        -------
        np.ndarray
            Deconvolved weight function

        Raises
        ------
        ValueError
            Description of when this error is raised
        """
        pad_shape = self.psf_cache["pad_shape"]

        if self._fft_buffer is None:
            test = rfft2(weight, pad_shape)
            self._fft_buffer = np.empty(test.shape, dtype=np.complex128)

        self._fft_buffer = rfft2(weight, pad_shape)
        self._fft_buffer *= self.psf_cache["PSF_prefactor"]

        result = irfft2(self._fft_buffer, pad_shape)
        return result[:size, :size]


class GAAPPhotometry:
    def __init__(
        self,
        image: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        pixel_scale: float = 1.0,
        image_conversion_factor: float = 1.0,
    ) -> None:

        self.image = image
        self.centers = centers
        self.sigmas = sigmas / pixel_scale
        self.pixel_scale = pixel_scale
        self.image_conversion_factor = image_conversion_factor

        self.ny, self.nx = image.shape
        self.flux = None
        self.variance = None

    def measure(
        self,
        size: int,
        maxlag: int,
        noise_model: NoiseModel,
        deconvolver: PSFDeconvolver,
        show_progress=True,
        uncorrelated=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the GAAP flux of the sources and weight functions.

        Parameters
        ----------
        size: int
            Description of param1
        maxlag: int
            Description of param2
        noise_model: NoiseModel
            Instance of NoiseModel
        deconvolver: PSFDeconvolver
            Instance of PSFDeconvolver

        Returns
        -------
        type
            Description of return value

        Raises
        ------
        ValueError
            Description of when this error is raised
        """
        x_c, y_c = self.centers
        n_sources = self.centers.shape[1]

        self.flux = np.full(n_sources, np.nan, dtype=np.float64)
        self.variance = np.full(n_sources, np.nan, dtype=np.float64)

        mask = (
            (x_c >= 0)
            & (x_c < self.nx)
            & (y_c >= 0)
            & (y_c < self.ny)
        )

        valid_idx = np.where(mask)[0]
        sorted_idx = valid_idx[
            np.argsort(self.sigmas[valid_idx])
        ]

        last_aperture = None

        for j, i in enumerate(tqdm(sorted_idx, desc="Measuring flux", disable=not show_progress)):
            xc, yc = x_c[i], y_c[i]
            aperture = self.sigmas[i]

            if aperture != last_aperture:
                weight = gaussian_weight(
                    size,
                    size,
                    size / 2,
                    size / 2,
                    aperture,
                )

                weight_rescale = deconvolver.deconvolve(weight, size)

                if noise_model.rms is None:
                    background_variance = weighted_variance_lag(
                        weight_rescale,
                        noise_model.noise_covariance,
                        maxlag,
                    )

                last_aperture = aperture

            cutout, (cx_cut, cy_cut) = padded_cutout_with_center(
                self.image,
                xc,
                yc,
                size,
            )

            cutout *= self.image_conversion_factor

            dx = cx_cut - np.floor(cx_cut)
            dy = cy_cut - np.floor(cy_cut)

            W_shifted = bilinear_shift(weight_rescale, dx, dy)

            cutout_trim = cutout[:W_shifted.shape[0], :W_shifted.shape[1]]

            self.flux[i] = np.sum(
                W_shifted * cutout_trim)

            if noise_model.rms is None:
                self.variance[i] = background_variance
            else:
                self.variance[i] = noise_model.rms_error(W_shifted, xc,
                                                         yc,
                                                         size, uncorrelated)

            if j % 2500 == 0:
                gc.collect()

        return self.flux, np.sqrt(self.variance)
