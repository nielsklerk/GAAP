from astropy.nddata import Cutout2D
from scipy.ndimage import uniform_filter
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from numpy.fft import rfft2, irfft2, ifftshift
import numpy as np
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def find_noise_square(image, box_size=50):
    """
    Automatically find a square region with low signal (noise-dominated).
    image: 2D array
    box_size: side of the square region (pixels)
    margin: exclude edges
    threshold: how many sigma above the median to consider as "source"
    returns: (y0, x0, y1, x1) slice indices of best noise square
    """
    img = np.asarray(image, float)
    h, w = img.shape

    # smooth absolute value to find low-variance zones
    local_mean = uniform_filter(img, size=box_size)
    local_var = uniform_filter(img**2, size=box_size) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))

    # exclude borders
    half = box_size // 2
    local_std[:half, :] = np.inf
    local_std[-half:, :] = np.inf
    local_std[:, :half] = np.inf
    local_std[:, -half:] = np.inf

    # pick minimum std region (least structured)
    cy, cx = np.unravel_index(np.nanargmin(local_std), local_std.shape)

    # ensure square fits inside image
    half = box_size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + box_size)
    x1 = min(w, x0 + box_size)

    return image[y0:y1, x0:x1]


def estimate_sigma(noise_image, weight, maxlag):
    local_covariance = covariance_fft2d(noise_image, maxlag)
    variance = weighted_variance_lag(weight, local_covariance, maxlag)
    return np.sqrt(variance)


def covariance_fft2d(image, maxlag):
    """
    Compute 2D covariance by FFT of background-subtracted residual.
    - image: 2D array (float)
    - maxlag: integer; returns covariance for lags -maxlag..+maxlag in both axes
    Returns: cov (2*maxlag+1, 2*maxlag+1) centered at lag (0,0)
    """
    img = image.astype(float)
    h, w = img.shape
    img -= np.mean(img)

    ac = fftconvolve(img, img[::-1, ::-1], mode="same")

    ac_norm = ac / (h * w)

    cy, cx = h//2, w//2
    window = ac_norm[cy-maxlag:cy+maxlag+1, cx-maxlag:cx+maxlag+1]
    return window


# @njit
def weighted_variance_lag(s, C_local, max_lag):
    H, W = s.shape
    V = 0.0
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            y0 = max(0, -dy)
            y1 = min(H, H - dy)
            x0 = max(0, -dx)
            x1 = min(W, W - dx)
            s1 = s[y0:y1, x0:x1]
            s2 = s[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            V += np.sum(s1 * s2) * C_local[dy + max_lag, dx + max_lag]
    return V

# @njit(fastmath=True)


def gaussian_1d(n, center, sigma):
    x = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def gaussian_weight(height, width, xc=0, yc=0, a=1):
    gx = gaussian_1d(width,  xc, a)
    gy = gaussian_1d(height, yc, a)

    weight = gy[:, None] * gx[None, :]
    weight /= weight.sum()
    return weight


def prepare_wiener_psf(psf, image_shape, K=0.0, dtype=np.float64):
    """
    Precompute PSF FFT terms for Wiener deconvolution.
    """

    psf = psf[::-1, ::-1].astype(dtype, copy=False)

    pad_shape = (
        image_shape[0] + psf.shape[0] - 1,
        image_shape[1] + psf.shape[1] - 1,
    )

    psf_padded = np.zeros(pad_shape, dtype=dtype)
    y0 = pad_shape[0] // 2 - psf.shape[0] // 2
    x0 = pad_shape[1] // 2 - psf.shape[1] // 2
    psf_padded[y0:y0 + psf.shape[0], x0:x0 + psf.shape[1]] = psf

    H = rfft2(ifftshift(psf_padded))
    H_conj = np.conj(H)
    denom = (H * H_conj) + K

    return {
        "H_conj": H_conj,
        "denom": denom,
        "pad_shape": pad_shape,
        "image_shape": image_shape,
    }


def wiener_deconvolution_fast(weight, psf_cache, dtype=np.float64):
    """
    Fast Wiener deconvolution using cached PSF FFT data.
    """

    weight = weight.astype(dtype, copy=False)
    W = rfft2(weight, psf_cache["pad_shape"])
    F = psf_cache["H_conj"] * W / psf_cache["denom"]
    result = irfft2(F, psf_cache["pad_shape"])

    h, w = psf_cache["image_shape"]
    return result[:h, :w]


def calculate_gaap_flux(image, psf_cache, weight, centers):
    """
    Placeholder function for flux calculation.
    """
    weight_rescale = wiener_deconvolution_fast(weight, psf_cache)
    flux_map = fftconvolve(image, weight_rescale[::-1, ::-1], mode='same')

    centers = np.asarray(centers)
    ys = centers[:, 1]
    xs = centers[:, 0]

    valid = np.isfinite(xs) & np.isfinite(ys)
    measured_F = np.full(len(centers), np.nan, dtype=np.float32)
    measured_F[valid] = map_coordinates(
        flux_map, [ys[valid], xs[valid]], order=1)

    return measured_F, weight_rescale


def gaussian_2d(xy, amplitude, sigma, x0, y0):
    x, y = xy
    r_2 = (x - x0)**2 + (y - y0)**2
    g = amplitude * np.exp(
        -r_2/(2*sigma**2)
    )
    return g.ravel()


def padded_cutout_with_center(image, cx, cy, size):
    """
    Extract a fixed-size cutout centered on (cy, cx).
    Pads with zeros when the cutout extends beyond the image.

    Returns
    -------
    cutout : (size, size) array
    (cy_c, cx_c) : subpixel center in cutout coordinates
    """

    h, w = image.shape
    half = size // 2

    # Integer anchor
    iy = int(np.floor(cy))
    ix = int(np.floor(cx))

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
    cutout = np.zeros((size, size), dtype=image.dtype)

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
    Create a point spread function (PSF) using saturated stars identified in a catalog.

    Args:
        image (np.ndarray): 2D image array.
        catalog (Table): Source catalog containing star positions (must include at least x, y).
        psf_size (int): Final PSF cutout size in pixels.
        window_size (float, optional): Initial window size factor used around saturated stars. Defaults to 0.1.
        lower_percentile (float, optional): Lower percentile for saturation thresholding. Defaults to 98.0.
        upper_percentile (float, optional): Upper percentile for saturation thresholding. Defaults to 99.9.
        increase_window_factor (float, optional): Factor by which the window may grow if needed. Defaults to 2.
        plot_chimney (bool, optional): If True, plot each star cutout (chimney plot). Defaults to False.
        plot_psf (bool, optional): If True, plot the final PSF image. Defaults to False.

    Returns:
        np.ndarray: The computed PSF image with shape (psf_size, psf_size).
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
