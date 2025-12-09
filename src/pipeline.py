from astropy.io import fits
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from astropy.nddata import Cutout2D
from astropy.table import Table
import matplotlib.pyplot as plt
from functools import lru_cache
from joblib import Memory
from scipy.ndimage import uniform_filter
memory = Memory("/net/vdesk/data2/deklerk/GAAP_data/cache_dir", verbose=0)

def find_noise_square(image, box_size=50, margin=3):
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
    local_std[:margin, :] = np.inf
    local_std[-margin:, :] = np.inf
    local_std[:, :margin] = np.inf
    local_std[:, -margin:] = np.inf

    # pick minimum std region (least structured)
    cy, cx = np.unravel_index(np.nanargmin(local_std), local_std.shape)

    # ensure square fits inside image
    half = box_size // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + box_size)
    x1 = min(w, x0 + box_size)

    return y0, x0, y1, x1


def estimate_sigma(noise_image, weight, maxlag):
    local_covariance = covariance_fft2d(noise_image, maxlag)
    negative_pixels = noise_image[noise_image<0]
    uncorrelated_variance = np.sum(negative_pixels**2)/len(negative_pixels)
    local_covariance = local_covariance / local_covariance[maxlag, maxlag] * uncorrelated_variance
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

    # straightforward FFT autocorrelation
    F = fft2(img)
    ac = fftshift(ifft2(F * np.conj(F)).real)
    ac_norm = ac / (h * w)

    cy, cx = h//2, w//2
    window = ac_norm[cy-maxlag:cy+maxlag+1, cx-maxlag:cx+maxlag+1]
    return window



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

@lru_cache(maxsize=None)
def gaussian_1d(n, center, sigma):
    x = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def gaussian_weight(height, width, xc=0, yc=0, a=1):
    gx = gaussian_1d(width,  xc, a)
    gy = gaussian_1d(height, yc, a)

    weight = gy[:, None] * gx[None, :]
    weight /= weight.sum()
    return weight

@memory.cache
def wiener_deconvolution(weight, psf, K=0.01, dtype=np.float64):
    """
    Perform Wiener deconvolution on an weight function using the given PSF.
    """
    print("Deconvolving weight function")
    # Convert to wanted dtype
    weight = weight.astype(dtype, copy=False)
    psf = psf[::-1, ::-1].astype(dtype, copy=False)

    # Compute padded shape
    pad_shape = [s1 + s2 - 1 for s1, s2 in zip(weight.shape, psf.shape)]

    # Center PSF in padded array
    psf_padded = np.zeros(pad_shape, dtype=dtype)
    y0 = pad_shape[0] // 2 - psf.shape[0] // 2
    x0 = pad_shape[1] // 2 - psf.shape[1] // 2
    psf_padded[y0 : y0 + psf.shape[0], x0 : x0 + psf.shape[1]] = psf

    # Compute FFTs of image and PSF
    psf_fft = fft2(ifftshift(psf_padded))
    weight_fft = fft2(weight, pad_shape)

    # Wiener deconvolve
    denom = (psf_fft * np.conj(psf_fft)) + K
    result_fft = (np.conj(psf_fft) * weight_fft) / denom
    result = np.real(ifft2(result_fft))

    # Crop back to original image size
    result = result[:weight.shape[0], :weight.shape[1]]
    return result


def find_flux(
    image_path: str,
    telescope: str,
    weight_size: float,
    ra: np.ndarray,
    dec: np.ndarray,
    correlated: int = None,
    psf_path: str = None,
    catalog_path: str = None,
    psf_size: float = 25,
    tilesize: int = 100,
    noise_y0: int = 0,
    noise_y1: int = -1,
    noise_x0: int = 0,
    noise_x1: int = -1
) -> tuple[np.ndarray, float]:
    """Find fluxes in an image at given coordinates using weighted aperture photometry.

    Args:
        path (str): Path to the image file.
        telescope (str): Name of the telescope ("Rubin" or "Euclid").
        weight_size (float): Size of the Gaussian weight function.
        ra (np.ndarray): Array of ra.
        dec (np.ndarray): Array of dec
        correlated (float, optional): Correlation length for noise estimation. Defaults to None.
        psf_path (str, optional): Path to the PSF file. If None, a PSF will be created from the image. Defaults to None.
        catalog_path (str, optional): Path to the catalog file. *.cat has to contain "FLUX_RADIUS", "FLUX_AUTO", "X_IMAGE", "Y_IMAGE".
        psf_size (float, optional): Size of the PSF to create if psf_path is None. Defaults to 25.
        tilesize (int, optional): Size of tiles for local flux measurement. If None, use full image. Defaults to None.
    Returns:
        np.ndarray: Array of fluxes at the given coordinates.
        float: Estimated noise sigma.
    """
    if psf_path == None and catalog_path == None:
        raise Exception("Need either psf_path or catalog_path")

    print("Analyzing image")
    # Open the image and extract useful information
    with fits.open(image_path) as hdul:
        hdu = hdul[1 if telescope == "Rubin" else 0]
        image = hdu.data
        wcs = WCS(hdu.header)
        nx = hdu.header["NAXIS1"]
        ny = hdu.header["NAXIS2"]

    noise_image = image[noise_y0: noise_y1, noise_x0:noise_x1]

    # Convert the image to μJy if in AB magnitude
    try:
        zeropoint = hdu.header["MAGZERO"]
        conversion_factor = 10 ** ((8.90 - zeropoint) / 2.5) * 10**9
    except:
        conversion_factor = 1
    image *= conversion_factor

    # Translate the ra and dec into pixel coordinates
    x, y = wcs.wcs_world2pix(ra, dec, 0, ra_dec_order=True)
    mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    x[~mask] = None
    y[~mask] = None

    if telescope == "Rubin":
        # Make weight map
        intrinsic_weight = gaussian_weight(
            nx, ny, nx // 2, ny // 2, weight_size / 2,
        )

        # Load PSF and if not available create one
        if psf_path is not None:
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data
        else:
            print("Creating PSF")
            catalog = Table.read(catalog_path, format="ascii")
            psf = create_psf(image, catalog_path, psf_size)
        
        # Calculate weight map and apply to image
        weight = wiener_deconvolution(intrinsic_weight, psf)
        flux_map = fftconvolve(image, weight[::-1, ::-1], mode="same")

        # Extracting fluxes from image
        measured_fluxes = map_coordinates(flux_map, [y, x], order=1)

        # Calculate error on the flux
        if correlated is not None:
            sigma = estimate_sigma(noise_image, weight, correlated)
        else:
            negative_pixels = image[image < 0].flatten()
            sigma = np.sqrt(
                np.sum(negative_pixels**2)
                * np.sum(weight**2)
                / len(negative_pixels)
            )

    else:
        # Load PSF and if not available create one
        if psf_path is not None:
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data
        else:
            print("Creating PSF")
            catalog = Table.read(catalog_path, format="fits", hdu=2)
            psf = create_psf(image, catalog, psf_size)

        # Set up boundaries of tiles
        x_edges = np.arange(0, nx + 1, tilesize)
        y_edges = np.arange(0, ny + 1, tilesize)

        # Calculate the flux and error of each tile
        measured_fluxes = np.full(len(x), np.nan)
        sigma_tiles = []
        weight = None
        for y_start, y_end in zip(y_edges[:-1], y_edges[1:]):
            for x_start, x_end in zip(x_edges[:-1], x_edges[1:]):
                print(
                    f"Processing x=({x_start},{x_end}), y=({y_start},{y_end})           ",
                    end="\r",
                    flush=True,
                )

                # Create tile
                image_cut = image[y_start:y_end, x_start:x_end]

                # Calculate weight function for the tile if not already calculated
                if weight is None:
                    intrinsic_weight = gaussian_weight(
                        image_cut.shape[0],
                        image_cut.shape[1],
                        image_cut.shape[0] / 2,
                        image_cut.shape[1] / 2,
                        weight_size
                    )
                    weight = wiener_deconvolution(intrinsic_weight, psf)

                # Apply weight to tile
                flux_tile = fftconvolve(image_cut, weight[::-1, ::-1], mode="same")

                # Select the sources that are in the image
                mask_tile = (x >= x_start) & (x < x_end) & (y >= y_start) & (y < y_end)
                idx_tile = np.where(mask_tile)[0]
                if len(idx_tile) == 0:
                    continue
                
                # Translate image coordinates to tile coordinates
                xt = x[idx_tile] - x_start
                yt = y[idx_tile] - y_start

                # Extract the flux
                measured_fluxes[idx_tile] = map_coordinates(
                    flux_tile, [yt, xt], order=1
                )

                # Calculate the error
                if correlated is not None:
                    sigma_local = estimate_sigma(noise_image, weight, correlated)
                else:
                    neg = image_cut[image_cut < 0].ravel()
                    sigma_local = np.sqrt(
                        np.sum(neg**2) * np.sum(weight**2) / len(neg)
                    )

                # Store error in tile
                sigma_tiles.append(sigma_local)

        # Calculate the error for the whole image
        sigma = np.mean(sigma_tiles) 
        
        # Factor 4 to convert to the same pixel scale as Rubin
        measured_fluxes *= 4
        sigma *= 4
        print()

    return measured_fluxes, sigma


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
    percentiles = np.percentile(log_flux[selection_mask], [lower_percentile, upper_percentile])
    selection_mask = (
        np.isfinite(log_flux_radius)
        & np.isfinite(log_flux)
        & (log_flux_radius > center_maximum - increase_window_factor * window_size)
        & (log_flux_radius < center_maximum + increase_window_factor * window_size)
        & (log_flux > percentiles[0])
        & (log_flux < percentiles[1])
    )

    # Plot the flux, flux radius plot with the selected sources highlighted
    if plot_chimney:
        plt.scatter(log_flux_radius[mask], log_flux[mask], color="b", s=1, alpha=0.5, label="Sources")
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
    
    # Make cutouts of the selected sources
    positions = catalog[selection_mask][["X_IMAGE", "Y_IMAGE"]]
    n_cutouts = len(positions)
    cutouts = np.empty((n_cutouts, psf_size, psf_size), dtype=image.dtype)
    for i, (x, y) in enumerate(positions):
        cutout = Cutout2D(image, (x, y), psf_size, mode="partial", fill_value=np.nan)
        cutouts[i] = cutout.data
    
    # Average the cutouts to create the PSF
    psf = np.nanmean(cutouts, axis=0)

    # Plot the PSF
    if plot_psf:
        plt.imshow(psf, cmap="gray")
        plt.show()

    # Normalize the PSF
    psf /= np.sum(psf)

    return psf


def find_correlated_sigma(*args):
    raise NotImplementedError("This function is yet to be created")
