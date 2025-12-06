from astropy.io import fits
import numpy as np
from src.analysis import gaussian_weight, wiener_deconvolution
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates

def find_flux(image_path: str, telescope: str, weight_size: float, coordinates: np.ndarray, correlated: float = None, psf_path:str = None, psf_size: float = 25, tilesize:int = None) -> tuple[np.ndarray, float]:
    """Find fluxes in an image at given coordinates using weighted aperture photometry.

    Args:
        path (str): Path to the image file.
        telescope (str): Name of the telescope ("Rubin" or "Euclid").
        weight_size (float): Size of the Gaussian weight function.
        coordinates (np.ndarray): Nx2 array of (ra, dec) coordinates.
        correlated (float, optional): Correlation length for noise estimation. Defaults to None.
        psf_path (str, optional): Path to the PSF file. If None, a PSF will be created from the image. Defaults to None.
        psf_size (float, optional): Size of the PSF to create if psf_path is None. Defaults to 25.
        tilesize (int, optional): Size of tiles for local flux measurement. If None, use full image. Defaults to None.
    Returns:
        np.ndarray: Array of fluxes at the given coordinates.
        float: Estimated noise sigma.
    """
    with fits.open(image_path) as hdul:
        hdu = hdul[1 if telescope == "Rubin" else 0]
        image = hdu.data
    wcs = WCS(hdu.header)
    nx = hdu.header["NAXIS1"]
    ny = hdu.header["NAXIS2"]

    x, y = wcs.wcs_world2pix(coordinates[:, 0], coordinates[:, 1], 0)
    mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    x[~mask] = None
    y[~mask] = None

    if tilesize == nx:
        intrinsic_weight = gaussian_weight(nx, ny, nx//2, ny//2, weight_size, weight_size)
        if psf_path is not None:
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data
        else:
            psf = create_psf(image, psf_size)

        weight = wiener_deconvolution(intrinsic_weight, psf)
        flux_map = fftconvolve(image, weight[::-1, ::-1], mode='same')
        measured_fluxes = map_coordinates(flux_map, [y, x], order=1)
        if correlated is not None:
            sigma = find_correlated_sigma(image, correlated)
        else:
            negative_pixels = image[image<0].flatten()
            sigma = np.sqrt(np.sum(negative_pixels ** 2) * np.sum(weight ** 2) / len(negative_pixels))
    else:
        if psf_path is not None:
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data
        else:
            psf = create_psf(image, psf_size)
        
        x_edges = np.arange(0, nx + 1, tilesize)
        y_edges = np.arange(0, ny + 1, tilesize)

        measured_fluxes = np.full(len(x), np.nan)
        sigma_tiles = []       # collect per-tile sigmas
        weight = None
        for y_start, y_end in zip(y_edges[:-1], y_edges[1:]):
            for x_start, x_end in zip(x_edges[:-1], x_edges[1:]):

                image_cut = image[y_start:y_end, x_start:x_end]
                if weight is None:
                    intrinsic_weight = gaussian_weight(
                    image_cut.shape[0],
                    image_cut.shape[1],
                    image_cut.shape[0] / 2,
                    image_cut.shape[1] / 2,
                    weight_size,
                    weight_size
                    )
                    weight = wiener_deconvolution(intrinsic_weight, psf)
                # Convolve local tile with full reversed weight
                flux_tile = fftconvolve(image_cut, weight[::-1, ::-1], mode='same')

                # Points inside this tile
                mask_tile = (
                    (x >= x_start) & (x < x_end) &
                    (y >= y_start) & (y < y_end)
                    )
                idx_tile = np.where(mask_tile)[0]
                if len(idx_tile) == 0:
                    continue

                # Local tile coordinates
                xt = x[idx_tile] - x_start
                yt = y[idx_tile] - y_start

                # Flux sampling
                measured_fluxes[idx_tile] = map_coordinates(flux_tile, [yt, xt], order=1)

                # Per-tile sigma
                if correlated is not None:
                    sigma_local = find_correlated_sigma(image_cut, correlated)
                else:
                    neg = image_cut[image_cut < 0].ravel()
                    sigma_local = np.sqrt(
                        np.sum(neg**2) * np.sum(weight**2) / len(neg)
                    )

                sigma_tiles.append(sigma_local)

        # Final sigma = mean over tiles
        sigma = np.mean(sigma_tiles)

    return measured_fluxes, sigma

