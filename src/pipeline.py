from astropy.io import fits
import numpy as np
from src.analysis import gaussian_weight, wiener_deconvolution
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from astropy.nddata import Cutout2D
from astropy.table import Table
import matplotlib.pyplot as plt

def find_flux(image_path: str,
              telescope: str,
              weight_size: float,
              ra: np.ndarray,
              dec: np.ndarray,
              correlated: float = None,
              psf_path:str = None,
              catalog_path: str = None,
              psf_size: float = 25,
              tilesize:int = 100
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
        psf_size (float, optional): Size of the PSF to create if psf_path is None. Defaults to 25.
        tilesize (int, optional): Size of tiles for local flux measurement. If None, use full image. Defaults to None.
    Returns:
        np.ndarray: Array of fluxes at the given coordinates.
        float: Estimated noise sigma.
    """
    print('Analyzing image')
    with fits.open(image_path) as hdul:
        hdu = hdul[1 if telescope == "Rubin" else 0]
    image = hdu.data
    wcs = WCS(hdu.header)
    nx = hdu.header["NAXIS1"]
    ny = hdu.header["NAXIS2"]

    try:
        zeropoint = hdu.header['MAGZERO']
        conversion_factor = 10**((8.90 - zeropoint)/2.5) * 10**9
    except:
        conversion_factor = 1
    image *= conversion_factor
    

    x, y = wcs.wcs_world2pix(ra, dec, 0, ra_dec_order=True)
    mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    x[~mask] = None
    y[~mask] = None

    if telescope == 'Rubin':
        intrinsic_weight = gaussian_weight(nx, ny, nx//2, ny//2, weight_size/2, weight_size/2)
        if psf_path is not None:
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data
        else:
            print('Creating PSF')
            catalog = Table.read(catalog_path, format='ascii')
            psf = create_psf(image, catalog_path, psf_size)
        
        print('Deconvolving weight function')
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
            print('Creating PSF')
            catalog = Table.read(catalog_path, format='fits', hdu=2)
            psf = create_psf(image, catalog, psf_size)
        
        x_edges = np.arange(0, nx + 1, tilesize)
        y_edges = np.arange(0, ny + 1, tilesize)

        measured_fluxes = np.full(len(x), np.nan)
        sigma_tiles = []
        weight = None
        for y_start, y_end in zip(y_edges[:-1], y_edges[1:]):
            for x_start, x_end in zip(x_edges[:-1], x_edges[1:]):
                print(f"Processing x=({x_start},{x_end}), y=({y_start},{y_end})           ", end='\r', flush=True)
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
                
                flux_tile = fftconvolve(image_cut, weight[::-1, ::-1], mode='same')

                mask_tile = (
                    (x >= x_start) & (x < x_end) &
                    (y >= y_start) & (y < y_end)
                    )
                idx_tile = np.where(mask_tile)[0]
                if len(idx_tile) == 0:
                    continue

                xt = x[idx_tile] - x_start
                yt = y[idx_tile] - y_start


                measured_fluxes[idx_tile] = map_coordinates(flux_tile, [yt, xt], order=1)

                if correlated is not None:
                    sigma_local = find_correlated_sigma(image_cut, correlated)
                else:
                    neg = image_cut[image_cut < 0].ravel()
                    sigma_local = np.sqrt(
                        np.sum(neg**2) * np.sum(weight**2) / len(neg)
                    )

                sigma_tiles.append(sigma_local)

        sigma = np.mean(sigma_tiles)
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
    plot_chimney: bool = False,
    plot_psf: bool = False
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
    x = np.log(catalog['FLUX_RADIUS'])
    y = np.log(catalog['FLUX_AUTO'])
    mask = np.isfinite(x) & np.isfinite(y)
    maximum = -np.inf
    x_0_maximum = -np.inf
    for x_0 in np.linspace(min(x[mask]), max(x[mask]), 100):
        new_mask = np.isfinite(x) & np.isfinite(y) & (x > x_0 - window_size) & (x < x_0 + window_size) & (y > 8)
        if np.sum(y[new_mask]) > maximum:
            maximum = np.sum(y[new_mask])
            x_0_maximum = x_0
    selection_mask = np.isfinite(x) & np.isfinite(y) & (x > x_0_maximum - increase_window_factor*window_size) & (x < x_0_maximum + increase_window_factor*window_size)
    percentiles = np.percentile(y[selection_mask], [lower_percentile, upper_percentile])
    selection_mask = np.isfinite(x) & np.isfinite(y) & (x > x_0_maximum - increase_window_factor*window_size) & (x < x_0_maximum + increase_window_factor*window_size) & (y > percentiles[0]) & (y < percentiles[1])
    if plot_chimney:
        plt.scatter(x[mask], y[mask], color='b', s=1, alpha=0.5, label='Sources')
        plt.scatter(x[selection_mask], y[selection_mask], s=1, alpha=0.5, color='r', label='Selected Sources for PSF')
        plt.xlabel('log(flux radius)')
        plt.ylabel('log flux')
        plt.legend()
        plt.show()
    positions = catalog[selection_mask][['X_IMAGE', 'Y_IMAGE']]
    n_cutouts = len(positions)
    cutouts = np.empty((n_cutouts, psf_size, psf_size), dtype=image.dtype)
    for i, (x, y) in enumerate(positions):
        cutout = Cutout2D(image, (x, y), psf_size, mode='partial', fill_value=np.nan)
        cutouts[i] = cutout.data
    psf = np.nanmean(cutouts, axis=0)
    if plot_psf:
        plt.imshow(psf, cmap='gray')
        plt.show()
    return psf/np.sum(psf)

def find_correlated_sigma(*args):
    raise NotImplementedError('This function is yet to be created')