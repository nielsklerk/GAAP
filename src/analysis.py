import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import pandas as pd


def gaussian_weight(height, width, xc=0, yc=0, a=0, b=0):
    """
    Compute a Gaussian weight map centered at (xc, yc).
    """
    y, x = np.indices((height, width), dtype=float)
    weight = np.exp(-0.5 * (((x - xc) / a) ** 2 + ((y - yc) / b) ** 2))
    return weight / weight.sum()


def wiener_deconvolution(weight, psf, K=0.01, dtype=np.float64):
    """
    Perform Wiener deconvolution on an weight function using the given PSF.
    """
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

def calculate_gaap_flux(image, psf, weight, centers):
    """
    Placeholder function for flux calculation.
    """
    weight_rescale = wiener_deconvolution(weight, psf, 0)
    flux_map = fftconvolve(image, weight_rescale[::-1, ::-1], mode='same')

    measured_F = np.zeros(len(centers))
    for i, (xc, yc) in enumerate(centers):
        measured_F[i] = map_coordinates(flux_map, [[yc], [xc]], order=1)

    x = image[image<0].flatten()

    sigma = np.sqrt(np.sum(x ** 2) * np.sum(weight_rescale ** 2) / len(x))

    return measured_F, sigma

def align_fluxes_by_reference(dfs, max_dist=2.0):
    # Find reference catalog (smallest)
    ref_band = min(dfs, key=lambda k: len(dfs[k]))
    ref_df = dfs[ref_band]
    x_ref = ref_df['x'].values
    y_ref = ref_df['y'].values
    ref_points = np.column_stack((x_ref, y_ref))

    bands = list(dfs.keys())
    flux_list = []

    for band in bands:
        df = dfs[band]
        x = df['x'].values
        y = df['y'].values
        points = np.column_stack((x, y))

        # Build KDTree and query nearest neighbor for each reference point
        tree = cKDTree(points)
        dist, idx = tree.query(ref_points, distance_upper_bound=max_dist)

        # Initialize flux array with NaNs
        flux_aligned = np.full(len(ref_points), np.nan)

        # Fill only valid matches
        valid = idx < len(df)
        flux_aligned[valid] = df['flux'].values[idx[valid]]

        flux_list.append(flux_aligned)

    # Combine into 2D array: rows = galaxies, columns = filters
    flux_matrix = np.column_stack(flux_list)
    return bands, flux_matrix