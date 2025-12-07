import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import pandas as pd
from functools import lru_cache
from joblib import Memory
memory = Memory("/net/vdesk/data2/deklerk/GAAP_data/cache_dir", verbose=0)


@lru_cache(maxsize=None)
def gaussian_1d(n, center, sigma):
    x = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def gaussian_weight(height, width, xc=0, yc=0, a=1, b=1):
    gx = gaussian_1d(width,  xc, a)
    gy = gaussian_1d(height, yc, b)

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