import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def gaussian_weight(height, width, xc=0, yc=0, a=0, b=0):
    """
    Compute a Gaussian weight map centered at (xc, yc).
    """
    y, x = np.indices((height, width), dtype=float)
    weight = np.exp(-0.5 * (((x - xc) / a) ** 2 + ((y - yc) / b) ** 2))
    return weight / weight.sum()

def wiener_deconvolution(img, psf, K=0.01, dtype=np.float32):
    """
    Memory-efficient Wiener deconvolution.

    Parameters
    ----------
    img : 2D array
        Input image or weight function.
    psf : 2D array
        Point-spread function (must be same or smaller size).
    K : float
        Wiener constant (noise-to-signal power ratio).
    dtype : np.dtype
        Use np.float32 to save memory.

    Returns
    -------
    result : 2D array
        Deconvolved image cropped to original shape.
    """
    # Convert to desired dtype
    img = img.astype(dtype, copy=False)
    psf = psf.astype(dtype, copy=False)

    # Normalize PSF
    psf = psf / psf.sum()

    # Compute padded shape (minimal)
    pad_shape = [s1 + s2 - 1 for s1, s2 in zip(img.shape, psf.shape)]

    # Center PSF in padded array
    psf_padded = np.zeros(pad_shape, dtype=dtype)
    y0 = pad_shape[0]//2 - psf.shape[0]//2
    x0 = pad_shape[1]//2 - psf.shape[1]//2
    psf_padded[y0:y0+psf.shape[0], x0:x0+psf.shape[1]] = psf

    # Compute FFTs
    psf_fft = fft2(ifftshift(psf_padded))
    img_fft = fft2(img, pad_shape)

    # Wiener filter
    denom = (psf_fft * np.conj(psf_fft)) + K
    result_fft = (np.conj(psf_fft) * img_fft) / denom
    result = np.real(ifft2(result_fft))

    # Crop back to original image size
    result = result[:img.shape[0], :img.shape[1]]
    return result