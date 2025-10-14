import numpy as np
import cv2

def sersic(width, height, xc, yc, k, n, inclination=0, rotation=0, n_arms=0, winding=1, bulge_strength=1, Rc=30, direction=1):
    """
    Compute the raw Sersic galaxy (no inclination or rotation) centered at (xc, yc)
    """
    y, x = np.indices((width, height), dtype=float)
    x_rel = x - xc
    y_rel = y - yc

    r = np.sqrt(x_rel ** 2 + y_rel ** 2)
    theta = np.arctan2(y_rel, x_rel) * direction

    bulge = np.exp(-k * r ** (1 / n))
    if n_arms > 0:
        arms = 1 + np.cos(n_arms * (theta - winding * np.log(r + 1e-6)))
    else:
        arms = 0

    galaxy = bulge * bulge_strength + arms * np.exp(-(r / Rc) ** 2)


    scale_y = np.cos(inclination)
    M_incline = np.float32([
        [1, 0, 0],
        [0, scale_y, yc * (1 - scale_y)]
    ])
    galaxy_inclined = cv2.warpAffine(
        galaxy, M_incline, (width, height), flags=cv2.INTER_CUBIC
    )

    # Rotation around (xc, yc)
    M_rot = cv2.getRotationMatrix2D((xc, yc), np.degrees(rotation), 1.0)
    galaxy_final = cv2.warpAffine(
        galaxy_inclined, M_rot, (width, height), flags=cv2.INTER_CUBIC
    )

    return galaxy_final

def moffat(height, width, m: float, a: float,
                n_spikes: int = 0, rotation: float = 0.0,
                spike_strength: float = 2.0, spike_width: float = 4.0,
                Rc: float = 10.0):
    y, x = np.indices((height, width), dtype=float)
    xc, yc = width / 2, height / 2
    x_rel = x - xc
    y_rel = y - yc

    r = np.sqrt(x_rel**2 + y_rel**2)
    centre = (1 + (r / a)**2)**(-m)

    theta = np.arctan2(y_rel, x_rel) + rotation
    spikes = np.zeros_like(centre)
    for i in range(n_spikes):
        angle = i * 2 * np.pi / n_spikes
        dist = np.abs(np.cos(theta - angle))
        spikes += np.exp(-(dist / spike_width)**2)
    moffat = centre + spike_strength * spikes * np.exp(-(r / Rc)**2)

    return moffat/moffat.sum()

