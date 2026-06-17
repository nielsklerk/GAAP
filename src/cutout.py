from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob

Image.MAX_IMAGE_PIXELS = 500_000_000

data_folder = '/net/ketelmeer/data2/kuijken/Q1'
catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'

def plot_cutout(tileindex, locations, cutout_size, axes, type=None):
    file = glob(f'{data_folder}/VIS_TILE{tileindex}.fits.gz')[0]
    with fits.open(file, memmap=True) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
    centers = wcs.wcs_world2pix(
        locations[:, 0], locations[:, 1], 0, ra_dec_order=True)

    if type is None:
        image_file = glob(f'{data_folder}/tile{tileindex}_40_99.jpg')[0]
    else:
        image_file = glob(f'{data_folder}/tile{tileindex}_{type}_40_99.jpg')[0]

    half = cutout_size // 2
    axes = axes.flatten()
    with Image.open(image_file) as img:
        width, height = img.size
        for i in range(len(centers[0])):

            x, y = centers[0][i], centers[1][i]

            left   = int(x - half)
            right  = int(x + half)
            top    = int(height - y + half)
            bottom = int(height - y - half)

            cutout = img.crop((left, bottom, right, top))

            axes[i].imshow(cutout)
            axes[i].axis('off')

if __name__=="__main__":
    locations = np.array([
    (274.749950, 67.252568),
    (275.222359, 67.259737),
    (275.039792, 67.287124),
    (274.792398, 67.286938),
    (274.793943, 67.287596),
    (275.264818, 67.305663),
    (274.563444, 67.320804),
    (274.918806, 67.327054),
    (275.515444, 67.327609),
    (275.622314, 67.368195),
    (275.622982, 67.368213),
    (275.354902, 67.380435),
    (275.339297, 67.387174),
    (275.600201, 67.388079),
    (274.714089, 67.412362),
    (274.524091, 67.452957),
    (275.107087, 67.461309),
    (274.514540, 67.467275),
    (274.826254, 67.470852),
    (274.675967, 67.494380),
    (274.815899, 67.539186),
    (274.493953, 67.593438),
    (274.572393, 67.619576),
    (274.869951, 67.621907),
    (274.868818, 67.621953),
])
    type = [None, 'VISgri', 'HYJH'][2] #None/VISgri/HYJH
    fig, axes = plt.subplots(5, 5, figsize=(12,12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plot_cutout(102160063, locations, 128, axes, type=type)
    plt.savefig(f'/home/deklerk/GAAP/results/figures/observation/extended_stars_north_{type}.pdf', bbox_inches='tight')
    plt.show()