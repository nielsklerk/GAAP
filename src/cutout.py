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
    (61.832056, -46.743209),
    (61.541357, -46.741297),
    (61.935374, -46.736008),
    (61.809048, -46.731911),
    (61.900805, -46.727309),
    (61.665714, -46.711666),
    (61.605969, -46.692705),
    (62.193169, -46.687846),
    (62.103728, -46.679355),
    (62.217077, -46.668883),
    (62.138907, -46.656858),
    (62.073871, -46.647535),
    (62.092280, -46.641555),
    (62.194893, -46.622678),
    (62.151768, -46.596422),
    (62.182402, -46.584712),
    (62.121012, -46.577391),
    (61.947913, -46.572830),
    (62.147903, -46.571894),
    (61.959524, -46.569874),
    (61.751195, -46.569705),
    (62.208205, -46.546347),
    (62.112725, -46.540828),
    (61.623135, -46.537384),
    (62.070159, -46.530708),
    (62.162942, -46.523508),
    (62.186983, -46.507316),
    (61.829472, -46.503637),
    (61.747296, -46.502829),
    (61.557772, -46.498298),
    (61.805925, -46.497107),
    (62.199271, -46.494180),
    (61.794534, -46.492715),
    (62.132361, -46.465781),
    (62.099753, -46.459269),
    (62.130424, -46.438906),
    (61.622829, -46.432703),
    (61.724527, -46.397121),
    (62.212996, -46.395325),
    (61.746192, -46.375255),
    (61.839130, -46.369543),
    (62.064784, -46.369799),
    (61.794552, -46.369438),
    (61.587927, -46.361864),
    (62.168153, -46.357925),
    (62.188239, -46.354296),
    (62.188251, -46.354172),
    (62.188413, -46.354145),
    (62.188622, -46.353883),
    (61.580763, -46.345643),
    (61.840161, -46.341519),
    (61.899521, -46.331054),
    (62.229175, -46.330314),
    (62.230764, -46.324222),
    (61.807430, -46.323018),
    (62.019034, -46.320069),
    (61.882056, -46.319568),
    (62.087392, -46.307578),
    (61.749623, -46.288583),
    (62.174314, -46.279206),
    (61.858540, -46.279099),
    (61.803585, -46.269551),
    (61.562051, -46.267127),
    (62.075244, -46.265725),
])
    type = 'VISgri' #None/VISgri/HYJH
    fig, axes = plt.subplots(8, 8, figsize=(12,12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plot_cutout(102022975, locations, 128, axes, type=type)
    plt.savefig(f'/home/deklerk/GAAP/results/figures/observation/extended_star_{type}.pdf', bbox_inches='tight')
    plt.show()