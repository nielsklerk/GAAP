from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob

Image.MAX_IMAGE_PIXELS = 500_000_000

data_folder = '/net/ketelmeer/data2/kuijken/Q1'
catalog_folder = '/net/vdesk/data2/deklerk/GAAP_data/catalog_files'

def plot_cutout(tileindex, locations, cutout_size, axes):
    file = glob(f'{data_folder}/VIS_TILE{tileindex}.fits.gz')[0]
    with fits.open(file, memmap=True) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
    centers = wcs.wcs_world2pix(
        locations[:, 0], locations[:, 1], 0, ra_dec_order=True)
    print(centers)

    image_file = glob(f'{data_folder}/tile{tileindex}_VISgri_40_99.png')[0] #VISgri/HYJH

    half = cutout_size // 2
    axes = axes.flatten()
    with Image.open(image_file) as img:
        for i in range(len(centers[0])):

            x, y = centers[0][i], centers[1][i]

            left   = int(x - half)
            right  = int(x + half)
            top    = int(19200 - y + half)
            bottom = int(19200 - y - half)

            cutout = img.crop((left, bottom, right, top))

            axes[i].imshow(cutout)
            axes[i].axis('off')

if __name__=="__main__":
    locations = np.array([
    (65.181499, -48.311950),
    (64.783448, -48.417136),
    (65.050304, -48.382914),
    (64.821947, -48.641980),
    (65.137400, -48.628975),
    (64.947667, -48.269722),
    (64.840756, -48.426907),
    (64.941710, -48.580059),
    (64.922530, -48.263065),
    (64.800598, -48.302688),
    (64.718120, -48.728753),
    (65.077173, -48.397897),
    (65.121417, -48.703007),
    (64.939640, -48.584542),
    (64.684870, -48.329492),
    (65.223488, -48.597427),
    (64.949780, -48.447843),
    (64.682335, -48.315756),
    (65.075448, -48.384112),
    (64.948398, -48.286922),
    (64.806086, -48.671300),
    (64.934463, -48.286358),
    (64.860575, -48.450265),
    (64.887086, -48.584199),
    (64.853026, -48.622467),
    (64.932432, -48.468943),
    (65.090061, -48.710553),
    (65.238966, -48.676300),
    (65.286507, -48.334526),
    (64.895839, -48.531202),
    (64.815216, -48.700532),
    (64.684603, -48.284113),
    (65.127073, -48.313019),
    (64.674146, -48.335356),
    (64.927381, -48.558108),
    (65.067489, -48.375045),
    (64.908458, -48.514025),
    (64.832954, -48.685466),
    (65.157536, -48.322235),
    (65.176530, -48.239848),
    (65.025988, -48.540968),
    (65.099699, -48.394581),
    (64.941074, -48.587165),
    (64.875429, -48.236690),
    (65.042329, -48.246195),
    (65.317665, -48.616674),
    (65.341349, -48.619125),
    (65.259422, -48.544980),
    (65.178267, -48.647657),
    (64.717712, -48.532158),
    (65.239738, -48.672717),
    (65.275579, -48.369518),
    (64.865484, -48.729865),
    (65.349626, -48.721734),
    (64.634596, -48.364222),
    (65.008859, -48.329614),
    (64.967439, -48.548208),
    (64.824395, -48.351965),
    (64.767220, -48.272513),
    (64.797760, -48.645978),
    (64.951664, -48.406571),
    (65.141242, -48.402111),
    (64.717538, -48.711648),
    (65.289796, -48.291590),
])
    fig, axes = plt.subplots(8, 8, figsize=(12,12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plot_cutout(102021019, locations, 1024, axes)
    plt.tight_layout()
    plt.savefig('/home/deklerk/GAAP/results/figures/observation/not_extended_stars.pdf', bbox_inches='tight')
    plt.show()