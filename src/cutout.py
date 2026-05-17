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

    image_file = glob(f'{data_folder}/tile{tileindex}_VISgri_40_99.png')[0]

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
    (52.748703, -28.251955),
    (52.969713, -28.249965),
    (52.732988, -28.246438),
    (52.761677, -28.245221),
    (53.067179, -28.242782),
    (52.718499, -28.239351),
    (52.883698, -28.239362),
    (52.880574, -28.238738),
    (52.848844, -28.237669),
    (52.920722, -28.237314),
    (52.839130, -28.237142),
    (53.133445, -28.235969),
    (53.073764, -28.233443),
    (53.067562, -28.232437),
    (52.892886, -28.232313),
    (52.925996, -28.229285),
    (52.926200, -28.229211),
    (52.754794, -28.228693),
    (53.232978, -28.228330),
    (52.750111, -28.228181),
    (53.066744, -28.227432),
    (53.067503, -28.227390),
    (53.065596, -28.227331),
    (53.055899, -28.226897),
    (53.067650, -28.226750),
    (52.913224, -28.226357),
    (52.924786, -28.224295),
    (53.078709, -28.223271),
    (53.078123, -28.222555),
    (52.915759, -28.222034),
    (52.915978, -28.221477),
    (53.075388, -28.221278),
    (53.119431, -28.221122),
    (53.055038, -28.221162),
    (53.126589, -28.220976),
    (53.068694, -28.220213),
    (53.074696, -28.218009),
    (52.761094, -28.216899),
    (52.943744, -28.214597),
    (52.892143, -28.214111),
    (52.979544, -28.212874),
    (52.979632, -28.213003),
    (52.885452, -28.211046),
    (53.158777, -28.210436),
    (52.891437, -28.210272),
    (52.939669, -28.209546),
    (52.939423, -28.209043),
    (52.836515, -28.208503),
    (52.900768, -28.208464),
    (52.965174, -28.207880),
    (52.898413, -28.207815),
    (52.972908, -28.204380),
    (52.972364, -28.203776),
    (52.916921, -28.203315),
    (52.819075, -28.202175),
    (53.123408, -28.201035),
    (53.126550, -28.200292),
    (52.960443, -28.198715),
    (52.959455, -28.198632),
    (53.229209, -28.197097),
    (52.930077, -28.196197),
    (53.164554, -28.193466),
    (52.927568, -28.193851),
    (52.952146, -28.192679),
])
    fig, axes = plt.subplots(8, 8, figsize=(12,12), gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plot_cutout(102044185, locations, 128, axes)
    plt.tight_layout()
    plt.savefig('/home/deklerk/GAAP/results/figures/observation/what.pdf', bbox_inches='tight')
    plt.show()