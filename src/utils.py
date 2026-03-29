import pandas as pd
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def gaussian_2d(xy, amplitude, sigma, x0, y0):
    x, y = xy
    r_2 = (x - x0)**2 + (y - y0)**2
    g = amplitude * np.exp(
        -r_2/(2*sigma**2)
    )
    return g.ravel()

def create_psf_from_psf_grid(psf_grid, size, num_to_combine):
    Ng = psf_grid.shape[0] // size
    psfs = psf_grid.reshape(Ng, size, Ng, size)
    psfs = psfs.transpose(0, 2, 1, 3)
    central = psfs[Ng//2 - num_to_combine//2:Ng//2 + num_to_combine//2, Ng//2 - num_to_combine//2:Ng//2 + num_to_combine//2]
    mean_psf = np.mean(central, axis=(0, 1))
    mean_psf /= mean_psf.sum()
    return mean_psf

def download_archive_files(tile_index, filename_file='EUCLID_ARCHIVE_files.pkl', data_folder='data', max_workers=4):
    def download_file(catalog_file):
        cmd = [
            "curl",
            "-k",
            "-L",
            "-o", f"{data_folder}/{catalog_file}",
            f"https://eas.esac.esa.int/sas-dd/data?file_name={catalog_file}&release=sedm&RETRIEVAL_TYPE=FILE"
        ]
        subprocess.run(cmd)

    # Load all file names
    filenames = pd.read_pickle(filename_file)

    # Select files of the tile_index
    tile_files = filenames.loc[tile_index]

    # Empty data folder
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Collect all files for downloading
    all_files = (
        tile_files['FINAL-CAT'] +
        tile_files['BGSUB'] +
        tile_files['CATALOG-PSF'] +
        tile_files['RMS']
    )

    # Run downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_file, all_files)
        
    return tile_files['FILTER']

def main():
    download_archive_files("102159780")

if __name__ == "__main__":
    main()