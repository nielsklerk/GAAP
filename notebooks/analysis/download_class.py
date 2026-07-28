import subprocess
import csv
import glob
from astropy.io import fits
from astropy.table import Table
import pandas as pd

def download_file(catalog_file):
    cmd = [
        "curl",
        "-k",
        "-L",
        "-o", f"/net/vdesk/data2/deklerk/GAAP_data/class_files/{catalog_file}",
        f"https://eas.esac.esa.int/sas-dd/data?file_name={catalog_file}&release=sedm&RETRIEVAL_TYPE=FILE"
    ]
    subprocess.run(cmd)

dfs = []
with open("/home/deklerk/GAAP/notebooks/analysis/class_files.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        file = glob.glob(f"/net/vdesk/data2/deklerk/GAAP_data/class_files/{row[0]}")[0]
        with fits.open(file, memmap=True) as hdul:
            cat = Table(hdul[1].data)
        cat = cat.to_pandas()
        cat = cat[['OBJECT_ID', 'PHZ_STAR_PROB', 'PHZ_GAL_PROB',
       'PHZ_QSO_PROB', 'PHZ_GLOB_CL_PROB', 'PHZ_CLASSIFICATION']]
        dfs.append(cat)
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_pickle('/net/vdesk/data2/deklerk/GAAP_data/class_files/all_class_data.pkl')