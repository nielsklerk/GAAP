import subprocess
import csv

def download_file(catalog_file):
    cmd = [
        "curl",
        "-k",
        "-L",
        "-o", f"D:/CATALOG/{catalog_file}",
        f"https://eas.esac.esa.int/sas-dd/data?file_name={catalog_file}&release=sedm&RETRIEVAL_TYPE=FILE"
    ]
    subprocess.run(cmd)

with open("class_files.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        download_file(row[0])