# %%
import urllib.request
from pathlib import Path

to_download = [
    "https://zenodo.org/record/7716839/files/ixi.S4_R4.csv?download=1",
    "https://zenodo.org/record/7716839/files/ixi.S4_R8.csv?download=1",
    "https://zenodo.org/record/7716839/files/ixi.S8_R4.csv?download=1",
    "https://zenodo.org/record/7716839/files/ixi.S8_R8.csv?download=1",
]

out_dir = Path(__file__).parent.parent / 'data'

for src_url in to_download:
    dst_fname = out_dir / src_url.split('/')[-1].split('?')[0]
    print(f"Downloading {src_url} to {dst_fname}")
    urllib.request.urlretrieve(src_url, dst_fname)
