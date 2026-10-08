"""Download the MP-20 training split of CDVAE (Xie et al., 2022): 27,136 crystal structures as CIF strings.

pixi run -e modalities python -m modalities.interatomic_potentials.download
"""

from modalities.common import download
from modalities.interatomic_potentials import STRUCTURES_CSV

URL = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/train.csv"
SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"

if __name__ == "__main__":
    print(f"wrote {download(URL, STRUCTURES_CSV, SHA256)}")
