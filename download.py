import fire
import requests, zipfile
from pathlib import Path

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import urllib.request


def download(OUTPUT_DIR):
    datapath = Path(OUTPUT_DIR)
    url = "https://github.com/meghanabhange/translation/releases/download/0.1/T5.zip"
    target_path = datapath/'T5.zip'
    with urlopen(url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(OUTPUT_DIR)

    url_trans = "https://github.com/meghanabhange/translation/releases/download/0.1/ende-model.pt"

    with urllib.request.urlopen(url_trans) as f:
        html = f.read().decode('utf-8')

fire.Fire(download)