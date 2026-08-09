import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path

# https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas


def load_dataset(path:str|Path):
    file_path = str(path)

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jaimetrickz/galaxy-zoo-2-images",
    file_path,

    )

def make_dirs():
    pass
