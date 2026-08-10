import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import subprocess

root = Path(__file__).parent.parent.parent

# https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas


def load_dataset(path:str|Path):
    file_path = str(path)

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jaimetrickz/galaxy-zoo-2-images",
    file_path,

    )

    return df

def make_dirs():

    subprocess.run(["bash","-c", f"source {root}/setup.sh", "&& dir_setup"], check=True)
