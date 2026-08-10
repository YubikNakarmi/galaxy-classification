
from loguru import logger
from src.cli import setup as st
from . import setup

@setup.command()
def main():
    st.KaggleDatasetAdapter()


