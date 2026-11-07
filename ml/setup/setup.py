

from . import setup
import click


@setup.command
@click.option("--extra_dirs", default="", help="Comma-separated list of extra directories to create.")
def mkdirs():

    from loguru import logger
    from src.cli import setup as st

    logger.info("Setting up the project...")
    st.make_dirs()
    logger.info("Project setup completed successfully.")

