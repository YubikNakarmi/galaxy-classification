''' Parser and overwritting default logging with file stream configuration
- setup_app_logging: for application logs
- setup_ml_logging: for machine learning logs


'''

from loguru import logger
from pathlib import Path
import sys


root = Path(__file__).resolve().parent.parent.parent

def setup_app_logging(config):
    logger.remove()

    app = config["logging"]["app"]

    logger.add(
        sys.stdout,
        level=app["level"]
    )

    logger.add(
        root / app["file"],
        level=app["level"],
        rotation=app["rotation"],
        retention=app["retention"],
        compression=app["compression"]
    )


def setup_ml_logging(config):
    logger.remove()

    ml = config["logging"]["ml"]

    logger.add(
        sys.stdout,
        level=ml["level"]
    )

    logger.add(
        root / ml["file"],
        level=ml["level"],
        rotation=ml["rotation"],
        retention=ml["retention"],
        compression=ml["compression"]
    )