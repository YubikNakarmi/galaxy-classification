from loguru import logger
import sys

''' Parser and overwritting default logging with file stream configuration'''

def setup_app_logging(config):
    logger.remove()

    app = config["logging"]["app"]

    logger.add(
        sys.stdout,
        level=app["level"]
    )

    logger.add(
        app["file"],
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
        ml["file"],
        level=ml["level"],
        rotation=ml["rotation"],
        retention=ml["retention"],
        compression=ml["compression"]
    )