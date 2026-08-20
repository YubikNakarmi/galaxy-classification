from dotenv import load_dotenv
from loguru import logger
import os
from typing import Literal
from dataclasses import dataclass

@dataclass
class EnvVar:
    name: str
    priority: Literal["required","optional"]

def check_env_vars(required_vars: list[EnvVar], env_path: str = ".env")-> list[str]:
    """
    Check if the required environment variables are set.
    """
    load_dotenv(dotenv_path=env_path)

    validated = []

    for c in required_vars:

        var = os.getenv(c.name)
        validated.append(var)

        if var is None:
            if c.priority == "required":
                logger.error(f"Required environment variable '{c.name}' is not set. \
                             Please set it in your environment or in the {env_path} file.")
                raise EnvironmentError(f"Required environment variable '{c.name}' is not set.")
            elif c.priority == "optional":
                logger.warning(f"Optional environment variable '{c.name}' is not set\
                               . Consider setting it in your environment or in the {env_path} file.")
            else:
                logger.info(f"Optional environment variable '{c.name}' is not set. \
                            You may want to set it in your environment or in the {env_path} file.")

    return validated




