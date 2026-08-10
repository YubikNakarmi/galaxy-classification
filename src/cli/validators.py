from dotenv import load_dotenv
from loguru import logger
import os

def check_env_vars(required_vars,env_path: str = ".env"):
    """
    Check if the required environment variables are set.
    """
    load_dotenv(dotenv_path=env_path)

    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")