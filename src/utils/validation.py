from __future__ import annotations

from src.utils.bootstrap import Context
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass
class EnvVar:
    name: str
    required:bool=True
    
@dataclass 
class EnvVarSts:
    name:str
    status:bool
    

def validate_mlflow_connection(ctx: Optional[Context]=None,uri: Optional[str]=None) -> bool:
    from urllib.request import Request, urlopen
    
    if ctx is None and uri is None:
        raise ValueError("Either 'ctx' or 'uri' must be provided.")

    ctx_uri = ctx.mlflow_tracking_uri if ctx is not None else None
    mlflow_tracking_uri = ctx_uri or uri
    health_url = f"{mlflow_tracking_uri}/health" #mlflow_tracking_uri set by env var in container or out of.
    try:
        req = Request(health_url)
        with urlopen(req) as response:
            if response.status == 200:
                return True
            
    except Exception:
        return False

class UnsafeCacheDirError(Exception):
    """Raised when a local_cache_dir resolves to a path outside the project root."""
    pass

def validate_path_exists(root: str|Path,
                          path: list[str | Path],
                          create: bool = True,
                          kind:Literal["dir", "file"] = "dir",
                          validate_root: bool = True) -> bool:

    from pathlib import Path

   

    root = Path(root).resolve()
    ok = True


    for p in path:

        p = Path(p)

        candidate = (root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()

        if validate_root:#only validate if the candidate path is within the root if validate_root is True
            try:
                candidate.relative_to(root)
                
            except ValueError:
                raise UnsafeCacheDirError(
                    f"dir '{path}' resolves to '{candidate}', "
                    f"which is outside the project root '{root}'. "
                    f"Use a path relative to the project (e.g. 'models/cached')."
                )

        if candidate.exists():
            continue  

        if not create:
            ok = False
            continue

        if kind == "dir":
            candidate.mkdir(parents=True, exist_ok=True)
        elif kind == "file":
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch(exist_ok=True)
        else:
            raise ValueError(f"kind must be 'dir' or 'file', got {kind!r}")
        
    return ok



def validate_env_vars(required_vars: list[EnvVar], env_path: str = ".env")-> list[EnvVarSts]:
    """
    Check if the required environment variables are set.
    """
    from loguru import logger
    load_dotenv(dotenv_path=env_path)
    
   
    validated = []

    for c in required_vars:
        if not isinstance(c, EnvVar):
            logger.exception(f"Invalid type for required_vars: {type(c)}. Expected EnvVar instance.")
            raise TypeError("required_vars must be a list of EnvVar instances.")
    
        var = os.getenv(c.name)
    
        if var is None:
            v = EnvVarSts(name=c.name, status=False)

            
            if c.required:
                logger.error(f"Required environment variable '{c.name}' is not set. \
                             Please set it in your environment or in the {env_path} file.")
            elif c.required == False:
                logger.warning(f"Optional environment variable '{c.name}' is not set\
                               . Consider setting it in your environment or in the {env_path} file.")
            else:
                logger.info(f"Optional environment variable '{c.name}' is not set. \
                            You may want to set it in your environment or in the {env_path} file.")
                
        else:
            v = EnvVarSts(name=c.name, status=True)
                
        validated.append(v)

                
        

    return validated