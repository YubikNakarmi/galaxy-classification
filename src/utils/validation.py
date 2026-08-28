from __future__ import annotations

from src.utils.bootstrap import Context
from pathlib import Path
from typing import Literal

def validate_mlflow_connection(ctx: Context) -> bool:
    from urllib.request import Request, urlopen

    mlflow_tracking_uri = ctx.mlflow_tracking_uri
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

