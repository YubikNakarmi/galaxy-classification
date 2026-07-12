"""Compatibility shim for BentoML module imports.

This lets `bentoml serve GalaxyClassification` resolve from the
`bentoml_api/` directory even though the actual service lives in
`bentoml_api.service`.
"""

from pathlib import Path
import sys


_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from bentoml_api.service import GalaxyClassification
