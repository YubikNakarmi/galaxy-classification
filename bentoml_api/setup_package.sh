#!/bin/bash
set -euxo pipefail
uv pip install --python /app/.venv -e "$BENTO_PATH"