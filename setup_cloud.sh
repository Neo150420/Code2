#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Code2
export UV_CACHE_DIR=/workspace/research/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/research/.python
export PYTHONDONTWRITEBYTECODE=1
uv python install --no-bin 3.10.22 3.8.20
if [ ! -d /workspace/research/.venv ]; then
  uv venv --python /workspace/research/.python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 /workspace/research/.venv
fi
if [ ! -d /workspace/research/.sumo-venv ]; then
  uv venv --python /workspace/research/.python/cpython-3.8-linux-x86_64-gnu/bin/python3.8 /workspace/research/.sumo-venv
fi
uv pip install --python /workspace/research/.venv/bin/python -r requirements-smoke.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/research/.sumo-venv/bin/python -r requirements-sumo.lock.txt
uv pip check --python /workspace/research/.venv/bin/python
uv pip check --python /workspace/research/.sumo-venv/bin/python
export PATH=/workspace/research/.venv/bin:/workspace/research/.sumo-venv/bin:$PATH
python -m unittest -v test_correctness test_validation test_reproduction
python - <<'PY'
import hashlib,json
from pathlib import Path
data=json.loads(Path('provenance.json').read_text())
for name,digest in data['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
        raise SystemExit(f'Original simulator asset changed: {name}')
print('Original simulator input hashes verified')
PY
