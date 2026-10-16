#!/usr/bin/env bash
# Builds mmdet3d 1.0.0rc6 into the pixi environment; run through `pixi run install-mmdet3d`.
# Upstream's __init__ refuses mmcv > 1.7.0; the stack here uses mmcv-full 1.7.2, so the one
# assertion bound is raised to match. Nothing else in mmdet3d is patched.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/.build/mmdetection3d"
if [ ! -d "$src/.git" ]; then
    mkdir -p "$here/.build"
    git clone https://github.com/open-mmlab/mmdetection3d.git "$src"
fi
git -C "$src" checkout --force 47285b3f1e9dba358e98fcd12e523cfd0769c876   # v1.0.0rc6
sed -i "s/^mmcv_maximum_version = '1.7.0'/mmcv_maximum_version = '1.7.2'/" "$src/mmdet3d/__init__.py"
grep -q "mmcv_maximum_version = '1.7.2'" "$src/mmdet3d/__init__.py"
cd "$src"
pip install -v --no-build-isolation --no-deps --no-cache-dir .
# mmdet3d.core.evaluation imports lyft_dataset_sdk at import time (nothing here evaluates on Lyft).
# Its declared dependencies (jupyter, black, plotly, an old pandas...) are skipped on purpose.
pip install --no-deps lyft-dataset-sdk==0.0.8
