#!/usr/bin/env bash
set -euo pipefail

_REPO_ROOT="${_REPO_ROOT:-${PROJECT_ROOT:-$PWD}}"
cd "${_REPO_ROOT}"

uv sync

TORCH_VERSION="$(uv run python -c 'import torch; print(torch.__version__)')"
echo "torch installed (version ${TORCH_VERSION})"
