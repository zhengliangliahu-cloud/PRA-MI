#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m autoeegencoder.train experiment=synthetic_pra_mi_smoke dataset=synthetic trainer=numpy_smoke subjects=1,2

