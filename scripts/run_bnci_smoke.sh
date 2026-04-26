#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m autoeegencoder.train experiment=bnci_pra_mi_smoke subjects=1,2

