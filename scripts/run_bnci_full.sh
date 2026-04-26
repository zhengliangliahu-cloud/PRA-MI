#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m autoeegencoder.train experiment=bnci_pra_mi_full subjects=1,2,3,4,5,6,7,8,9

