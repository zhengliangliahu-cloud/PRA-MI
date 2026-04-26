# PRA-MI

Hydra + PyTorch Lightning scaffold for the PRA-MI residual-audit pilot.
The current paper-level claim is protocolized residual audit, not QCA-MI
model superiority.

## Quick Start

```bash
cd PRA-MI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If `python -m venv` fails because `ensurepip` is unavailable, install the
system `python3.10-venv` package first or use an existing conda environment.
The code also has a PyYAML fallback for config loading, so unit tests and the
synthetic smoke path can run before Hydra/OmegaConf is installed.

Run the synthetic smoke path without downloading EEG data:

```bash
python -m autoeegencoder.train experiment=synthetic_pra_mi_smoke dataset=synthetic trainer=numpy_smoke subjects=1,2
python -m autoeegencoder.audit run_dir=results/<run_dir>
```

Run the BNCI2014_001 pilot through MOABB:

```bash
python -m autoeegencoder.train experiment=bnci_pra_mi_smoke subjects=1,2
python -m autoeegencoder.audit run_dir=results/<run_dir>
```

Frozen 9-subject pilot with the current claim-safe method bundle:

```bash
python -m autoeegencoder.train experiment=bnci_pra_mi_frozen_full model=qca_eegnet_lite trainer=gpu_pilot
```

QCA ablation smoke:

```bash
python -m autoeegencoder.train experiment=bnci_pra_mi_ablation_smoke model=qca_eegnet_lite trainer=gpu_smoke
```

QCA failure-analysis full run:

```bash
python -m autoeegencoder.train experiment=bnci_pra_mi_qca_failure_full model=qca_eegnet_lite trainer=gpu_pilot
```

Cho2017 full external validation with the frozen core bundle:

```bash
python -m autoeegencoder.train dataset=cho2017 experiment=cho2017_pra_mi_full trainer=gpu_external_full
```

Generate the PRA-MI audit card and publication-style figures:

```bash
python scripts/generate_pra_mi_audit_report.py --project-root . --output-dir work/pra_mi_audit --fig-dir figs/pra_mi_audit
```

Generate the paper-facing main tables:

```bash
python scripts/summarize_paper_main_tables.py --project-root . --output-dir work/paper_tables
```

## Outputs

Each training run writes a timestamped directory under `results/` with:

- `config_resolved.yaml`
- `subject_metrics.csv`
- `diagnostics.csv`
- `harmonization_ledger.csv`
- `residual_audit.csv`
- `audit_summary.md`

The current frozen BNCI pilot identifies `p0_baseline_robust` as the strongest
strict P0 method among the implemented baselines. QCA-MI is kept as a
diagnostic-actionability probe unless future multi-seed and external-dataset
experiments justify a stronger method claim.
