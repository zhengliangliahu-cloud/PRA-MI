from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from autoeegencoder.analysis.residuals import compute_residual_rows, summarize_audit
from autoeegencoder.data import load_dataset, make_loso_split
from autoeegencoder.models.classical_baseline import run_classical_method
from autoeegencoder.models.lightning_module import run_lightning_method
from autoeegencoder.models.numpy_baseline import run_numpy_method
from autoeegencoder.protocols import prepare_method_arrays
from autoeegencoder.utils.config import load_config, project_root, save_resolved_config
from autoeegencoder.utils.io import ensure_dir, write_csv_rows, write_text
from autoeegencoder.utils.random import derive_run_seed, seed_everything


def _run_dir(cfg: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg['experiment']['name']}_{cfg['dataset']['name']}_seed-{cfg['seed']}_{timestamp}"
    return ensure_dir(project_root() / cfg.get("output_root", "results") / name)


def _aggregate_diagnostics(
    method_name: str,
    protocol_tier: str,
    subject: int,
    dataset: str,
    method_arrays,
    *,
    bundle_version: str,
    qri_version: str,
) -> list[dict]:
    rows = []
    diag = method_arrays.test_diagnostics
    for component, values in diag.components.items():
        rows.append(
            {
                "dataset": dataset,
                "method": method_name,
                "protocol_tier": protocol_tier,
                "test_subject": subject,
                "component": component,
                "bundle_version": bundle_version,
                "qri_version": qri_version,
                "availability": diag.availability[component],
                "mean_value": f"{float(np.mean(values)):.6f}",
                "can_enter_method": "yes" if diag.availability[component] in {"D0a", "D0b"} else "no",
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "method": method_name,
            "protocol_tier": protocol_tier,
            "test_subject": subject,
            "component": "qri_trial_risk",
            "bundle_version": bundle_version,
            "qri_version": qri_version,
            "availability": diag.trial_availability,
            "mean_value": f"{float(np.mean(diag.trial_risk)):.6f}",
            "can_enter_method": "yes" if diag.trial_availability in {"D0a", "D0b"} else "no",
        }
    )
    return rows


def _run_backend(method_arrays, cfg: dict, method: dict, run_seed: int) -> dict[str, float]:
    engine = method.get("engine", cfg["trainer"].get("engine", "lightning"))
    if engine == "numpy":
        return run_numpy_method(method_arrays)
    if engine == "classical":
        return run_classical_method(method_arrays, method, run_seed)
    if engine == "lightning":
        return run_lightning_method(method_arrays, cfg, method, run_seed)
    raise NotImplementedError(f"Unknown trainer engine: {engine}")


def run(cfg: dict) -> Path:
    seed_everything(int(cfg["seed"]))
    arrays = load_dataset(cfg)
    run_dir = _run_dir(cfg)
    save_resolved_config(cfg, run_dir / "config_resolved.yaml")
    bundle_version = str(cfg["experiment"].get("bundle_version", "unversioned"))
    qri_version = str(cfg["experiment"].get("qri_version", "qri_unspecified"))

    eval_subjects = [int(s) for s in cfg["eval_subjects"]]
    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    ledger_rows: list[dict] = []

    for subject in eval_subjects:
        split = make_loso_split(
            arrays.subjects,
            subject,
            val_subjects=int(cfg["trainer"].get("val_subjects", 1)),
            seed=int(cfg["seed"]),
        )
        for method in cfg["experiment"]["methods"]:
            run_seed = derive_run_seed(int(cfg["seed"]), subject, str(method["name"]))
            seed_everything(run_seed)
            method_arrays = prepare_method_arrays(arrays, split, cfg, method, method_seed=run_seed)
            metrics = _run_backend(method_arrays, cfg, method, run_seed)
            metric_rows.append(
                {
                    "dataset": cfg["dataset"]["name"],
                    "method": method["name"],
                    "protocol_tier": method["protocol_tier"],
                    "bundle_version": bundle_version,
                    "qri_version": method.get("qri_version", qri_version),
                    "adapter_version": method.get("adapter_version", "none"),
                    "preprocessing": method["preprocessing"],
                    "adapter": method.get("adapter", "none"),
                    "engine": method.get("engine", cfg["trainer"].get("engine", "lightning")),
                    "baseline": method.get("baseline", "none"),
                    "run_seed": run_seed,
                    "test_subject": subject,
                    "loaded_subjects": len(np.unique(arrays.subjects)),
                    "n_train_subjects": len(split.train_subjects),
                    "n_val_subjects": len(split.val_subjects),
                    "val_strategy": split.val_strategy,
                    "n_train_trials": len(split.train_idx),
                    "n_val_trials": len(split.val_idx),
                    "n_test_trials": len(split.test_idx),
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                    "classical_val_balanced_accuracy": metrics.get("classical_val_balanced_accuracy", ""),
                    "classical_selected_params": metrics.get("classical_selected_params", ""),
                }
            )
            diagnostic_rows.extend(
                _aggregate_diagnostics(
                    method["name"],
                    method["protocol_tier"],
                    subject,
                    cfg["dataset"]["name"],
                    method_arrays,
                    bundle_version=bundle_version,
                    qri_version=str(method.get("qri_version", qri_version)),
                )
            )
            ledger_rows.extend(method_arrays.guard.rows(cfg["dataset"]["name"], method["name"], subject))

    ledger_rows.append(
        {
            "dataset": cfg["dataset"]["name"],
            "method": "all",
            "test_subject": "all",
            "protocol_tier": "all",
            "action": "frozen_method_bundle",
            "availability": "metadata",
            "detail": bundle_version,
        }
    )

    for note in cfg["dataset"].get("harmonization_notes", []):
        ledger_rows.append(
            {
                "dataset": cfg["dataset"]["name"],
                "method": "all",
                "test_subject": "all",
                "protocol_tier": "all",
                "action": "harmonization_note",
                "availability": "metadata",
                "detail": note,
            }
        )

    residual_rows = compute_residual_rows(metric_rows)
    write_csv_rows(run_dir / "subject_metrics.csv", metric_rows)
    write_csv_rows(run_dir / "diagnostics.csv", diagnostic_rows)
    write_csv_rows(run_dir / "harmonization_ledger.csv", ledger_rows)
    write_csv_rows(run_dir / "residual_audit.csv", residual_rows)
    write_text(run_dir / "audit_summary.md", summarize_audit(residual_rows, diagnostic_rows))
    print(f"Run complete: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(sys.argv[1:] if argv is None else argv)
    run(cfg)


if __name__ == "__main__":
    main()
