from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import wilcoxon


DEFAULT_RUNS = [
    "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2026_20260424_172420",
    "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2027_20260424_192555",
    "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2028_20260425_125859",
]
REFERENCE_METHOD = "qca_enhanced"
COMPARISON_METHODS = [
    "p0_baseline_robust",
    "generic_film",
    "qca_shuffle_adapter_only",
    "qca_shuffle_norm_only",
    "qca_shuffled_reliability",
    "qca_random_reliability",
]


def _read_seed(run_dir: Path) -> int:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return int(cfg["seed"])


def load_metrics(project_root: Path, run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_rel in run_dirs:
        run_dir = project_root / run_rel
        path = run_dir / "subject_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["seed"] = _read_seed(run_dir)
        frame["run_dir"] = run_dir.name
        frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_methods(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = metrics.groupby(["seed", "method"], as_index=False)["balanced_accuracy"].mean()
    summary = (
        per_seed.groupby("method")["balanced_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return per_seed, summary


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["seed", "test_subject"],
        columns="method",
        values="balanced_accuracy",
    )
    rows = []
    for baseline in COMPARISON_METHODS:
        if REFERENCE_METHOD not in pivot.columns or baseline not in pivot.columns:
            continue
        diff = pivot[REFERENCE_METHOD] - pivot[baseline]
        try:
            p_value = float(wilcoxon(diff).pvalue)
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "comparison": f"{REFERENCE_METHOD}_minus_{baseline}",
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "n": int(diff.shape[0]),
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def write_report(path: Path, method_summary: pd.DataFrame, comparisons: pd.DataFrame, run_dirs: list[Path]) -> None:
    lines = [
        "# QCA Failure Controls Multi-Seed Summary",
        "",
        "This report repeats the QCA failure-control analysis over three BNCI full LOSO seeds. "
        "It is intended to test whether QCA v1's reliability semantics are stronger than generic/shuffled controls.",
        "",
        "## Runs",
        "",
    ]
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.extend(["", "## Method Means Across Seeds", "", "| Method | Mean | Std | Min | Max |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in method_summary.itertuples(index=False):
        lines.append(f"| `{row.method}` | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |")
    lines.extend(
        [
            "",
            "## Paired Subject-Seed Comparisons",
            "",
            "| Comparison | Mean diff | Median diff | Wins | Losses | Ties | N | Wilcoxon p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.comparison}` | {row.mean_diff:.4f} | {row.median_diff:.4f} | "
            f"{row.wins} | {row.losses} | {row.ties} | {row.n} | {row.wilcoxon_p:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If generic or shuffled controls match/exceed `qca_enhanced`, QCA v1 should remain an actionability probe.",
            "- The robust P0 baseline remains the method that the paper must treat as the hard source-only reference.",
            "- This summary should be used to support claim contraction, not to claim QCA superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("work/qca_failure_multiseed"))
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = parser.parse_args()
    run_dirs = [Path(item) for item in args.runs.split(",") if item]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.project_root, run_dirs)
    per_seed, method_summary = summarize_methods(metrics)
    comparisons = paired_comparisons(metrics)

    metrics.to_csv(args.output_dir / "qca_failure_multiseed_metrics_long.csv", index=False)
    per_seed.to_csv(args.output_dir / "qca_failure_multiseed_per_seed.csv", index=False)
    method_summary.to_csv(args.output_dir / "qca_failure_multiseed_method_summary.csv", index=False)
    comparisons.to_csv(args.output_dir / "qca_failure_multiseed_paired_comparisons.csv", index=False)
    report = args.output_dir / "qca_failure_multiseed_summary.md"
    write_report(report, method_summary, comparisons, run_dirs)
    print(report)


if __name__ == "__main__":
    main()
