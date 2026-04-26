from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


COMPARISONS = [
    ("qca_enhanced", "p0_baseline_robust"),
    ("qca_enhanced", "generic_film"),
    ("qca_enhanced", "qca_adapter_only"),
    ("qca_enhanced", "qca_norm_only"),
    ("qca_enhanced", "qca_shuffle_adapter_only"),
    ("qca_enhanced", "qca_shuffle_norm_only"),
    ("qca_enhanced", "qca_d0b_local_only"),
    ("qca_enhanced", "qca_d0a_source_ref_only"),
    ("qca_enhanced", "qca_random_reliability"),
    ("qca_enhanced", "qca_shuffled_reliability"),
    ("qca_adapter_only", "generic_film"),
    ("qca_norm_only", "p0_baseline_robust"),
    ("qca_shuffle_adapter_only", "qca_shuffle_norm_only"),
    ("qca_d0b_local_only", "qca_random_reliability"),
    ("qca_d0a_source_ref_only", "qca_random_reliability"),
]


def summarize(run_dir: Path, output_dir: Path) -> Path:
    metrics_path = run_dir / "subject_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_path)
    means = (
        df.groupby("method")["balanced_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    pivot = df.pivot(index="test_subject", columns="method", values="balanced_accuracy")
    comparisons = []
    for left, right in COMPARISONS:
        if left not in pivot or right not in pivot:
            continue
        diff = pivot[left] - pivot[right]
        comparisons.append(
            {
                "comparison": f"{left}_minus_{right}",
                "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()),
                "n": int(diff.shape[0]),
                "wilcoxon_p": float(wilcoxon(diff).pvalue),
            }
        )

    stem = run_dir.name
    means_path = output_dir / f"{stem}_means.csv"
    comparisons_path = output_dir / f"{stem}_comparisons.csv"
    md_path = output_dir / f"{stem}_summary.md"
    means.to_csv(means_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparisons_path, index=False)

    lines = [
        "# BNCI Full Ablation Summary",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Method Means",
        "",
        "| Method | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in means.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Comparison | Mean diff | Wins | N | Wilcoxon p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| `{row['comparison']}` | {row['mean_diff']:.4f} | {row['wins']} | {row['n']} | {row['wilcoxon_p']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Claim Implication",
            "",
            "This full ablation tests whether P0-safe structured QRI remains useful beyond a smoke run. "
            "A strong QCA claim requires `qca_enhanced` to beat robust normalization and reliability controls "
            "with subject-level consistency, not just a small mean advantage.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("work"))
    args = parser.parse_args()
    print(summarize(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
