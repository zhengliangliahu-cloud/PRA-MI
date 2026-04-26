from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import wilcoxon


def _seed_for_run(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    seed = cfg.get("seed")
    return int(seed) if seed is not None else None


def _collect_runs(results_root: Path, prefix: str, seeds: set[int]) -> list[tuple[int, Path]]:
    runs: list[tuple[int, Path]] = []
    for run_dir in sorted(results_root.glob(f"{prefix}_*")):
        metrics_path = run_dir / "subject_metrics.csv"
        if not metrics_path.exists():
            continue
        seed = _seed_for_run(run_dir)
        if seed in seeds:
            runs.append((int(seed), run_dir))
    return sorted(runs, key=lambda item: item[0])


def summarize(results_root: Path, output_dir: Path, prefix: str, seeds: list[int]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(results_root, prefix, set(seeds))
    if not runs:
        raise FileNotFoundError(f"No runs found for prefix={prefix!r} seeds={seeds}")

    metric_frames = []
    run_rows = []
    for seed, run_dir in runs:
        df = pd.read_csv(run_dir / "subject_metrics.csv")
        df["seed"] = seed
        df["run_dir"] = str(run_dir)
        metric_frames.append(df)
        run_rows.append({"seed": seed, "run_dir": str(run_dir)})

    metrics = pd.concat(metric_frames, ignore_index=True)
    per_seed = (
        metrics.groupby(["seed", "method"], as_index=False)["balanced_accuracy"]
        .mean()
        .rename(columns={"balanced_accuracy": "mean_balanced_accuracy"})
    )
    aggregate = (
        per_seed.groupby("method")["mean_balanced_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    pivot_subject = metrics.pivot_table(
        index=["seed", "test_subject"],
        columns="method",
        values="balanced_accuracy",
    )
    comparisons = []
    if "qca_enhanced" in pivot_subject:
        for baseline in ["p0_baseline_robust", "p1_target_stat", "riemann_tangent_lr"]:
            if baseline not in pivot_subject:
                continue
            diff = pivot_subject["qca_enhanced"] - pivot_subject[baseline]
            comparisons.append(
                {
                    "comparison": f"qca_enhanced_minus_{baseline}",
                    "mean_diff": float(diff.mean()),
                    "wins": int((diff > 0).sum()),
                    "n": int(diff.shape[0]),
                    "wilcoxon_p": float(wilcoxon(diff).pvalue),
                }
            )

    runs_path = output_dir / "frozen_multiseed_runs.csv"
    per_seed_path = output_dir / "frozen_multiseed_per_seed.csv"
    aggregate_path = output_dir / "frozen_multiseed_summary.csv"
    comparisons_path = output_dir / "frozen_multiseed_comparisons.csv"
    md_path = output_dir / "frozen_multiseed_summary.md"

    pd.DataFrame(run_rows).to_csv(runs_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparisons_path, index=False)

    lines = [
        "# Frozen BNCI Multi-Seed Summary",
        "",
        f"Runs included: {len(runs)}",
        "",
        "## Run Directories",
        "",
    ]
    for seed, run_dir in runs:
        lines.append(f"- seed `{seed}`: `{run_dir}`")
    lines.extend(
        [
            "",
            "## Method Means Across Seeds",
            "",
            "| Method | Mean | Std | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |"
        )
    lines.extend(
        [
            "",
            "## QCA Paired Comparisons",
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
            "This multi-seed validation should be interpreted as frozen-pilot evidence. "
            "A QCA method-superiority claim requires QCA to beat the strongest strict P0 baseline "
            "consistently across seeds and subjects; otherwise QCA should remain an actionability probe.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("work"))
    parser.add_argument("--prefix", default="bnci_pra_mi_frozen_full")
    parser.add_argument("--seeds", default="2026,2027,2028")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    summary_path = summarize(args.results_root, args.output_dir, args.prefix, seeds)
    print(summary_path)


if __name__ == "__main__":
    main()
