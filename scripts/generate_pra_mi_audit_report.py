from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr, wilcoxon


DEFAULT_FROZEN_RUNS = [
    "results/bnci_pra_mi_frozen_full_bnci2014_001_seed-2026_20260424_145549",
    "results/bnci_pra_mi_frozen_full_bnci2014_001_seed-2027_20260424_164406",
    "results/bnci_pra_mi_frozen_full_bnci2014_001_seed-2028_20260424_164853",
]
DEFAULT_FAILURE_RUN = "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2026_20260424_172420"
DEFAULT_ABLATION_RUN = "results/bnci_pra_mi_ablation_full_bnci2014_001_seed-2026_20260424_170037"

CORE_METHODS = ["p0_baseline_robust", "qca_enhanced", "p1_target_stat", "riemann_tangent_lr"]
METHOD_LABELS = {
    "p0_baseline_robust": "Robust P0",
    "qca_enhanced": "QCA v1",
    "p1_target_stat": "P1 target-stat",
    "riemann_tangent_lr": "Riemann TS-LR",
    "qca_shuffle_adapter_only": "Shuffle adapter",
    "qca_shuffled_reliability": "Shuffled QRI",
    "generic_film": "Generic FiLM",
    "qca_shuffle_norm_only": "Shuffle norm",
    "qca_adapter_only": "Adapter only",
    "qca_norm_only": "Norm only",
    "qca_random_reliability": "Random QRI",
}


def _read_seed(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    seed = cfg.get("seed")
    return int(seed) if seed is not None else None


def _load_metrics(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "subject_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["run_dir"] = str(run_dir)
        df["seed"] = _read_seed(run_dir)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_one(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
        }
    )


def _save_fig(fig: plt.Figure, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def plot_multiseed_bar(metrics: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    per_seed = (
        metrics[metrics["method"].isin(CORE_METHODS)]
        .groupby(["seed", "method"], as_index=False)["balanced_accuracy"]
        .mean()
    )
    summary = (
        per_seed.groupby("method")["balanced_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary["label"] = summary["method"].map(_method_label)

    palette = sns.color_palette("colorblind", n_colors=len(summary))
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.bar(summary["label"], summary["mean"], yerr=summary["std"], capsize=3, color=palette)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0.45, max(0.70, float(summary["mean"].max() + summary["std"].fillna(0).max() + 0.03)))
    ax.set_title("Frozen BNCI full: 3-seed method means")
    ax.tick_params(axis="x", rotation=20)
    sns.despine(fig=fig, ax=ax)
    _save_fig(fig, fig_dir / "frozen_multiseed_method_means")
    return summary


def plot_subject_heatmap(metrics: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    filtered = metrics[metrics["method"].isin(CORE_METHODS)].copy()
    subject_mean = (
        filtered.groupby(["test_subject", "method"], as_index=False)["balanced_accuracy"]
        .mean()
    )
    heat = subject_mean.pivot(index="method", columns="test_subject", values="balanced_accuracy")
    heat = heat.reindex(CORE_METHODS)
    heat.index = [_method_label(m) for m in heat.index]

    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="viridis",
        vmin=0.45,
        vmax=0.80,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Balanced accuracy"},
    )
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("")
    ax.set_title("Subject-level performance averaged over 3 seeds")
    _save_fig(fig, fig_dir / "frozen_multiseed_subject_heatmap")
    return subject_mean


def plot_qca_delta(metrics: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    pivot = metrics.pivot_table(
        index=["seed", "test_subject"],
        columns="method",
        values="balanced_accuracy",
    )
    rows = []
    for baseline in ["p0_baseline_robust", "p1_target_stat", "riemann_tangent_lr"]:
        if baseline not in pivot.columns:
            continue
        diff = pivot["qca_enhanced"] - pivot[baseline]
        for (seed, subject), value in diff.items():
            rows.append(
                {
                    "seed": seed,
                    "test_subject": subject,
                    "baseline": baseline,
                    "baseline_label": _method_label(baseline),
                    "delta": value,
                }
            )
    delta_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    sns.stripplot(
        data=delta_df,
        x="baseline_label",
        y="delta",
        hue="baseline_label",
        dodge=False,
        jitter=0.18,
        palette="colorblind",
        ax=ax,
        legend=False,
    )
    sns.pointplot(
        data=delta_df,
        x="baseline_label",
        y="delta",
        errorbar=("ci", 95),
        color="black",
        markers="_",
        linestyles="none",
        ax=ax,
    )
    ax.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("QCA v1 minus baseline BAcc")
    ax.set_title("Subject-seed paired QCA deltas")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(fig=fig, ax=ax)
    _save_fig(fig, fig_dir / "qca_delta_subject_seed")
    return delta_df


def plot_failure_ablation(failure_metrics: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    summary = (
        failure_metrics.groupby("method")["balanced_accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    summary["label"] = summary["method"].map(_method_label)
    colors = ["#999999" if m == "qca_enhanced" else "#4477AA" for m in summary["method"]]
    colors = ["#228833" if m == "p0_baseline_robust" else c for m, c in zip(summary["method"], colors)]

    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    ax.barh(summary["label"], summary["mean"], xerr=summary["std"], capsize=3, color=colors)
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("")
    ax.set_xlim(0.45, max(0.70, float(summary["mean"].max() + summary["std"].fillna(0).max() + 0.03)))
    ax.set_title("QCA failure-analysis ablation")
    sns.despine(fig=fig, ax=ax)
    _save_fig(fig, fig_dir / "qca_failure_ablation")
    return summary.sort_values("mean", ascending=False)


def diagnostic_residual_stats(run_dir: Path, fig_dir: Path) -> pd.DataFrame:
    residuals = _load_one(run_dir, "residual_audit.csv")
    diagnostics = _load_one(run_dir, "diagnostics.csv")
    method = "qca_enhanced"
    resid = residuals[residuals["method"] == method][["test_subject", "residual_error"]].copy()
    diag = diagnostics[diagnostics["method"] == method].copy()
    merged = diag.merge(resid, on="test_subject", how="inner")
    merged["mean_value"] = pd.to_numeric(merged["mean_value"])
    merged["residual_error"] = pd.to_numeric(merged["residual_error"])

    rows = []
    for component, part in merged.groupby("component"):
        if part["mean_value"].nunique() < 2:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(part["mean_value"], part["residual_error"])
        rows.append(
            {
                "component": component,
                "availability": part["availability"].iloc[0],
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "n": int(part.shape[0]),
            }
        )
    stats = pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)

    plot_df = stats.dropna(subset=["spearman_rho"]).head(10).copy()
    plot_df = plot_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.barh(plot_df["component"], plot_df["spearman_rho"], color="#4477AA")
    ax.axvline(0.0, color="0.25", linewidth=1.0)
    ax.set_xlabel("Spearman rho with QCA residual error")
    ax.set_ylabel("")
    ax.set_title("Diagnostic-residual association, QCA v1 (n=9)")
    sns.despine(fig=fig, ax=ax)
    _save_fig(fig, fig_dir / "qca_diagnostic_residual_spearman")

    if not plot_df.empty:
        top_component = stats.dropna(subset=["spearman_rho"]).iloc[0]["component"]
        scatter = merged[merged["component"] == top_component].copy()
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        sns.regplot(
            data=scatter,
            x="mean_value",
            y="residual_error",
            ax=ax,
            scatter_kws={"s": 28, "color": "#4477AA"},
            line_kws={"color": "#CC6677", "linewidth": 1.2},
            ci=None,
        )
        for _, row in scatter.iterrows():
            ax.text(row["mean_value"], row["residual_error"], str(int(row["test_subject"])), fontsize=7)
        ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
        ax.set_xlabel(top_component.replace("qri_", "").replace("_", " "))
        ax.set_ylabel("Residual error")
        ax.set_title("Top diagnostic-residual scatter")
        sns.despine(fig=fig, ax=ax)
        _save_fig(fig, fig_dir / "qca_top_diagnostic_residual_scatter")

    return stats


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot_table(index=["seed", "test_subject"], columns="method", values="balanced_accuracy")
    rows = []
    if "qca_enhanced" not in pivot.columns:
        return pd.DataFrame()
    for baseline in ["p0_baseline_robust", "p1_target_stat", "riemann_tangent_lr"]:
        if baseline not in pivot.columns:
            continue
        diff = pivot["qca_enhanced"] - pivot[baseline]
        rows.append(
            {
                "comparison": f"qca_enhanced_minus_{baseline}",
                "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()),
                "n": int(diff.shape[0]),
                "wilcoxon_p": float(wilcoxon(diff).pvalue),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    frozen_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostic_stats: pd.DataFrame,
    fig_dir: Path,
) -> None:
    lines = [
        "# PRA-MI Audit Card",
        "",
        "This report turns the completed BNCI pilot runs into paper-facing audit evidence. "
        "It emphasizes protocol separation and failure structure; it does not elevate QCA v1 into a method-superiority claim.",
        "",
        "## Main Figures",
        "",
        f"- Frozen method means: `{fig_dir / 'frozen_multiseed_method_means.pdf'}`",
        f"- Subject heatmap: `{fig_dir / 'frozen_multiseed_subject_heatmap.pdf'}`",
        f"- QCA paired deltas: `{fig_dir / 'qca_delta_subject_seed.pdf'}`",
        f"- QCA failure ablation: `{fig_dir / 'qca_failure_ablation.pdf'}`",
        f"- Diagnostic-residual Spearman: `{fig_dir / 'qca_diagnostic_residual_spearman.pdf'}`",
        "",
        "## Frozen Multi-Seed Result",
        "",
        "| Method | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in frozen_summary.itertuples(index=False):
        lines.append(f"| `{row.method}` | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |")

    lines.extend(["", "## Paired QCA Deltas", "", "| Comparison | Mean diff | Wins | N | Wilcoxon p |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in comparisons.itertuples(index=False):
        lines.append(f"| `{row.comparison}` | {row.mean_diff:.4f} | {row.wins} | {row.n} | {row.wilcoxon_p:.4f} |")

    lines.extend(["", "## QCA Failure Analysis", "", "| Method | Mean | Std | Min | Max |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in failure_summary.itertuples(index=False):
        lines.append(f"| `{row.method}` | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |")

    top_diag = diagnostic_stats.dropna(subset=["spearman_rho"]).head(8)
    lines.extend(["", "## Diagnostic-Residual Associations", "", "| Component | Availability | rho | p | n |", "| --- | --- | ---: | ---: | ---: |"])
    for row in top_diag.itertuples(index=False):
        lines.append(f"| `{row.component}` | {row.availability} | {row.spearman_rho:.3f} | {row.p_value:.3f} | {row.n} |")

    lines.extend(
        [
            "",
            "## Claim Implication",
            "",
            "- PRA-MI protocol audit remains the defensible main contribution.",
            "- Robust P0 normalization is a hard baseline and must remain central in tables.",
            "- QCA v1 remains a negative/diagnostic actionability probe because generic and shuffled controls are competitive or stronger.",
            "- Diagnostic-residual correlations are exploratory because BNCI has only 9 subjects.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate(project_root: Path, frozen_runs: list[Path], failure_run: Path, output_dir: Path, fig_dir: Path) -> Path:
    _set_style()
    frozen_metrics = _load_metrics([project_root / run for run in frozen_runs])
    failure_metrics = _load_metrics([project_root / failure_run])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frozen_summary = plot_multiseed_bar(frozen_metrics, fig_dir)
    subject_mean = plot_subject_heatmap(frozen_metrics, fig_dir)
    delta_df = plot_qca_delta(frozen_metrics, fig_dir)
    failure_summary = plot_failure_ablation(failure_metrics, fig_dir)
    diagnostic_stats = diagnostic_residual_stats(project_root / failure_run, fig_dir)
    comparisons = paired_comparisons(frozen_metrics)

    frozen_summary.to_csv(output_dir / "frozen_multiseed_method_summary.csv", index=False)
    subject_mean.to_csv(output_dir / "frozen_multiseed_subject_means.csv", index=False)
    delta_df.to_csv(output_dir / "qca_subject_seed_deltas.csv", index=False)
    failure_summary.to_csv(output_dir / "qca_failure_ablation_summary.csv", index=False)
    diagnostic_stats.to_csv(output_dir / "qca_diagnostic_residual_spearman.csv", index=False)
    comparisons.to_csv(output_dir / "qca_paired_comparisons.csv", index=False)

    report_path = output_dir / "pra_mi_audit_card.md"
    write_report(report_path, frozen_summary, failure_summary, comparisons, diagnostic_stats, fig_dir)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("work/pra_mi_audit"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figs/pra_mi_audit"))
    parser.add_argument("--failure-run", type=Path, default=Path(DEFAULT_FAILURE_RUN))
    parser.add_argument("--frozen-runs", default=",".join(DEFAULT_FROZEN_RUNS))
    args = parser.parse_args()
    frozen_runs = [Path(item) for item in args.frozen_runs.split(",") if item]
    report_path = generate(args.project_root, frozen_runs, args.failure_run, args.output_dir, args.fig_dir)
    print(report_path)


if __name__ == "__main__":
    main()
