from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BNCI_MAIN = Path("results/bnci_pra_mi_frozen_full_bnci2014_001_seed-2026_20260424_145549")
DEFAULT_BNCI_CLASSICAL = Path("results/bnci_classical_fairness_full_bnci2014_001_seed-2026_20260424_191348")


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "subject_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(float)
    return frame


def summarize_run(run_dir: Path) -> pd.DataFrame:
    metrics = _load_metrics(run_dir)
    summary = (
        metrics.groupby(["dataset", "method", "protocol_tier"], as_index=False)
        .agg(
            mean=("balanced_accuracy", "mean"),
            std=("balanced_accuracy", "std"),
            min=("balanced_accuracy", "min"),
            max=("balanced_accuracy", "max"),
        )
    )
    return summary


def merge_bnci_main_and_classical(bnci_main: pd.DataFrame, bnci_classical: pd.DataFrame) -> pd.DataFrame:
    keep_main = {"p0_baseline_robust", "p1_target_stat", "qca_enhanced"}
    keep_classical = {
        "riemann_tangent_lr_fixed",
        "riemann_tangent_lr_tuned",
        "csp_logreg_tuned",
        "csp_svm_tuned",
        "riemann_mdm_tuned",
    }
    left = bnci_main[bnci_main["method"].isin(keep_main)].copy()
    right = bnci_classical[bnci_classical["method"].isin(keep_classical)].copy()
    table = pd.concat([left, right], ignore_index=True)
    order = [
        "p0_baseline_robust",
        "qca_enhanced",
        "p1_target_stat",
        "riemann_tangent_lr_tuned",
        "riemann_tangent_lr_fixed",
        "csp_logreg_tuned",
        "csp_svm_tuned",
        "riemann_mdm_tuned",
    ]
    table["rank_order"] = table["method"].map({name: idx for idx, name in enumerate(order)})
    table = table.sort_values(["dataset", "rank_order"]).drop(columns=["rank_order"])
    return table


def write_report(
    output_path: Path,
    bnci_table: pd.DataFrame,
    cho_table: pd.DataFrame | None,
    lee_table: pd.DataFrame | None,
) -> None:
    lines = [
        "# Paper Main Tables",
        "",
        "This report prepares paper-facing main tables for the conservative PRA-MI story.",
        "",
        "## BNCI Main Table",
        "",
        "| Method | Protocol | Mean | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in bnci_table.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.protocol_tier} | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |"
        )
    if cho_table is not None:
        lines.extend(
            [
                "",
                "## Cho2017 Full Table",
                "",
                "| Method | Protocol | Mean | Std | Min | Max |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in cho_table.itertuples(index=False):
            lines.append(
                f"| `{row.method}` | {row.protocol_tier} | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |"
            )
    if lee_table is not None:
        lines.extend(
            [
                "",
                "## Lee2019 Full Table",
                "",
                "| Method | Protocol | Mean | Std | Min | Max |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in lee_table.itertuples(index=False):
            lines.append(
                f"| `{row.method}` | {row.protocol_tier} | {row.mean:.4f} | {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Claim Use",
            "",
            "- The BNCI table is the main evidence that robust P0 remains the hard source-only baseline.",
            "- The tuned classical rows should be in the paper to reduce the risk that the neural baseline only wins because classical methods were undertuned.",
            "- The external full tables, when present, support protocol transfer and help lift the paper beyond single-dataset pilot status.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("work/paper_tables"))
    parser.add_argument("--bnci-main", type=Path, default=DEFAULT_BNCI_MAIN)
    parser.add_argument("--bnci-classical", type=Path, default=DEFAULT_BNCI_CLASSICAL)
    parser.add_argument("--cho-full", type=Path, default=None)
    parser.add_argument("--lee-full", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bnci_main = summarize_run(args.project_root / args.bnci_main)
    bnci_classical = summarize_run(args.project_root / args.bnci_classical)
    bnci_table = merge_bnci_main_and_classical(bnci_main, bnci_classical)

    cho_table = None
    if args.cho_full is not None:
        cho_table = summarize_run(args.project_root / args.cho_full)
    lee_table = None
    if args.lee_full is not None:
        lee_table = summarize_run(args.project_root / args.lee_full)

    bnci_table.to_csv(args.output_dir / "bnci_main_table.csv", index=False)
    if cho_table is not None:
        cho_table.to_csv(args.output_dir / "cho2017_full_table.csv", index=False)
    if lee_table is not None:
        lee_table.to_csv(args.output_dir / "lee2019_full_table.csv", index=False)
    report = args.output_dir / "paper_main_tables.md"
    write_report(report, bnci_table, cho_table, lee_table)
    print(report)


if __name__ == "__main__":
    main()
