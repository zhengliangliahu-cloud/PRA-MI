from __future__ import annotations

from collections import defaultdict
from statistics import median


def _to_float(value: str | float | int) -> float:
    return float(value)


def compute_residual_rows(metric_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        key = (row["dataset"], row["protocol_tier"], row["method"])
        groups[key].append(row)

    residual_rows: list[dict] = []
    for key, rows in groups.items():
        for row in rows:
            others = [r for r in rows if r["test_subject"] != row["test_subject"]]
            error = 1.0 - _to_float(row["balanced_accuracy"])
            if others:
                ref = median(1.0 - _to_float(other["balanced_accuracy"]) for other in others)
                residual = error - ref
            else:
                ref = ""
                residual = ""
            residual_rows.append(
                {
                    **row,
                    "error": f"{error:.6f}",
                    "median_other_subject_error": f"{ref:.6f}" if ref != "" else "",
                    "residual_error": f"{residual:.6f}" if residual != "" else "",
                }
            )
    return residual_rows


def summarize_audit(residual_rows: list[dict], diagnostic_rows: list[dict]) -> str:
    lines = [
        "# PRA-MI Audit Summary",
        "",
        "This pilot summary is generated from fixed subject-level residual definitions.",
        "It supports protocol-audit interpretation only; it does not elevate QCA-MI by itself.",
        "",
        "## Method Means",
        "",
        "| Protocol | Method | N | Mean balanced accuracy | Mean residual error |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in residual_rows:
        groups[(row["protocol_tier"], row["method"])].append(row)
    for (protocol, method), rows in sorted(groups.items()):
        bacc = [float(row["balanced_accuracy"]) for row in rows]
        residuals = [float(row["residual_error"]) for row in rows if row.get("residual_error") not in {"", None}]
        mean_resid = sum(residuals) / len(residuals) if residuals else 0.0
        lines.append(
            f"| {protocol} | {method} | {len(rows)} | {sum(bacc) / len(bacc):.4f} | {mean_resid:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- P0 rows must not use target-subject aggregate statistics.",
            "- P1 rows are unlabeled target-stat adaptation and must not be reported as strict source-only zero-shot.",
            "- BNCI2014_001 has only 9 subjects, so residual stability is pilot-level evidence.",
            "- MEI-D3 target accuracy proxies are not used as regression controls in this scaffold.",
            "",
            f"Diagnostic rows recorded: {len(diagnostic_rows)}",
            "",
        ]
    )
    return "\n".join(lines)

