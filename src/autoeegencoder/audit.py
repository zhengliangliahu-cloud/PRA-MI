from __future__ import annotations

import sys
from pathlib import Path

from autoeegencoder.analysis.residuals import compute_residual_rows, summarize_audit
from autoeegencoder.utils.config import project_root
from autoeegencoder.utils.io import latest_run_dir, read_csv_rows, write_csv_rows, write_text


def _parse_args(argv: list[str]) -> dict[str, str]:
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def audit_run(run_dir: Path) -> None:
    metrics_path = run_dir / "subject_metrics.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    metric_rows = read_csv_rows(metrics_path)
    diagnostic_rows = read_csv_rows(diagnostics_path) if diagnostics_path.exists() else []
    residual_rows = compute_residual_rows(metric_rows)
    write_csv_rows(run_dir / "residual_audit.csv", residual_rows)
    write_text(run_dir / "audit_summary.md", summarize_audit(residual_rows, diagnostic_rows))
    print(f"Audit complete: {run_dir / 'audit_summary.md'}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if "run_dir" in args:
        run_dir = Path(args["run_dir"])
        if not run_dir.is_absolute():
            run_dir = project_root() / run_dir
    else:
        run_dir = latest_run_dir(project_root() / "results")
    audit_run(run_dir)


if __name__ == "__main__":
    main()

