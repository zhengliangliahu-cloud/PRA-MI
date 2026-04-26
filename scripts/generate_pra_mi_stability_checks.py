from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes


DEFAULT_PROTOCOL_RUNS = [
    "results/bnci_pra_mi_frozen_full_bnci2014_001_seed-2026_20260424_145549",
    "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2026_20260424_172420",
    "results/cho2017_pra_mi_smoke_cho2017_seed-2026_20260425_132854",
    "results/lee2019_pra_mi_smoke_lee2019_mi_seed-2026_20260424_181346",
]
DEFAULT_DIAGNOSTIC_RUN = "results/bnci_pra_mi_qca_failure_full_bnci2014_001_seed-2026_20260424_172420"


def _read_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _as_bool_text(value: bool) -> str:
    return "yes" if value else "no"


def protocol_audit_table(project_root: Path, run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for run_rel in run_dirs:
        run_dir = project_root / run_rel
        metrics = _read_csv(run_dir, "subject_metrics.csv")
        ledger = _read_csv(run_dir, "harmonization_ledger.csv")
        diagnostics = _read_csv(run_dir, "diagnostics.csv")
        run_name = run_dir.name
        for (dataset, method), method_metrics in metrics.groupby(["dataset", "method"]):
            protocol_values = sorted(str(v) for v in method_metrics["protocol_tier"].dropna().unique())
            protocol_tier = ";".join(protocol_values)
            ledger_part = ledger[(ledger["dataset"] == dataset) & (ledger["method"] == method)].copy()
            diag_part = diagnostics[
                (diagnostics["dataset"] == dataset) & (diagnostics["method"] == method)
            ].copy()

            has_target_aggregate = bool(
                ((ledger_part["action"] == "use_target_aggregate") | (ledger_part["availability"] == "D1")).any()
            )
            p0_target_aggregate = bool(
                (
                    (ledger_part["protocol_tier"] == "P0")
                    & ((ledger_part["action"] == "use_target_aggregate") | (ledger_part["availability"] == "D1"))
                ).any()
            )
            target_label_terms = ledger_part.fillna("").astype(str).agg(" ".join, axis=1).str.contains(
                "target label|test label|oracle accuracy|D3", case=False, regex=True
            )
            has_target_label_or_oracle = bool(target_label_terms.any())
            allowed_method_availability = set()
            forbidden_method_availability = set()
            if not diag_part.empty:
                method_side = diag_part[diag_part["can_enter_method"] == "yes"]
                allowed_method_availability = set(str(v) for v in method_side["availability"].dropna().unique())
                forbidden_method_availability = allowed_method_availability.difference({"D0a", "D0b"})

            rows.append(
                {
                    "run_dir": run_name,
                    "dataset": dataset,
                    "method": method,
                    "protocol_tier": protocol_tier,
                    "n_subjects": int(method_metrics["test_subject"].nunique()),
                    "mean_balanced_accuracy": float(method_metrics["balanced_accuracy"].astype(float).mean()),
                    "uses_target_aggregate": _as_bool_text(has_target_aggregate),
                    "target_aggregate_in_P0": _as_bool_text(p0_target_aggregate),
                    "uses_target_label_or_D3_oracle": _as_bool_text(has_target_label_or_oracle),
                    "method_side_availability": ";".join(sorted(allowed_method_availability)) or "none",
                    "forbidden_method_side_availability": ";".join(sorted(forbidden_method_availability)) or "none",
                    "ledger_actions": ";".join(sorted(str(v) for v in ledger_part["action"].dropna().unique()))
                    or "none",
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "run_dir", "protocol_tier", "method"])


def _bootstrap_spearman(
    values: np.ndarray,
    residuals: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    boot = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(values[idx]).size < 2 or np.unique(residuals[idx]).size < 2:
            continue
        rho, _ = spearmanr(values[idx], residuals[idx])
        if np.isfinite(rho):
            boot.append(float(rho))
    if not boot:
        return np.nan, np.nan, np.nan
    arr = np.asarray(boot, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _jackknife_spearman(values: np.ndarray, residuals: np.ndarray) -> tuple[float, float, float, float]:
    rhos = []
    n = len(values)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        if np.unique(values[keep]).size < 2 or np.unique(residuals[keep]).size < 2:
            continue
        rho, _ = spearmanr(values[keep], residuals[keep])
        if np.isfinite(rho):
            rhos.append(float(rho))
    if not rhos:
        return np.nan, np.nan, np.nan, np.nan
    arr = np.asarray(rhos, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std(ddof=0))


def diagnostic_residual_stability(
    project_root: Path,
    run_dir: Path,
    *,
    method: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_run_dir = project_root / run_dir
    residuals = _read_csv(full_run_dir, "residual_audit.csv")
    diagnostics = _read_csv(full_run_dir, "diagnostics.csv")
    resid = residuals[residuals["method"] == method][["test_subject", "residual_error"]].copy()
    diag = diagnostics[diagnostics["method"] == method].copy()
    merged = diag.merge(resid, on="test_subject", how="inner")
    merged["mean_value"] = pd.to_numeric(merged["mean_value"])
    merged["residual_error"] = pd.to_numeric(merged["residual_error"])
    rng = np.random.default_rng(seed)

    summary_rows: list[dict] = []
    jackknife_rows: list[dict] = []
    for component, part in merged.groupby("component"):
        part = part.sort_values("test_subject")
        x = part["mean_value"].to_numpy(dtype=float)
        y = part["residual_error"].to_numpy(dtype=float)
        if len(part) < 4 or np.unique(x).size < 2:
            continue
        rho, p_value = spearmanr(x, y)
        boot_mean, boot_low, boot_high = _bootstrap_spearman(x, y, rng=rng, n_boot=n_boot)
        jk_mean, jk_min, jk_max, jk_std = _jackknife_spearman(x, y)
        slope, intercept, slope_low, slope_high = theilslopes(y, x, alpha=0.95)
        sign_stable = bool(np.isfinite(jk_min) and np.isfinite(jk_max) and np.sign(jk_min) == np.sign(jk_max))

        summary_rows.append(
            {
                "run_dir": full_run_dir.name,
                "method": method,
                "component": component,
                "availability": part["availability"].iloc[0],
                "n": int(part.shape[0]),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
                "bootstrap_rho_mean": boot_mean,
                "bootstrap_rho_ci_low": boot_low,
                "bootstrap_rho_ci_high": boot_high,
                "jackknife_rho_mean": jk_mean,
                "jackknife_rho_min": jk_min,
                "jackknife_rho_max": jk_max,
                "jackknife_rho_std": jk_std,
                "jackknife_sign_stable": _as_bool_text(sign_stable),
                "theil_sen_slope": float(slope),
                "theil_sen_slope_ci_low": float(slope_low),
                "theil_sen_slope_ci_high": float(slope_high),
            }
        )
        for _, row in part.iterrows():
            keep = part["test_subject"] != row["test_subject"]
            jk_rho, jk_p = spearmanr(part.loc[keep, "mean_value"], part.loc[keep, "residual_error"])
            jackknife_rows.append(
                {
                    "component": component,
                    "left_out_subject": int(row["test_subject"]),
                    "rho_after_leave_one_out": float(jk_rho),
                    "p_after_leave_one_out": float(jk_p),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )
    jackknife = pd.DataFrame(jackknife_rows).sort_values(["component", "left_out_subject"])
    return summary, jackknife


def write_markdown(
    output_path: Path,
    protocol: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    diagnostic_run: Path,
) -> None:
    leakage_count = int((protocol["target_aggregate_in_P0"] == "yes").sum())
    forbidden_count = int((protocol["forbidden_method_side_availability"] != "none").sum())
    lines = [
        "# PRA-MI Protocol and Diagnostic Stability Checks",
        "",
        "This report focuses on reviewer-facing robustness checks for the conservative PRA-MI paper story.",
        "",
        "## Protocol Leakage Check",
        "",
        f"- P0 rows using target aggregate statistics: `{leakage_count}`",
        f"- Methods with forbidden method-side diagnostic availability: `{forbidden_count}`",
        "- `uses_target_label_or_D3_oracle` should remain `no` for model-side methods.",
        "",
        "| Dataset | Run | Method | Protocol | Target aggregate | P0 leakage | Method-side availability |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in protocol.itertuples(index=False):
        lines.append(
            f"| `{row.dataset}` | `{row.run_dir}` | `{row.method}` | {row.protocol_tier} | "
            f"{row.uses_target_aggregate} | {row.target_aggregate_in_P0} | `{row.method_side_availability}` |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic-Residual Stability",
            "",
            f"Diagnostic run: `{diagnostic_run}`",
            "",
            "| Component | Availability | rho | p | bootstrap 95% CI | jackknife range | sign stable |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in stability.head(12).itertuples(index=False):
        lines.append(
            f"| `{row.component}` | {row.availability} | {row.spearman_rho:.3f} | "
            f"{row.spearman_p:.3f} | [{row.bootstrap_rho_ci_low:.3f}, {row.bootstrap_rho_ci_high:.3f}] | "
            f"[{row.jackknife_rho_min:.3f}, {row.jackknife_rho_max:.3f}] | {row.jackknife_sign_stable} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Protocol separation is the primary evidence: P0 should show no D1 target aggregate usage.",
            "- Diagnostic-residual associations are exploratory on BNCI because there are only 9 subjects.",
            "- A wide bootstrap interval or unstable jackknife sign means the diagnostic should not be promoted into a main biological claim.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("work/pra_mi_stability"))
    parser.add_argument("--protocol-runs", default=",".join(DEFAULT_PROTOCOL_RUNS))
    parser.add_argument("--diagnostic-run", type=Path, default=Path(DEFAULT_DIAGNOSTIC_RUN))
    parser.add_argument("--method", default="qca_enhanced")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    protocol_runs = [Path(item) for item in args.protocol_runs.split(",") if item]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_audit_table(args.project_root, protocol_runs)
    stability, jackknife = diagnostic_residual_stability(
        args.project_root,
        args.diagnostic_run,
        method=args.method,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    protocol.to_csv(args.output_dir / "protocol_leakage_table.csv", index=False)
    stability.to_csv(args.output_dir / "diagnostic_residual_stability.csv", index=False)
    jackknife.to_csv(args.output_dir / "diagnostic_residual_jackknife_subjects.csv", index=False)
    write_markdown(
        args.output_dir / "pra_mi_stability_checks.md",
        protocol,
        stability,
        diagnostic_run=args.diagnostic_run,
    )
    print(args.output_dir / "pra_mi_stability_checks.md")


if __name__ == "__main__":
    main()
