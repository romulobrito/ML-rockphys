#!/usr/bin/env python3
"""
Run EDA, feature study, GR-Vc diagnostics, porosity baselines, optional sensitivity
and leave-one-well-out; copy figure PNGs used by relatorio_porosidade_ml.tex into docs/figs/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DOCS_FIGS = ROOT / "docs" / "figs"
_VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(_VENV_PY) if _VENV_PY.is_file() else sys.executable


def run_cmd(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=str(SCRIPTS))


def copy_fig(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing output figure: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"fig -> {dst.relative_to(ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate ML outputs and figs for LaTeX.")
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory_analysis.py (faster if only ML refresh).",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip depth-cut grid, max_depth grid, and LOO CSV exports.",
    )
    args = parser.parse_args()

    if not args.skip_eda:
        run_cmd([PY, str(SCRIPTS / "exploratory_analysis.py")])

    run_cmd(
        [
            PY,
            str(SCRIPTS / "feature_relationship_study.py"),
            "--path",
            str(DATA / "F03-4_7logs.txt"),
        ]
    )
    run_cmd(
        [
            PY,
            str(SCRIPTS / "gr_vc_rank_dependence.py"),
            "--path",
            str(DATA / "F03-4_7logs.txt"),
        ]
    )

    f03 = DATA / "F03-4_7logs.txt"
    f06 = DATA / "F06-1_7logs.txt"
    porosity_jobs: list[tuple[list[str], Path, Path]] = [
        (
            [PY, str(SCRIPTS / "porosity_baseline.py"), "--path", str(f03), "--plot"],
            OUT / "porosity_depth_F03-4_7logs.txt.png",
            DOCS_FIGS / "porosity_F03-4_com_Vp.png",
        ),
        (
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--path",
                str(f03),
                "--no-vp",
                "--plot",
            ],
            OUT / "porosity_depth_F03-4_7logs.txt_noVp.png",
            DOCS_FIGS / "porosity_F03-4_sem_Vp.png",
        ),
        (
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--path",
                str(f03),
                "--no-ai",
                "--plot",
            ],
            OUT / "porosity_depth_F03-4_7logs.txt_noAI.png",
            DOCS_FIGS / "porosity_F03-4_sem_AI.png",
        ),
        (
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--path",
                str(f03),
                "--no-vp",
                "--no-ai",
                "--plot",
            ],
            OUT / "porosity_depth_F03-4_7logs.txt_noVp_noAI.png",
            DOCS_FIGS / "porosity_F03-4_sem_Vp_sem_AI.png",
        ),
        (
            [PY, str(SCRIPTS / "porosity_baseline.py"), "--path", str(f06), "--plot"],
            OUT / "porosity_depth_F06-1_7logs.txt.png",
            DOCS_FIGS / "porosity_F06-1_com_Vp.png",
        ),
    ]
    for argv, src, dst in porosity_jobs:
        run_cmd(argv)
        copy_fig(src, dst)

    fs = OUT / "feature_study_F03-4_7logs"
    copy_fig(fs / "corr_pearson.png", DOCS_FIGS / "study_F03-4_corr_pearson.png")
    copy_fig(fs / "corr_spearman.png", DOCS_FIGS / "study_F03-4_corr_spearman.png")
    copy_fig(fs / "permutation_importance.png", DOCS_FIGS / "study_F03-4_perm_importance.png")
    copy_fig(fs / "shap_summary.png", DOCS_FIGS / "study_F03-4_shap_summary.png")
    copy_fig(
        fs / "pca_cumulative_variance_elastic_train.png",
        DOCS_FIGS / "study_F03-4_pca_cumulative_variance.png",
    )

    gr = OUT / "gr_vc_rank_F03-4_7logs"
    copy_fig(gr / "gr_vc_ranks_scatter.png", DOCS_FIGS / "gr_vc_F03-4_ranks_scatter.png")
    copy_fig(gr / "rank_method_sensitivity.png", DOCS_FIGS / "gr_vc_F03-4_rank_method_sensitivity.png")

    if not args.skip_sensitivity:
        run_cmd(
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--path",
                str(f03),
                "--depth-cut-grid",
                "0.15,0.2,0.25",
            ]
        )
        run_cmd(
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--path",
                str(f03),
                "--max-depth-grid",
                "4,6,8",
            ]
        )
        wells = [
            str(DATA / "F02-1_7logs.txt"),
            str(DATA / "F03-2_7logs.txt"),
            str(DATA / "F03-4_7logs.txt"),
            str(DATA / "F06-1_7logs.txt"),
        ]
        run_cmd(
            [
                PY,
                str(SCRIPTS / "porosity_baseline.py"),
                "--eval-protocol",
                "leave_one_well_out",
                "--well-paths",
                *wells,
            ]
        )

    run_cmd(
        [
            PY,
            str(SCRIPTS / "porosity_baseline.py"),
            "--path",
            str(f03),
            "--bootstrap",
            "2000",
            "--bootstrap-seed",
            "0",
        ]
    )

    print("run_report_pipeline: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
