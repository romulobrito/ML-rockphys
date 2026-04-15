"""
Exploratory study: correlations, VIF-like scores, mutual information with Porosity,
permutation importance (same pipeline as porosity_baseline), optional SHAP summary.
Outputs CSV and PNG under a chosen directory (default: outputs/feature_study_<stem>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exploratory_analysis import normalize_columns, read_table  # noqa: E402
from porosity_baseline import build_pipeline, depth_block_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FEATURES_7 = ["AC", "Rho_b", "AI", "Vp", "GR", "Vc"]
TARGET = "Porosity"


def compute_vif_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor via auxiliary R^2 (sklearn only)."""
    names = list(x.columns)
    rows: list[tuple[str, float]] = []
    for j, name in enumerate(names):
        y_col = x[name].to_numpy(dtype=float)
        others = x.drop(columns=[name])
        if others.shape[1] == 0:
            rows.append((name, 1.0))
            continue
        lr = LinearRegression().fit(others.to_numpy(dtype=float), y_col)
        r2 = float(lr.score(others.to_numpy(dtype=float), y_col))
        if r2 >= 0.999999:
            rows.append((name, float("inf")))
        else:
            rows.append((name, 1.0 / (1.0 - r2)))
    return pd.DataFrame(rows, columns=["feature", "VIF"])


def plot_corr_matrix(corr: pd.DataFrame, title: str, outfile: Path) -> None:
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_permutation_importance(
    importances_mean: np.ndarray,
    importances_std: np.ndarray,
    names: list[str],
    outfile: Path,
) -> None:
    order = np.argsort(importances_mean)
    y_pos = np.arange(len(names))
    plt.figure(figsize=(7, 5))
    plt.barh(
        y_pos,
        importances_mean[order],
        xerr=importances_std[order],
        color="steelblue",
        ecolor="0.35",
        capsize=3,
    )
    plt.yticks(y_pos, [names[i] for i in order])
    plt.xlabel("Mean decrease in R2 (permutation)")
    plt.title("Permutation importance (test set)")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature relationship study (7logs).")
    parser.add_argument(
        "--path",
        type=str,
        default=str(ROOT / "data" / "F03-4_7logs.txt"),
        help="Path to 7logs TSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: outputs/feature_study_<file_stem>).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Same depth block as porosity_baseline for permutation importance.",
    )
    parser.add_argument(
        "--shap-max",
        type=int,
        default=300,
        help="Max rows from test set for SHAP (0 disables).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for MI and SHAP subsample.",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()
    stem = path.name.replace(",", "_").replace(".txt", "")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (ROOT / "outputs" / f"feature_study_{stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(path)
    df, schema = normalize_columns(raw)
    if schema != "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        print("Este script espera esquema 7logs.", file=sys.stderr)
        return 2

    work = df[["Depth_m"] + FEATURES_7 + [TARGET]].dropna().reset_index(drop=True)
    x_all = work[FEATURES_7]
    y_all = work[TARGET]

    corr_p = x_all.assign(Porosity=y_all).corr(method="pearson")
    corr_s = x_all.assign(Porosity=y_all).corr(method="spearman")
    corr_p.to_csv(out_dir / "corr_pearson.csv")
    corr_s.to_csv(out_dir / "corr_spearman.csv")
    plot_corr_matrix(corr_p, f"Pearson | {path.name}", out_dir / "corr_pearson.png")
    plot_corr_matrix(corr_s, f"Spearman | {path.name}", out_dir / "corr_spearman.png")

    mi = mutual_info_regression(
        x_all.to_numpy(dtype=float),
        y_all.to_numpy(dtype=float),
        random_state=args.random_state,
    )
    mi_df = pd.DataFrame({"feature": FEATURES_7, "mutual_info_porosity": mi})
    mi_df.to_csv(out_dir / "mutual_info_porosity.csv", index=False)

    vif_df = compute_vif_frame(x_all)
    vif_df.to_csv(out_dir / "vif.csv", index=False)

    depth = work["Depth_m"].to_numpy(dtype=float)
    train_mask, test_mask = depth_block_split(depth, args.test_fraction)
    x_train = work.loc[train_mask, FEATURES_7]
    y_train = work.loc[train_mask, TARGET]
    x_test = work.loc[test_mask, FEATURES_7]
    y_test = work.loc[test_mask, TARGET]

    pipe = build_pipeline(FEATURES_7)
    pipe.fit(x_train, y_train)

    perm = permutation_importance(
        pipe,
        x_test,
        y_test,
        n_repeats=25,
        random_state=args.random_state,
        scoring="r2",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": FEATURES_7,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(out_dir / "permutation_importance_test.csv", index=False)
    plot_permutation_importance(
        perm.importances_mean,
        perm.importances_std,
        FEATURES_7,
        out_dir / "permutation_importance.png",
    )

    meta = {
        "file": str(path),
        "n_rows": len(work),
        "train_n": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
        "test_fraction": float(args.test_fraction),
        "features": FEATURES_7,
        "out_dir": str(out_dir),
    }
    pd.Series(meta).to_csv(out_dir / "run_meta.csv", header=False)

    if args.shap_max > 0 and len(x_test) > 0:
        rng = np.random.default_rng(args.random_state)
        n_take = min(args.shap_max, len(x_test))
        idx = rng.choice(x_test.index.to_numpy(), size=n_take, replace=False)
        x_sub = x_test.loc[idx]
        y_sub = y_test.loc[idx]
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
        x_t = pre.transform(x_sub)
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_t)
            plt.figure()
            shap.summary_plot(
                sv,
                x_sub,
                feature_names=FEATURES_7,
                show=False,
                plot_size=(9, 6),
            )
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"shap_saved={out_dir / 'shap_summary.png'} n={n_take}")
        except Exception as exc:
            print(f"SHAP skipped: {exc}", file=sys.stderr)

    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
