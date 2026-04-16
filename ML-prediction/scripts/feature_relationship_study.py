"""
Exploratory study: correlations, VIF, mutual information (train-aligned), PCA,
condition number, permutation importance and optional SHAP (same GBDT as baseline).
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exploratory_analysis import normalize_columns, read_table  # noqa: E402
from porosity_baseline import build_pipeline, depth_block_split  # noqa: E402
from run_metadata import write_experiment_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FEATURES_7 = ["AC", "Rho_b", "AI", "Vp", "GR", "Vc"]
ELASTIC_FEATURES = ["AC", "Rho_b", "AI", "Vp"]
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


def model_input_matrix(pipe, x_df: pd.DataFrame) -> np.ndarray:
    """Matrix passed to HistGradientBoosting (after optional scaler)."""
    if "pre" in pipe.named_steps:
        return pipe.named_steps["pre"].transform(x_df)
    return x_df.to_numpy(dtype=float)


def plot_pca_cumulative_explained_variance(
    pc_index: np.ndarray,
    evr: np.ndarray,
    cum: np.ndarray,
    outfile: Path,
) -> None:
    """Bar chart per PC plus cumulative explained variance (elastic PCA, train)."""
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.bar(pc_index.astype(float), evr, width=0.55, color="0.78", edgecolor="0.45", label="Ratio per PC")
    ax1.set_xlabel("Principal component index")
    ax1.set_ylabel("Explained variance ratio")
    ax2 = ax1.twinx()
    ax2.step(
        np.concatenate([[float(pc_index[0]) - 0.5], pc_index.astype(float)]),
        np.concatenate([[0.0], cum]),
        where="post",
        color="0.15",
        linewidth=2,
        label="Cumulative",
    )
    ax2.plot(pc_index, cum, "o", color="0.15", markersize=5)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0.0, min(1.02, max(1.05 * float(cum[-1]), 0.05)))
    if len(evr) > 0:
        ax1.set_ylim(0.0, max(float(np.max(evr)) * 1.25, 0.03))
    ax1.set_xticks(pc_index)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    plt.title("PCA (elastic predictors, standardized train)")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def pca_and_condition_reports(
    x_train: pd.DataFrame, out_dir: Path, random_state: int
) -> None:
    """Condition number on full train correlation; PCA on elastic block for geometry."""
    cmat = x_train.corr().to_numpy(dtype=float)
    cond = float(np.linalg.cond(cmat))
    pd.DataFrame([{"condition_number_corr_train": cond}]).to_csv(
        out_dir / "feature_condition_train.csv", index=False
    )

    elastic = [c for c in ELASTIC_FEATURES if c in x_train.columns]
    if len(elastic) < 2:
        return
    xe = x_train[elastic]
    scaler = StandardScaler()
    xs = scaler.fit_transform(xe.to_numpy(dtype=float))
    n_comp = min(xe.shape[1], xe.shape[0])
    pca = PCA(n_components=n_comp, random_state=random_state).fit(xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pc_df = pd.DataFrame(
        {
            "PC_index": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": cum,
        }
    )
    pc_df.to_csv(out_dir / "pca_explained_variance_train.csv", index=False)
    plot_pca_cumulative_explained_variance(
        np.arange(1, len(evr) + 1),
        evr,
        cum,
        out_dir / "pca_cumulative_variance_elastic_train.png",
    )

    scaler_all = StandardScaler()
    xs_all = scaler_all.fit_transform(x_train.to_numpy(dtype=float))
    n_all = min(x_train.shape[1], x_train.shape[0])
    pca_all = PCA(n_components=n_all, random_state=random_state).fit(xs_all)
    evr_all = pca_all.explained_variance_ratio_
    cum_all = np.cumsum(evr_all)
    pd.DataFrame(
        {
            "PC_index": np.arange(1, len(evr_all) + 1),
            "explained_variance_ratio": evr_all,
            "cumulative_explained_variance": cum_all,
        }
    ).to_csv(out_dir / "pca_explained_variance_all_predictors_train.csv", index=False)


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
        "--use-scaler",
        action="store_true",
        help="If set, match porosity_baseline with StandardScaler before GBDT.",
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
        help="Random seed for MI, PCA, SHAP subsample.",
    )
    parser.add_argument(
        "--mi-also-test",
        action="store_true",
        help="If set, also write mutual_info_porosity_test.csv for split stability.",
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

    depth = work["Depth_m"].to_numpy(dtype=float)
    train_mask, test_mask = depth_block_split(depth, args.test_fraction)
    x_train = work.loc[train_mask, FEATURES_7]
    y_train = work.loc[train_mask, TARGET]
    x_test = work.loc[test_mask, FEATURES_7]
    y_test = work.loc[test_mask, TARGET]

    # Correlations on train (aligned with evaluation protocol)
    corr_p = x_train.assign(Porosity=y_train).corr(method="pearson")
    corr_s = x_train.assign(Porosity=y_train).corr(method="spearman")
    corr_p.to_csv(out_dir / "corr_pearson.csv")
    corr_s.to_csv(out_dir / "corr_spearman.csv")
    plot_corr_matrix(corr_p, f"Pearson (train) | {path.name}", out_dir / "corr_pearson.png")
    plot_corr_matrix(corr_s, f"Spearman (train) | {path.name}", out_dir / "corr_spearman.png")

    # Full-sample correlations (EDA only, not used for model diagnostics)
    corr_p_full = x_all.assign(Porosity=y_all).corr(method="pearson")
    corr_s_full = x_all.assign(Porosity=y_all).corr(method="spearman")
    corr_p_full.to_csv(out_dir / "corr_pearson_full_sample.csv")
    corr_s_full.to_csv(out_dir / "corr_spearman_full_sample.csv")

    mi_train = mutual_info_regression(
        x_train.to_numpy(dtype=float),
        y_train.to_numpy(dtype=float),
        random_state=args.random_state,
    )
    mi_train_df = pd.DataFrame({"feature": FEATURES_7, "mutual_info_porosity": mi_train})
    mi_train_df.to_csv(out_dir / "mutual_info_porosity_train.csv", index=False)
    # Backward-compatible name for pipelines expecting mutual_info_porosity.csv
    mi_train_df.to_csv(out_dir / "mutual_info_porosity.csv", index=False)

    if args.mi_also_test and len(x_test) > 1:
        mi_test = mutual_info_regression(
            x_test.to_numpy(dtype=float),
            y_test.to_numpy(dtype=float),
            random_state=args.random_state,
        )
        pd.DataFrame({"feature": FEATURES_7, "mutual_info_porosity": mi_test}).to_csv(
            out_dir / "mutual_info_porosity_test.csv", index=False
        )

    vif_df = compute_vif_frame(x_train)
    vif_df.to_csv(out_dir / "vif.csv", index=False)

    pca_and_condition_reports(x_train, out_dir, args.random_state)

    pipe = build_pipeline(FEATURES_7, use_scaler=args.use_scaler, random_state=args.random_state)
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
    perm_df["likely_redundant_or_noise"] = perm_df["importance_mean"] <= 0.0
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
        "use_scaler": bool(args.use_scaler),
        "mi_primary_split": "train",
        "corr_heatmaps_split": "train",
    }
    pd.Series(meta).to_csv(out_dir / "run_meta.csv", header=False)

    write_experiment_json(
        out_dir / "experiment.json",
        {"script": "feature_relationship_study.py", **meta},
        repo_root=ROOT.parent,
    )

    if args.shap_max > 0 and len(x_test) > 0:
        rng = np.random.default_rng(args.random_state)
        n_take = min(args.shap_max, len(x_test))
        idx = rng.choice(x_test.index.to_numpy(), size=n_take, replace=False)
        x_sub = x_test.loc[idx]
        model = pipe.named_steps["model"]
        x_model = model_input_matrix(pipe, x_sub)
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_model)
            plt.figure()
            shap.summary_plot(
                sv,
                x_model,
                feature_names=FEATURES_7,
                show=False,
                plot_size=(9, 6),
            )
            plt.tight_layout()
            plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"shap_saved={out_dir / 'shap_summary.png'} n={n_take} space=model_input")
        except Exception as exc:
            print(f"SHAP skipped: {exc}", file=sys.stderr)

    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
