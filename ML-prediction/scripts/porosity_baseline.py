"""
Train a baseline regressor for porosity on 7logs or 3logs TSV files.
Holdout: deepest fraction of the sorted depth column (reduces depth leakage).
Optional: save model+metadata, plot observed vs predicted porosity vs depth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse loaders from exploratory_analysis (same folder).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exploratory_analysis import normalize_columns, read_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Default training file for this study (override with --path).
OFFICIAL_TRAIN_DEFAULT = ROOT / "data" / "F03-4_7logs.txt"
OUTPUT_DIR = ROOT / "outputs"


def depth_block_split(
    depths: np.ndarray, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks: train (shallower block), test (deepest block)."""
    order = np.argsort(depths)
    n = len(depths)
    cut = int(np.floor(n * (1.0 - test_fraction)))
    train_idx = np.zeros(n, dtype=bool)
    test_idx = np.zeros(n, dtype=bool)
    train_positions = order[:cut]
    test_positions = order[cut:]
    train_idx[train_positions] = True
    test_idx[test_positions] = True
    return train_idx, test_idx


def build_pipeline(feature_names: list[str]) -> Pipeline:
    """Scaling + histogram GBDT (strong default for tabular well data)."""
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_names)],
        remainder="drop",
    )
    est = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=0,
    )
    return Pipeline(steps=[("pre", pre), ("model", est)])


def plot_porosity_vs_depth(
    depth_sorted: np.ndarray,
    y_obs_sorted: np.ndarray,
    y_pred_sorted: np.ndarray,
    depth_split: float,
    title: str,
    outfile: Path,
) -> None:
    """Save PNG: porosity vs depth, observed vs predicted; vertical line at train/test split."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(y_obs_sorted, depth_sorted, color="0.35", linewidth=1.2, label="Porosidade observada")
    ax.plot(y_pred_sorted, depth_sorted, color="tab:blue", linewidth=1.0, label="Porosidade prevista")
    ax.axhline(depth_split, color="tab:red", linestyle="--", linewidth=1.0, label="Limite treino / teste")
    ax.set_xlabel("Porosidade")
    ax.set_ylabel("Profundidade (m)")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def bootstrap_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    """
    Paired bootstrap on the test set (fixed predictions): resample indices with
    replacement and recompute MAE, RMSE, R2. Returns for each metric (p2.5, p50, p97.5).
    """
    n = int(y_true.shape[0])
    if n < 2 or n_boot < 1:
        raise ValueError("n_boot>=1 and at least two test points are required.")
    mae_s = np.empty(n_boot, dtype=float)
    rmse_s = np.empty(n_boot, dtype=float)
    r2_s = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        yt = y_true[idx]
        yp = y_pred[idx]
        mae_s[b] = mean_absolute_error(yt, yp)
        rmse_s[b] = float(np.sqrt(mean_squared_error(yt, yp)))
        r2_s[b] = r2_score(yt, yp)
    out: dict[str, tuple[float, float, float]] = {}
    for name, arr in (("MAE", mae_s), ("RMSE", rmse_s), ("R2", r2_s)):
        lo, mid, hi = (
            float(np.nanpercentile(arr, 2.5)),
            float(np.nanpercentile(arr, 50.0)),
            float(np.nanpercentile(arr, 97.5)),
        )
        out[name] = (lo, mid, hi)
    return out


def resolve_plot_path(
    args_plot: str | None,
    data_path: Path,
    no_vp: bool,
    no_ai: bool,
) -> Path | None:
    """args.plot: None skip; '' auto path with ablation suffix; else explicit path."""
    if args_plot is None:
        return None
    if args_plot == "":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe = data_path.name.replace(",", "_")
        parts: list[str] = []
        if no_vp:
            parts.append("noVp")
        if no_ai:
            parts.append("noAI")
        suffix = ("_" + "_".join(parts)) if parts else ""
        return OUTPUT_DIR / f"porosity_depth_{safe}{suffix}.png"
    return Path(args_plot).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Porosity baseline regressor.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(OFFICIAL_TRAIN_DEFAULT),
        help="Path to TSV (7logs or 3logs). Default is the official baseline file for this study.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of deepest samples used for test.",
    )
    parser.add_argument(
        "--no-vp",
        action="store_true",
        help="For 7logs only: remove Vp from features (stronger scientific ablation).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="For 7logs only: remove AI (acoustic impedance) from features.",
    )
    parser.add_argument(
        "--clip-gr-nonnegative",
        action="store_true",
        help="If set, clip GR at 0 before training (3logs noise fix).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save joblib bundle (pipeline + metadata).",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        default=None,
        metavar="CAMINHO.png",
        help="Save depth plot. Use without value for auto path under ML-prediction/outputs/.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="B",
        help="If B>0, run B paired bootstrap draws on the test set (fixed model) for MAE, RMSE, R2.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="RNG seed for --bootstrap.",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()
    raw = read_table(path)
    df, schema = normalize_columns(raw)

    if schema == "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        feature_cols = ["AC", "Rho_b", "AI", "Vp", "GR", "Vc"]
        if args.no_vp:
            feature_cols = [c for c in feature_cols if c != "Vp"]
        if args.no_ai:
            feature_cols = [c for c in feature_cols if c != "AI"]
        target_col = "Porosity"
    elif schema == "3logs_AC_GR_Phi":
        feature_cols = ["AC", "GR"]
        if args.no_vp:
            print("--no-vp applies only to 7logs schema.", file=sys.stderr)
        if args.no_ai:
            print("--no-ai applies only to 7logs schema.", file=sys.stderr)
        target_col = "Porosity"
    else:
        print(
            "This baseline expects 7logs (with Porosity) or 3logs. "
            "6logs files in data/ do not include a Porosity column.",
            file=sys.stderr,
        )
        return 2

    work = df[["Depth_m"] + feature_cols + [target_col]].copy()
    if args.clip_gr_nonnegative and "GR" in work.columns:
        work["GR"] = work["GR"].clip(lower=0.0)
    work = work.dropna().reset_index(drop=True)

    depth = work["Depth_m"].to_numpy(dtype=float)
    train_mask, test_mask = depth_block_split(depth, args.test_fraction)
    x_train = work.loc[train_mask, feature_cols]
    y_train = work.loc[train_mask, target_col]
    x_test = work.loc[test_mask, feature_cols]
    y_test = work.loc[test_mask, target_col]

    pipe = build_pipeline(feature_cols)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    print(f"official_default_file={OFFICIAL_TRAIN_DEFAULT.name}")
    print(f"file={path.name} schema={schema}")
    print(f"train_n={len(y_train)} test_n={len(y_test)}")
    print(f"features={feature_cols}")
    print(f"test MAE={mae:.5f} RMSE={rmse:.5f} R2={r2:.4f}")

    if args.bootstrap > 0:
        rng = np.random.default_rng(args.bootstrap_seed)
        y_t = y_test.to_numpy(dtype=float)
        pr = pred.astype(float)
        ci = bootstrap_test_metrics(y_t, pr, args.bootstrap, rng)
        boot_path = OUTPUT_DIR / (
            "bootstrap_test_"
            + path.name.replace(",", "_").replace(".txt", "")
            + ("_noVp" if args.no_vp else "")
            + ("_noAI" if args.no_ai else "")
            + ".csv"
        )
        points = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        n_test = int(y_t.shape[0])
        rows = [
            {
                "n_boot": int(args.bootstrap),
                "seed": int(args.bootstrap_seed),
                "test_n": n_test,
                "metric": k,
                "point": points[k],
                "p025": ci[k][0],
                "p50": ci[k][1],
                "p975": ci[k][2],
            }
            for k in ("MAE", "RMSE", "R2")
        ]
        pd.DataFrame(rows).to_csv(boot_path, index=False)
        print(
            f"bootstrap_n={args.bootstrap} seed={args.bootstrap_seed} "
            f"MAE_IC=({ci['MAE'][0]:.5f},{ci['MAE'][2]:.5f}) "
            f"RMSE_IC=({ci['RMSE'][0]:.5f},{ci['RMSE'][2]:.5f}) "
            f"R2_IC=({ci['R2'][0]:.4f},{ci['R2'][2]:.4f})"
        )
        print(f"bootstrap_csv={boot_path}")

    plot_path = resolve_plot_path(args.plot, path, args.no_vp, args.no_ai)
    if plot_path is not None:
        w_sorted = work.sort_values("Depth_m").reset_index(drop=True)
        depth_sorted = w_sorted["Depth_m"].to_numpy(dtype=float)
        y_obs_sorted = w_sorted[target_col].to_numpy(dtype=float)
        pred_sorted = pipe.predict(w_sorted[feature_cols])
        order = np.argsort(work["Depth_m"].to_numpy())
        cut = int(np.floor(len(order) * (1.0 - args.test_fraction)))
        if cut <= 0 or cut >= len(order):
            depth_split = float(depth_sorted[0])
        else:
            d_ord = work["Depth_m"].to_numpy(dtype=float)[order]
            depth_split = float(0.5 * (d_ord[cut - 1] + d_ord[cut]))
        tags: list[str] = []
        tags.append("sem_Vp" if args.no_vp else "com_Vp")
        tags.append("sem_AI" if args.no_ai else "com_AI")
        title = f"{path.name} | test R2={r2:.3f} | " + " | ".join(tags)
        plot_porosity_vs_depth(
            depth_sorted,
            y_obs_sorted,
            pred_sorted,
            depth_split,
            title,
            plot_path,
        )
        print(f"plot_saved={plot_path}")

    if args.save:
        out = Path(args.save).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": pipe,
            "feature_cols": feature_cols,
            "schema": schema,
            "training_path": str(path),
            "test_fraction": float(args.test_fraction),
            "exclude_vp": bool(args.no_vp),
            "exclude_ai": bool(args.no_ai),
            "metrics_test": {"mae": float(mae), "rmse": rmse, "r2": float(r2)},
        }
        joblib.dump(payload, out)
        print(f"saved={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
