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
from run_metadata import write_experiment_json  # noqa: E402

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


def build_pipeline(
    feature_names: list[str],
    *,
    use_scaler: bool = False,
    random_state: int = 0,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    max_iter: int = 200,
) -> Pipeline:
    """
    Histogram GBDT. Optional StandardScaler (for linear-style comparisons);
    trees do not require scaling (default use_scaler=False).
    """
    est = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
    if use_scaler:
        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(), feature_names)],
            remainder="drop",
        )
        return Pipeline(steps=[("pre", pre), ("model", est)])
    return Pipeline(steps=[("model", est)])


def load_7logs_or_3logs_frame(
    path: Path,
    *,
    no_vp: bool,
    no_ai: bool,
    clip_gr_nonnegative: bool,
) -> tuple[pd.DataFrame, list[str], str, str]:
    """Return (work frame with Depth + features + target, feature_cols, schema, target_col)."""
    raw = read_table(path)
    df, schema = normalize_columns(raw)
    if schema == "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        feature_cols = ["AC", "Rho_b", "AI", "Vp", "GR", "Vc"]
        if no_vp:
            feature_cols = [c for c in feature_cols if c != "Vp"]
        if no_ai:
            feature_cols = [c for c in feature_cols if c != "AI"]
        target_col = "Porosity"
    elif schema == "3logs_AC_GR_Phi":
        feature_cols = ["AC", "GR"]
        target_col = "Porosity"
    else:
        raise ValueError(
            "Expected 7logs (with Porosity) or 3logs; 6logs files lack Porosity."
        )
    work = df[["Depth_m"] + feature_cols + [target_col]].copy()
    if clip_gr_nonnegative and "GR" in work.columns:
        work["GR"] = work["GR"].clip(lower=0.0)
    work = work.dropna().reset_index(drop=True)
    return work, feature_cols, schema, target_col


def run_leave_one_well_out_7logs(
    paths: list[Path],
    *,
    no_vp: bool,
    no_ai: bool,
    clip_gr: bool,
    use_scaler: bool,
    random_state: int,
    out_csv: Path,
) -> pd.DataFrame:
    """Train on union of all wells except the held-out file; test on held-out well."""
    rows: list[dict[str, object]] = []
    for test_path in paths:
        train_paths = [p for p in paths if p.resolve() != test_path.resolve()]
        if not train_paths:
            continue
        tw, t_feats, t_schema, t_targ = load_7logs_or_3logs_frame(
            test_path, no_vp=no_vp, no_ai=no_ai, clip_gr_nonnegative=clip_gr
        )
        if t_schema != "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
            raise ValueError("leave_one_well_out requires 7logs files for this study.")
        train_parts: list[pd.DataFrame] = []
        for tp in train_paths:
            w, feats, sch, _ = load_7logs_or_3logs_frame(
                tp, no_vp=no_vp, no_ai=no_ai, clip_gr_nonnegative=clip_gr
            )
            if sch != t_schema or feats != t_feats:
                raise ValueError(f"Schema or feature mismatch: {tp.name} vs {test_path.name}")
            train_parts.append(w)
        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        x_train = train_df[t_feats]
        y_train = train_df[t_targ]
        x_test = tw[t_feats]
        y_test = tw[t_targ]
        pipe = build_pipeline(
            t_feats,
            use_scaler=use_scaler,
            random_state=random_state,
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        rows.append(
            {
                "test_well_file": test_path.name,
                "train_n": int(len(y_train)),
                "test_n": int(len(y_test)),
                "MAE": float(mae),
                "RMSE": rmse,
                "R2": float(r2),
            }
        )
    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def parse_float_list(s: str) -> list[float]:
    """Parse '0.15,0.2,0.25' into floats."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


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
        "--eval-protocol",
        type=str,
        choices=("depth_holdout", "leave_one_well_out"),
        default="depth_holdout",
        help="depth_holdout: single file, deepest fraction test. "
        "leave_one_well_out: needs --well-paths (2+ 7logs files).",
    )
    parser.add_argument(
        "--well-paths",
        type=str,
        nargs="*",
        default=[],
        metavar="PATH",
        help="For leave_one_well_out: list of 7logs TSV paths (same feature schema).",
    )
    parser.add_argument(
        "--use-scaler",
        action="store_true",
        help="If set, apply StandardScaler before GBDT (not required for trees).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of deepest samples used for test (depth_holdout only).",
    )
    parser.add_argument(
        "--depth-cut-grid",
        type=str,
        default="",
        help="Comma-separated test fractions, e.g. 0.15,0.2,0.25. "
        "Writes depth_cut_sensitivity.csv and skips single-run metrics.",
    )
    parser.add_argument(
        "--max-depth-grid",
        type=str,
        default="",
        help="Comma-separated max_depth values for GBDT, e.g. 4,6,8. "
        "Writes gbdt_max_depth_sensitivity.csv (depth_holdout on --path).",
    )
    parser.add_argument(
        "--write-experiment-json",
        type=str,
        default="",
        help="Optional path to write experiment metadata JSON.",
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
    random_state = 0

    if args.eval_protocol == "leave_one_well_out":
        wells = [Path(p).resolve() for p in args.well_paths]
        if len(wells) < 2:
            print("leave_one_well_out requires at least two paths in --well-paths.", file=sys.stderr)
            return 2
        out_loo = OUTPUT_DIR / "loo_well_metrics.csv"
        loo_df = run_leave_one_well_out_7logs(
            wells,
            no_vp=args.no_vp,
            no_ai=args.no_ai,
            clip_gr=args.clip_gr_nonnegative,
            use_scaler=args.use_scaler,
            random_state=random_state,
            out_csv=out_loo,
        )
        print(f"loo_csv={out_loo}")
        print(loo_df.to_string(index=False))
        if args.write_experiment_json:
            write_experiment_json(
                Path(args.write_experiment_json),
                {
                    "script": "porosity_baseline.py",
                    "eval_protocol": "leave_one_well_out",
                    "well_paths": [str(p) for p in wells],
                    "use_scaler": bool(args.use_scaler),
                    "no_vp": bool(args.no_vp),
                    "no_ai": bool(args.no_ai),
                    "loo_summary": loo_df.to_dict(orient="records"),
                },
                repo_root=ROOT.parent,
            )
            print(f"experiment_json={args.write_experiment_json}")
        return 0

    try:
        work, feature_cols, schema, target_col = load_7logs_or_3logs_frame(
            path,
            no_vp=args.no_vp,
            no_ai=args.no_ai,
            clip_gr_nonnegative=args.clip_gr_nonnegative,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.no_vp and schema != "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        print("--no-vp applies only to 7logs schema.", file=sys.stderr)
    if args.no_ai and schema != "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        print("--no-ai applies only to 7logs schema.", file=sys.stderr)

    if args.depth_cut_grid:
        fracs = parse_float_list(args.depth_cut_grid)
        sens_rows: list[dict[str, float | str]] = []
        depth = work["Depth_m"].to_numpy(dtype=float)
        for tf in fracs:
            train_mask, test_mask = depth_block_split(depth, tf)
            x_train = work.loc[train_mask, feature_cols]
            y_train = work.loc[train_mask, target_col]
            x_test = work.loc[test_mask, feature_cols]
            y_test = work.loc[test_mask, target_col]
            if len(y_test) < 2 or len(y_train) < 2:
                continue
            pipe = build_pipeline(
                feature_cols,
                use_scaler=args.use_scaler,
                random_state=random_state,
            )
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test)
            sens_rows.append(
                {
                    "file": path.name,
                    "test_fraction": float(tf),
                    "train_n": float(len(y_train)),
                    "test_n": float(len(y_test)),
                    "MAE": float(mean_absolute_error(y_test, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": float(r2_score(y_test, pred)),
                }
            )
        sens_df = pd.DataFrame(sens_rows)
        stem = path.name.replace(",", "_").replace(".txt", "")
        sens_path = OUTPUT_DIR / f"depth_cut_sensitivity_{stem}.csv"
        sens_df.to_csv(sens_path, index=False)
        print(f"depth_cut_sensitivity_csv={sens_path}")
        print(sens_df.to_string(index=False))
        if args.write_experiment_json:
            write_experiment_json(
                Path(args.write_experiment_json),
                {
                    "script": "porosity_baseline.py",
                    "eval_protocol": "depth_holdout_sensitivity",
                    "path": str(path),
                    "depth_cut_grid": fracs,
                    "use_scaler": bool(args.use_scaler),
                    "rows": sens_df.to_dict(orient="records"),
                },
                repo_root=ROOT.parent,
            )
        return 0

    if args.max_depth_grid:
        depths_try = [int(x) for x in args.max_depth_grid.split(",") if x.strip()]
        depth_arr = work["Depth_m"].to_numpy(dtype=float)
        train_mask, test_mask = depth_block_split(depth_arr, args.test_fraction)
        x_train = work.loc[train_mask, feature_cols]
        y_train = work.loc[train_mask, target_col]
        x_test = work.loc[test_mask, feature_cols]
        y_test = work.loc[test_mask, target_col]
        tune_rows: list[dict[str, float | int | str]] = []
        for md in depths_try:
            pipe = build_pipeline(
                feature_cols,
                use_scaler=args.use_scaler,
                random_state=random_state,
                max_depth=int(md),
            )
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test)
            tune_rows.append(
                {
                    "file": path.name,
                    "test_fraction": float(args.test_fraction),
                    "max_depth": int(md),
                    "MAE": float(mean_absolute_error(y_test, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": float(r2_score(y_test, pred)),
                }
            )
        tune_df = pd.DataFrame(tune_rows)
        stem = path.name.replace(",", "_").replace(".txt", "")
        tune_path = OUTPUT_DIR / f"gbdt_max_depth_sensitivity_{stem}.csv"
        tune_df.to_csv(tune_path, index=False)
        print(f"gbdt_max_depth_sensitivity_csv={tune_path}")
        print(tune_df.to_string(index=False))
        if args.write_experiment_json:
            write_experiment_json(
                Path(args.write_experiment_json),
                {
                    "script": "porosity_baseline.py",
                    "eval_protocol": "max_depth_sensitivity",
                    "path": str(path),
                    "max_depth_grid": depths_try,
                    "test_fraction": float(args.test_fraction),
                    "use_scaler": bool(args.use_scaler),
                    "rows": tune_df.to_dict(orient="records"),
                },
                repo_root=ROOT.parent,
            )
        return 0

    depth = work["Depth_m"].to_numpy(dtype=float)
    train_mask, test_mask = depth_block_split(depth, args.test_fraction)
    x_train = work.loc[train_mask, feature_cols]
    y_train = work.loc[train_mask, target_col]
    x_test = work.loc[test_mask, feature_cols]
    y_test = work.loc[test_mask, target_col]

    pipe = build_pipeline(
        feature_cols,
        use_scaler=args.use_scaler,
        random_state=random_state,
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    print(f"official_default_file={OFFICIAL_TRAIN_DEFAULT.name}")
    print(f"file={path.name} schema={schema}")
    print(f"train_n={len(y_train)} test_n={len(y_test)}")
    print(f"features={feature_cols}")
    print(f"use_scaler={args.use_scaler}")
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
            "use_scaler": bool(args.use_scaler),
            "metrics_test": {"mae": float(mae), "rmse": rmse, "r2": float(r2)},
        }
        joblib.dump(payload, out)
        print(f"saved={out}")

    if args.write_experiment_json and args.eval_protocol == "depth_holdout":
        write_experiment_json(
            Path(args.write_experiment_json),
            {
                "script": "porosity_baseline.py",
                "eval_protocol": "depth_holdout",
                "path": str(path),
                "schema": schema,
                "features": feature_cols,
                "test_fraction": float(args.test_fraction),
                "use_scaler": bool(args.use_scaler),
                "train_n": int(len(y_train)),
                "test_n": int(len(y_test)),
                "metrics_test": {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)},
            },
            repo_root=ROOT.parent,
        )
        print(f"experiment_json={args.write_experiment_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
