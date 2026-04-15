"""
GR vs Vc dependence under ranks and ties: diagnostics for 7logs tables.

Writes CSV summaries and PNG figures under ML-prediction/outputs/ by default.
Uses only numpy, pandas, scipy, sklearn, matplotlib (same stack as other scripts).
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
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.linear_model import LinearRegression

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exploratory_analysis import normalize_columns, read_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

RANK_METHODS = ("average", "min", "max")


def tie_diagnostics(name: str, s: pd.Series) -> dict[str, float | int]:
    """Counts unique values and rows involved in ties (duplicate values)."""
    v = s.dropna().to_numpy(dtype=float)
    n = int(v.size)
    if n == 0:
        return {"column": name, "n": 0, "n_unique": 0, "tie_row_fraction": 0.0}
    uniq, counts = np.unique(v, return_counts=True)
    n_unique = int(uniq.size)
    tied_mask = counts > 1
    rows_in_ties = int(counts[tied_mask].sum()) if np.any(tied_mask) else 0
    return {
        "column": name,
        "n": n,
        "n_unique": n_unique,
        "tie_row_fraction": float(rows_in_ties / n) if n else 0.0,
        "max_multiplicity": int(counts.max()) if n else 0,
    }


def pearson_on_ranks(
    a: np.ndarray, b: np.ndarray, method_a: str, method_b: str
) -> float:
    """Pearson correlation between rank transforms (Spearman tie variants)."""
    ra = rankdata(a, method=method_a)
    rb = rankdata(b, method=method_b)
    if np.std(ra) < 1e-15 or np.std(rb) < 1e-15:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def linear_residuals(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """OLS residuals of y on intercept + z (z shape (n, k))."""
    n = y.shape[0]
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    x_design = np.column_stack([np.ones(n, dtype=float), z.astype(float)])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(x_design, y)
    pred = lr.predict(x_design)
    return y - pred


def partial_pearson_residual(
    a: np.ndarray, b: np.ndarray, controls: np.ndarray
) -> float:
    """Partial correlation via correlation of OLS residuals (linear controls)."""
    ea = linear_residuals(a, controls)
    eb = linear_residuals(b, controls)
    if np.std(ea) < 1e-15 or np.std(eb) < 1e-15:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


def jittered_spearman_stability(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
    noise_scale: float,
) -> tuple[float, float]:
    """Break ties by tiny uniform jitter; Spearman mean and std over draws."""
    if n_draws <= 0:
        return float("nan"), float("nan")
    vals: list[float] = []
    for _ in range(n_draws):
        ja = a + rng.uniform(-noise_scale, noise_scale, size=a.shape)
        jb = b + rng.uniform(-noise_scale, noise_scale, size=b.shape)
        r, _ = spearmanr(ja, jb, nan_policy="omit")
        vals.append(float(r))
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GR vs Vc rank/tie dependence (7logs)."
    )
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
        help="Output directory (default: outputs/gr_vc_rank_<stem>).",
    )
    parser.add_argument(
        "--jitter-draws",
        type=int,
        default=80,
        help="Monte Carlo draws for tie jitter robustness (0 skips).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="RNG seed for jitter draws.",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()
    stem = path.name.replace(",", "_").replace(".txt", "")
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (ROOT / "outputs" / f"gr_vc_rank_{stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(path)
    df, schema = normalize_columns(raw)
    if schema != "7logs_AC_Rho_AI_Vp_Phi_GR_Vc":
        print("Este script espera esquema 7logs com GR e Vc.", file=sys.stderr)
        return 2

    cols_needed = ["Depth_m", "GR", "Vc", "AC", "Rho_b", "AI", "Vp"]
    aligned = df.loc[df[cols_needed].notna().all(axis=1), cols_needed].reset_index(
        drop=True
    )
    depth = aligned["Depth_m"].to_numpy(dtype=float)
    gr = aligned["GR"].to_numpy(dtype=float)
    vc = aligned["Vc"].to_numpy(dtype=float)
    work = aligned[["Depth_m", "GR", "Vc"]]

    tie_rows = [tie_diagnostics("GR", work["GR"]), tie_diagnostics("Vc", work["Vc"])]
    pd.DataFrame(tie_rows).to_csv(out_dir / "tie_diagnostics.csv", index=False)

    r_pearson = float(np.corrcoef(gr, vc)[0, 1])
    sp, sp_p = spearmanr(gr, vc, nan_policy="omit")
    kt, kt_p = kendalltau(gr, vc, nan_policy="omit")

    rows_sens: list[dict[str, str | float]] = []
    for ma in RANK_METHODS:
        for mb in RANK_METHODS:
            rows_sens.append(
                {
                    "rank_GR": ma,
                    "rank_Vc": mb,
                    "pearson_on_ranks": pearson_on_ranks(gr, vc, ma, mb),
                }
            )
    sens_df = pd.DataFrame(rows_sens)
    sens_df.to_csv(out_dir / "rank_tie_sensitivity.csv", index=False)

    # Linear partial correlations (raw and rank-average), controlling depth only.
    z_depth = depth.reshape(-1, 1)
    partial_raw_depth = partial_pearson_residual(gr, vc, z_depth)
    rgr = rankdata(gr, method="average")
    rvc = rankdata(vc, method="average")
    rdep = rankdata(depth, method="average")
    partial_rank_depth = partial_pearson_residual(
        rgr.astype(float), rvc.astype(float), rdep.reshape(-1, 1).astype(float)
    )

    # Control for elastic block (same rows): remove shared trend with AC, Vp, etc.
    elastic = aligned[["AC", "Rho_b", "AI", "Vp"]].to_numpy(dtype=float)
    partial_raw_elastic = partial_pearson_residual(gr, vc, elastic)
    rac = rankdata(aligned["AC"].to_numpy(dtype=float), method="average")
    rrb = rankdata(aligned["Rho_b"].to_numpy(dtype=float), method="average")
    rai = rankdata(aligned["AI"].to_numpy(dtype=float), method="average")
    rvp = rankdata(aligned["Vp"].to_numpy(dtype=float), method="average")
    z_el_rank = np.column_stack(
        [rdep, rac.astype(float), rrb.astype(float), rai.astype(float), rvp.astype(float)]
    )
    partial_rank_elastic = partial_pearson_residual(
        rgr.astype(float), rvc.astype(float), z_el_rank
    )

    rng = np.random.default_rng(args.random_state)
    scale = float(
        max(np.ptp(gr), np.ptp(vc), 1.0) * 1e-9
    )  # tiny relative to signal range
    j_mean, j_std = jittered_spearman_stability(
        gr, vc, rng, args.jitter_draws, scale
    )

    summary = pd.DataFrame(
        [
            {
                "file": path.name,
                "n": len(aligned),
                "pearson_GR_Vc": r_pearson,
                "spearman_GR_Vc": float(sp),
                "spearman_pvalue": float(sp_p),
                "kendall_tau": float(kt),
                "kendall_pvalue": float(kt_p),
                "partial_pearson_GR_Vc_given_Depth": partial_raw_depth,
                "partial_pearson_on_avg_ranks_given_rank_Depth": partial_rank_depth,
                "partial_pearson_GR_Vc_given_elastic_raw": partial_raw_elastic,
                "partial_pearson_on_avg_ranks_given_rank_Depth_elastic_ranks": partial_rank_elastic,
                "jitter_spearman_mean": j_mean,
                "jitter_spearman_std": j_std,
                "jitter_draws": int(args.jitter_draws),
                "jitter_scale": scale,
            }
        ]
    )
    summary.to_csv(out_dir / "summary_correlations.csv", index=False)

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hexbin(gr, vc, gridsize=45, cmap="viridis", mincnt=1)
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("Vc")
    axes[0].set_title(
        f"GR vs Vc (hexbin) | Spearman={float(sp):.3f}, Kendall={float(kt):.3f}"
    )
    ra = rankdata(gr, method="average")
    rb = rankdata(vc, method="average")
    axes[1].scatter(ra, rb, s=4, alpha=0.35, c=depth, cmap="coolwarm")
    cb = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cb.set_label("Depth_m")
    axes[1].set_xlabel("rank(GR), average ties")
    axes[1].set_ylabel("rank(Vc), average ties")
    axes[1].set_title("Ranks (average method), colored by depth")
    plt.tight_layout()
    fig.savefig(out_dir / "gr_vc_ranks_scatter.png", dpi=150)
    plt.close(fig)

    # Heatmap rank-method sensitivity (annotate each cell so 0.999... is visible)
    pivot = sens_df.pivot(index="rank_GR", columns="rank_Vc", values="pearson_on_ranks")
    arr = pivot.to_numpy(dtype=float)
    fig_h, ax_h = plt.subplots(figsize=(6.2, 5.0))
    im = ax_h.imshow(arr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="Pearson(ranks)")
    ax_h.set_xticks(range(len(pivot.columns)))
    ax_h.set_xticklabels(list(pivot.columns))
    ax_h.set_yticks(range(len(pivot.index)))
    ax_h.set_yticklabels(list(pivot.index))
    ax_h.set_xlabel("rankdata method for Vc")
    ax_h.set_ylabel("rankdata method for GR")
    ax_h.set_title("Spearman tie variants (Pearson on ranked values)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = float(arr[i, j])
            txt = f"{val:.6f}"
            use_white = abs(val) >= 0.85
            ax_h.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if use_white else "0.15",
            )
    fig_h.tight_layout()
    fig_h.savefig(out_dir / "rank_method_sensitivity.png", dpi=150)
    plt.close(fig_h)

    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
