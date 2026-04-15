"""
Exploratory analysis for tab-separated well log files in ../data/.
Reads all .txt files, groups them by column schema, prints summaries,
and writes figures under ../eda_output/.
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

_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "eda_output"

COLUMN_ALIASES_6LOGS = {
    "Depth(m)": "Depth_m",
    "Sonic": "Sonic",
    "Density": "Density",
    "Gamma_Ray": "Gamma_Ray",
    "P_Impedance": "P_Impedance",
    "Vp": "Vp",
    "Vc": "Vc",
}

COLUMN_ALIASES_3LOGS = {
    "Depth(m)": "Depth_m",
    "AC(us/ft)": "AC",
    "GR(API)": "GR",
    "Porosity": "Porosity",
}


def read_table(path: Path) -> pd.DataFrame:
    """Load TSV trying common encodings used by Windows-exported logs."""
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    assert last_err is not None
    raise last_err


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rename columns to ASCII identifiers; return (df, schema_key)."""
    cols = list(df.columns)
    if cols == list(COLUMN_ALIASES_6LOGS.keys()):
        return df.rename(columns=COLUMN_ALIASES_6LOGS), "6logs_Sonic_Density_GR_Zp_Vp_Vc"
    if cols == list(COLUMN_ALIASES_3LOGS.keys()):
        return df.rename(columns=COLUMN_ALIASES_3LOGS), "3logs_AC_GR_Phi"
    # 7logs: third header may be corrupted in some exports; match by position.
    if (
        len(cols) == 8
        and cols[0] == "Depth(m)"
        and cols[1] == "AC"
        and cols[3] == "AI"
        and cols[4] == "Vp"
        and cols[5] == "Porosity"
        and cols[6] == "GR"
        and cols[7] == "Vc"
    ):
        out = df.copy()
        out.columns = [
            "Depth_m",
            "AC",
            "Rho_b",
            "AI",
            "Vp",
            "Porosity",
            "GR",
            "Vc",
        ]
        return out, "7logs_AC_Rho_AI_Vp_Phi_GR_Vc"
    unknown = "|".join(cols)
    raise ValueError(f"Unknown column schema: {unknown}")


def numeric_features(df: pd.DataFrame) -> list[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Depth_m" in num:
        num.remove("Depth_m")
    return num


def plot_correlation(
    df: pd.DataFrame, title: str, outfile: Path, method: str = "pearson"
) -> None:
    feats = numeric_features(df)
    if len(feats) < 2:
        return
    corr = df[feats].corr(method=method)
    plt.figure(figsize=(max(6, len(feats) * 0.5), max(5, len(feats) * 0.45)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_histograms(df: pd.DataFrame, title: str, outfile: Path) -> None:
    feats = numeric_features(df)
    if not feats:
        return
    n = len(feats)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, feats):
        ax.hist(df[col].dropna().values, bins=40, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("")
    for j in range(len(feats), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def file_report(path: Path) -> dict:
    raw = read_table(path)
    df, schema = normalize_columns(raw)
    feats = numeric_features(df)
    desc = df[feats].describe().T if feats else pd.DataFrame()
    nulls = df[feats].isna().sum() if feats else pd.Series(dtype=int)
    depth_span = (df["Depth_m"].min(), df["Depth_m"].max())
    step = df["Depth_m"].diff().median()
    return {
        "path": path.name,
        "schema": schema,
        "rows": len(df),
        "depth_min": depth_span[0],
        "depth_max": depth_span[1],
        "depth_step_median": step,
        "columns": list(df.columns),
        "describe": desc,
        "null_counts": nulls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EDA: catalog well TSV files, summaries, correlation heatmaps, histograms."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_DEFAULT_DATA),
        help="Directory containing *.txt well log tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_DEFAULT_OUT),
        help="Directory for summary CSV and PNG outputs.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        print("No .txt files in", data_dir, file=sys.stderr)
        return 1

    print("=== Catalogo de arquivos (todos) ===\n")
    reports: list[dict] = []
    for p in paths:
        try:
            rep = file_report(p)
            reports.append(rep)
        except ValueError as exc:
            print(f"[SKIP] {p.name}: {exc}")
            continue

    # Summary table
    rows_summary = []
    for rep in reports:
        rows_summary.append(
            {
                "file": rep["path"],
                "schema": rep["schema"],
                "n_rows": rep["rows"],
                "depth_min": rep["depth_min"],
                "depth_max": rep["depth_max"],
                "dz_median": rep["depth_step_median"],
            }
        )
    summary_df = pd.DataFrame(rows_summary)
    print(summary_df.to_string(index=False))
    summary_path = out_dir / "summary_all_files.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nTabela salva: {summary_path}")

    # One detailed block per schema (first file of each)
    seen: set[str] = set()
    for rep in reports:
        sk = rep["schema"]
        if sk in seen:
            continue
        seen.add(sk)
        print(f"\n=== Exemplo de schema: {sk} ===")
        print(f"Arquivo exemplo: {rep['path']}")
        print(f"Linhas: {rep['rows']}, profundidade: {rep['depth_min']:.2f} a {rep['depth_max']:.2f} m")
        print(f"Passo mediano (Depth): {rep['depth_step_median']:.6f} m")
        if not rep["null_counts"].empty and rep["null_counts"].sum() > 0:
            print("Valores nulos por coluna:")
            print(rep["null_counts"][rep["null_counts"] > 0].to_string())
        print("\ndescribe():")
        print(rep["describe"].to_string())

    # Figures: Pearson and Spearman correlation maps + histograms per file
    for rep in reports:
        p = data_dir / rep["path"]
        raw = read_table(p)
        df, schema = normalize_columns(raw)
        safe_name = rep["path"].replace(",", "_").replace(" ", "_")
        for method, tag in (("pearson", "pearson"), ("spearman", "spearman")):
            plot_correlation(
                df,
                f"{rep['path']} | {schema} | {tag}",
                out_dir / f"corr_{safe_name}_{tag}.png",
                method=method,
            )
        plot_histograms(
            df,
            f"{rep['path']} | {schema}",
            out_dir / f"hist_{safe_name}.png",
        )

    print(f"\nFiguras PNG em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
