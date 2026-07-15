"""
analyze_runs.py
---------------
Scansiona ricorsivamente la cartella `multirun` alla ricerca di tutti i file
`best_model_results.csv`, li aggrega e produce un'analisi comparativa.

Uso:
    python analyze_runs.py                        # cerca multirun/ nella directory corrente
    python analyze_runs.py --multirun C:\path\to\multirun
    python analyze_runs.py --top 10 --sort custom_score
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Parsing iperparametri dal nome della cartella run
# ---------------------------------------------------------------------------
# Esempio subdir: "1...seq_l=128...i_att=True...t_att=True...h_size=48...ovlap=0.0...schd=cosine...lr=0.007"
HPARAM_PATTERNS = {
    "folder_joint_id": (r"^(\d+)\.\.\.", lambda m: int(m.group(1))),
    "folder_seq_l":    (r"seq_l=([^\.]+)",  lambda m: _try_num(m.group(1))),
    "folder_i_att":    (r"i_att=([^\.]+)",  lambda m: m.group(1) == "True"),
    "folder_t_att":    (r"t_att=([^\.]+)",  lambda m: m.group(1) == "True"),
    "folder_h_size":   (r"h_size=([^\.]+)", lambda m: _try_num(m.group(1))),
    "folder_ovlap":    (r"ovlap=(\d+\.?\d*)",                        lambda m: _try_num(m.group(1))),
    "folder_schd":     (r"schd=([^\.]+)",                            lambda m: m.group(1)),
    "folder_lr":       (r"lr=(\d+\.?\d*(?:e[+-]?\d+)?)",            lambda m: _try_num(m.group(1))),
}


def _try_num(s: str):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_hparams_from_path(csv_path: Path) -> dict:
    """Estrae gli iperparametri dal nome della cartella run (genitore di /output/)."""
    # Struttura: multirun/DATE/TIME/SUBDIR/output/best_model_results.csv
    subdir_name = csv_path.parent.parent.name   # es. "1...seq_l=128...i_att=True..."
    time_name   = csv_path.parent.parent.parent.name   # es. "21-57-59"
    date_name   = csv_path.parent.parent.parent.parent.name  # es. "2026-05-21"

    hparams = {
        "run_date":    date_name,
        "run_time":    time_name,
        "run_subdir":  subdir_name,
    }
    for col, (pattern, conv) in HPARAM_PATTERNS.items():
        m = re.search(pattern, subdir_name)
        hparams[col] = conv(m) if m else None

    return hparams


# ---------------------------------------------------------------------------
# Lettura CSV (formato italiano: sep=';', decimal=',')
# ---------------------------------------------------------------------------

def read_csv_italian(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", decimal=",")
        return df
    except Exception as e:
        print(f"  ⚠️  Impossibile leggere {path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Raccolta di tutti i risultati
# ---------------------------------------------------------------------------

def collect_results(multirun_dir: Path) -> pd.DataFrame:
    csv_files = sorted(multirun_dir.rglob("best_model_results.csv"))
    if not csv_files:
        print(f"❌ Nessun file best_model_results.csv trovato in: {multirun_dir}")
        sys.exit(1)

    print(f"📂 Trovati {len(csv_files)} file CSV.\n")
    rows = []
    for csv_path in csv_files:
        df = read_csv_italian(csv_path)
        if df.empty:
            continue
        hparams = parse_hparams_from_path(csv_path)
        for _, row in df.iterrows():
            merged = {**hparams, **row.to_dict()}
            merged["csv_path"] = str(csv_path)
            rows.append(merged)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analisi
# ---------------------------------------------------------------------------

METRICS = ["custom_score", "MSE", "std", "eval_loss"]
HPARAM_COLS = [
    "folder_joint_id", "folder_seq_l", "folder_i_att", "folder_t_att",
    "folder_h_size", "folder_ovlap", "folder_schd", "folder_lr",
]


def numeric_cols(df: pd.DataFrame, cols):
    present = [c for c in cols if c in df.columns]
    return df[present].apply(pd.to_numeric, errors="coerce")


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze(df: pd.DataFrame, sort_by: str, top_n: int):
    # Assicuriamo i tipi numerici sulle metriche
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # ── 1. Top N run globali ────────────────────────────────────────────────
    print_section(f"TOP {top_n} RUN  (ordinati per '{sort_by}' ↑)")
    sort_col = sort_by if sort_by in df.columns else "custom_score"
    display_cols = [c for c in HPARAM_COLS + METRICS if c in df.columns]
    top = df.sort_values(sort_col, ascending=True).head(top_n)
    print(top[display_cols].to_string(index=False))

    # ── 2. Best run assoluto ────────────────────────────────────────────────
    print_section("MIGLIOR RUN ASSOLUTO")
    best = df.loc[df[sort_col].idxmin()]
    for k, v in best.items():
        if k != "csv_path":
            print(f"  {k:<25} {v}")

    # ── 3. Analisi per giunto ───────────────────────────────────────────────
    if "folder_joint_id" in df.columns:
        print_section("RIEPILOGO PER GIUNTO  (media metriche)")
        grp = df.groupby("folder_joint_id")[
            [m for m in METRICS if m in df.columns]
        ].agg(["mean", "min", "std"]).round(6)
        print(grp.to_string())

    # ── 4. Impatto iperparametri (media custom_score per valore univoco) ────
    if "custom_score" in df.columns:
        print_section("IMPATTO IPERPARAMETRI  (media custom_score per valore)")
        for hp in HPARAM_COLS:
            if hp not in df.columns:
                continue
            grp = (
                df.groupby(hp)["custom_score"]
                .agg(["mean", "min", "count"])
                .rename(columns={"mean": "avg_score", "min": "best_score", "count": "n_runs"})
                .sort_values("avg_score")
                .round(6)
            )
            print(f"\n  📌 {hp}")
            print(grp.to_string())

    # ── 5. Correlazione iperparametri ↔ custom_score ────────────────────────
    print_section("CORRELAZIONE  (iperparametri numerici ↔ custom_score)")
    num_hp = numeric_cols(df, HPARAM_COLS)
    if "custom_score" in df.columns and not num_hp.empty:
        corr = num_hp.corrwith(df["custom_score"]).dropna().sort_values()
        for col, val in corr.items():
            bar = "█" * int(abs(val) * 20)
            sign = "+" if val > 0 else "-"
            print(f"  {col:<25} {sign}{abs(val):.4f}  {bar}")

    # ── 6. Distribuzione run per data ───────────────────────────────────────
    if "run_date" in df.columns:
        print_section("RUN PER DATA")
        print(df.groupby("run_date").size().rename("n_runs").to_string())

    # ── 7. Export CSV aggregato ─────────────────────────────────────────────
    out_path = Path("all_runs_summary.csv")
    df.sort_values(sort_col, ascending=True).to_csv(
        out_path, sep=";", decimal=",", index=False
    )
    print(f"\n💾 Riepilogo completo salvato in: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analisi run grid-search multirun Hydra.")
    parser.add_argument(
        "--multirun",
        type=Path,
        default=Path("multirun"),
        help="Percorso alla cartella multirun (default: ./multirun)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Numero di top run da mostrare (default: 10)",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="custom_score",
        choices=["custom_score", "MSE", "std", "eval_loss"],
        help="Metrica da usare per ordinare (default: custom_score)",
    )
    args = parser.parse_args()

    if not args.multirun.exists():
        print(f"❌ Cartella non trovata: {args.multirun.resolve()}")
        sys.exit(1)

    print(f"🔍 Scansione: {args.multirun.resolve()}")
    df = collect_results(args.multirun)

    if df.empty:
        print("❌ Nessun dato valido trovato.")
        sys.exit(1)

    print(f"✅ {len(df)} run caricate.")
    analyze(df, sort_by=args.sort, top_n=args.top)


if __name__ == "__main__":
    main()
