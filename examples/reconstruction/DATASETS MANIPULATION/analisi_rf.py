"""
analisi_rf.py
=============
Analisi statistica del classificatore Random Forest a partire dal file
unified_anomaly_log.csv prodotto da adV9999.py.

Uso:
    python analisi_rf.py                              # cerca il CSV nella directory corrente
    python analisi_rf.py --csv /path/to/file.csv      # percorso esplicito
    python analisi_rf.py --csv file.csv --out report  # salva output in cartella "report"

Output:
    - Stampa a terminale di tutte le analisi
    - (opzionale) Salvataggio delle tabelle come CSV nella cartella --out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

# Mapping: stringa del dataset → giunto perturbato
# Adatta questi valori se i nomi dei dataset nel tuo CSV sono diversi.
DATASET_TO_JOINT: dict[str, int] = {
    "QUERY_CSV_j2test_EXCEL":   2,
    "QUERY_CSV_j5test_EXCEL":   5,
    "QUERY_CSV_j2_3test_EXCEL": 2,
    "QUERY_CSV_j5_3test_EXCEL": 5,
}

# Mapping: stringa del dataset → etichetta attesa (ground truth)
DATASET_TO_GT: dict[str, str] = {
    "QUERY_CSV_j2test_EXCEL":   "backlash",
    "QUERY_CSV_j5test_EXCEL":   "friction",
    "QUERY_CSV_j2_3test_EXCEL": "noise",
    "QUERY_CSV_j5_3test_EXCEL": "noise",
}

# Mapping: stringa del dataset → nome leggibile per la stampa
DATASET_LABEL: dict[str, str] = {
    "QUERY_CSV_j2test_EXCEL":   "j2_test  (backlash)",
    "QUERY_CSV_j5test_EXCEL":   "j5_test  (friction)",
    "QUERY_CSV_j2_3test_EXCEL": "j2_3_test (noise)",
    "QUERY_CSV_j5_3test_EXCEL": "j5_3_test (noise)",
}

# Classi RF attese (usate per costruire le colonne della tabella probabilità)
CLASSES = ["backlash", "friction", "noise"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def sep(char: str = "─", n: int = 65) -> str:
    return char * n


def header(title: str) -> None:
    print(f"\n{sep('═')}")
    print(f"  {title}")
    print(sep('═'))


def subheader(title: str) -> None:
    print(f"\n{sep()}")
    print(f"  {title}")
    print(sep())


def parse_prob_string(s: str) -> dict[str, float]:
    """
    Converte la stringa 'backlash: 0.96, friction: 0.02, noise: 0.02'
    in un dizionario {'backlash': 0.96, 'friction': 0.02, 'noise': 0.02}.
    Restituisce {} se la stringa è vuota o non parsabile.
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        cls, prob = part.rsplit(":", 1)
        try:
            out[cls.strip()] = float(prob.strip())
        except ValueError:
            pass
    return out


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carica il CSV, filtra i test set, restringe al giunto perturbato e
    arricchisce il DataFrame con colonne di analisi.

    Restituisce:
        df_filt   : tutte le righe dei test set, ristrette al giunto perturbato
        df_anomale: solo le finestre con esito 'saltata' (presentate al RF)
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # Filtra solo i dataset di test configurati
    df_test = df[df["dataset"].isin(DATASET_TO_JOINT)].copy()
    if df_test.empty:
        print("[ERRORE] Nessun dataset di test trovato nel CSV.")
        print(f"  Dataset presenti: {df['dataset'].unique().tolist()}")
        print(f"  Dataset attesi:   {list(DATASET_TO_JOINT.keys())}")
        sys.exit(1)

    # Aggiunge colonne di supporto
    df_test["giunto_target"] = df_test["dataset"].map(DATASET_TO_JOINT)
    df_test["label_attesa"]  = df_test["dataset"].map(DATASET_TO_GT)

    # Restringe al giunto perturbato
    df_filt = df_test[df_test["joint_id"] == df_test["giunto_target"]].copy()

    # Subset: finestre presentate al RF (esito = saltata)
    df_anomale = df_filt[df_filt["esito"] == "saltata"].copy()

    # Parsing probabilità RF
    df_anomale["probs"]      = df_anomale["probabilita_rf"].apply(parse_prob_string)
    df_anomale["pred_class"] = df_anomale["probs"].apply(
        lambda d: max(d, key=d.get) if d else None
    )
    df_anomale["pred_prob"]  = df_anomale["probs"].apply(
        lambda d: max(d.values()) if d else None
    )
    df_anomale["correct"]    = df_anomale["pred_class"] == df_anomale["label_attesa"]

    # Z-score rispetto alla soglia dinamica
    df_anomale["zscore"] = (
        (df_anomale["mse"] - df_anomale["mu_training"]) / df_anomale["sigma_training"]
    )

    return df_filt, df_anomale


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 1 — CORPUS DI TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def analisi_training(csv_path: Path, out_dir: Path | None) -> None:
    """Conta le finestre etichettate nei train set."""
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df_train = df[df["esito"] == "confermata"]

    header("CORPUS DI TRAINING")
    table = (
        df_train["label"]
        .value_counts()
        .rename_axis("Classe")
        .reset_index(name="Finestre di training")
    )
    total = table["Finestre di training"].sum()
    totale_row = pd.DataFrame([{"Classe": "TOTALE", "Finestre di training": total}])
    table = pd.concat([table, totale_row], ignore_index=True)
    print(table.to_string(index=False))

    if out_dir:
        table.to_csv(out_dir / "01_corpus_training.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 2 — OVERVIEW ESITI PER DATASET (giunto perturbato)
# ─────────────────────────────────────────────────────────────────────────────

def analisi_overview(df_filt: pd.DataFrame, out_dir: Path | None) -> None:
    """Distribuzione degli esiti (normale, FP_noto, saltata) per dataset."""
    header("OVERVIEW ESITI PER DATASET (giunto perturbato)")
    rows = []
    for ds, grp in df_filt.groupby("dataset"):
        r = {"Dataset": DATASET_LABEL.get(ds, ds)}
        for esito in ["normale", "FP_noto", "saltata"]:
            r[esito] = (grp["esito"] == esito).sum()
        r["Totale finestre"] = len(grp)
        rows.append(r)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if out_dir:
        table.to_csv(out_dir / "02_overview_esiti.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 3 — ACCURATEZZA PER DATASET
# ─────────────────────────────────────────────────────────────────────────────

def analisi_accuracy(df_anomale: pd.DataFrame, out_dir: Path | None) -> None:
    """Accuracy, conteggi e statistiche di confidenza per dataset."""
    header("ACCURATEZZA CLASSIFICATORE RF PER DATASET")
    rows = []
    for ds, grp in df_anomale.groupby("dataset"):
        n     = len(grp)
        ok    = grp["correct"].sum()
        acc   = ok / n * 100 if n > 0 else float("nan")
        p_mean = grp["pred_prob"].mean()
        p_min  = grp["pred_prob"].min()
        p_max  = grp["pred_prob"].max()
        rows.append({
            "Dataset":         DATASET_LABEL.get(ds, ds),
            "GT":              DATASET_TO_GT.get(ds, "?"),
            "Finestre":        n,
            "Corrette":        f"{ok}/{n}",
            "Accuracy":        f"{acc:.1f}%",
            "P̄(GT) media":    f"{p_mean:.3f}",
            "P̄(GT) min":      f"{p_min:.3f}",
            "P̄(GT) max":      f"{p_max:.3f}",
        })

    # Riga totale
    n_tot = len(df_anomale)
    ok_tot = df_anomale["correct"].sum()
    rows.append({
        "Dataset":      "TOTALE",
        "GT":           "—",
        "Finestre":     n_tot,
        "Corrette":     f"{ok_tot}/{n_tot}",
        "Accuracy":     f"{ok_tot/n_tot*100:.1f}%" if n_tot > 0 else "—",
        "P̄(GT) media": "—",
        "P̄(GT) min":   "—",
        "P̄(GT) max":   "—",
    })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if out_dir:
        table.to_csv(out_dir / "03_accuracy_per_dataset.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 4 — DISTRIBUZIONE PROBABILITÀ PER CLASSE
# ─────────────────────────────────────────────────────────────────────────────

def analisi_prob_per_classe(df_anomale: pd.DataFrame, out_dir: Path | None) -> None:
    """Probabilità media ± std per ciascuna classe, per dataset."""
    header("DISTRIBUZIONE PROBABILITÀ PER CLASSE (media ± std)")
    rows = []
    for ds, grp in df_anomale.groupby("dataset"):
        for cls in CLASSES:
            vals = grp["probs"].apply(lambda d: d.get(cls, 0.0))
            rows.append({
                "Dataset": DATASET_LABEL.get(ds, ds),
                "GT":      DATASET_TO_GT.get(ds, "?"),
                "Classe":  cls,
                "Media":   round(vals.mean(), 3),
                "Std":     round(vals.std(), 3),
                "Min":     round(vals.min(), 3),
                "Max":     round(vals.max(), 3),
            })

    table = pd.DataFrame(rows)
    # Stampa raggruppata per dataset
    for ds in df_anomale["dataset"].unique():
        label = DATASET_LABEL.get(ds, ds)
        gt    = DATASET_TO_GT.get(ds, "?")
        sub   = table[table["Dataset"] == label][["Classe","Media","Std","Min","Max"]]
        print(f"\n  {label}  [GT: {gt}]")
        print(sub.to_string(index=False))

    if out_dir:
        table.to_csv(out_dir / "04_prob_per_classe.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 5 — MSE e Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def analisi_mse_zscore(df_anomale: pd.DataFrame, out_dir: Path | None) -> None:
    """Statistiche di MSE e Z-score per dataset."""
    header("MSE e Z-SCORE DELLE FINESTRE ANOMALE (giunto perturbato)")
    rows = []
    for ds, grp in df_anomale.groupby("dataset"):
        rows.append({
            "Dataset":        DATASET_LABEL.get(ds, ds),
            "n finestre":     len(grp),
            "MSE medio":      f"{grp['mse'].mean():.6f}",
            "MSE min":        f"{grp['mse'].min():.6f}",
            "MSE max":        f"{grp['mse'].max():.6f}",
            "Z-score medio":  f"{grp['zscore'].mean():.2f}",
            "Z-score mediano":f"{grp['zscore'].median():.2f}",
            "Z-score max":    f"{grp['zscore'].max():.2f}",
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if out_dir:
        table.to_csv(out_dir / "05_mse_zscore.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 6 — DETECTION RATE A LIVELLO DI TRAIETTORIA
# ─────────────────────────────────────────────────────────────────────────────

def analisi_traiettorie(
    df_filt: pd.DataFrame, df_anomale: pd.DataFrame, out_dir: Path | None
) -> None:
    """Detection Rate a livello di traiettoria."""
    header("DETECTION RATE A LIVELLO DI TRAIETTORIA")
    rows = []
    for ds in DATASET_TO_JOINT:
        grp_all = df_filt[df_filt["dataset"] == ds]
        if grp_all.empty:
            continue
        grp_rf = df_anomale[df_anomale["dataset"] == ds]

        traj_tot     = grp_all["traj_id"].nunique()
        traj_fp      = grp_all[grp_all["esito"] == "FP_noto"]["traj_id"].nunique()
        traj_allert  = grp_all[grp_all["esito"] == "saltata"]["traj_id"].nunique()
        traj_correct = grp_rf[grp_rf["correct"]]["traj_id"].nunique()
        dr = traj_correct / traj_tot * 100 if traj_tot > 0 else float("nan")

        rows.append({
            "Dataset":                    DATASET_LABEL.get(ds, ds),
            "GT":                         DATASET_TO_GT.get(ds, "?"),
            "Traj. totali":               traj_tot,
            "Traj. FP_noto (FAISS)":      traj_fp,
            "Traj. allertate":            traj_allert,
            "Traj. classif. corrett.":    traj_correct,
            "DR classificazione":         f"{dr:.1f}%",
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if out_dir:
        table.to_csv(out_dir / "06_detection_rate_traiettorie.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ANALISI 7 — DETTAGLIO FINESTRA PER FINESTRA
# ─────────────────────────────────────────────────────────────────────────────

def analisi_dettaglio_finestre(df_anomale: pd.DataFrame, out_dir: Path | None) -> None:
    """Stampa ogni finestra anomala con probabilità complete e flag corretto."""
    header("DETTAGLIO FINESTRA PER FINESTRA")
    all_rows = []
    for ds, grp in df_anomale.groupby("dataset"):
        label = DATASET_LABEL.get(ds, ds)
        gt    = DATASET_TO_GT.get(ds, "?")
        subheader(f"{label}  [GT: {gt}]")

        for _, row in grp.iterrows():
            probs_sorted = sorted(row["probs"].items(), key=lambda x: -x[1])
            probs_str    = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs_sorted)
            flag         = "✓" if row["correct"] else "✗"
            print(
                f"  traj={int(row['traj_id']):>4}  win={int(row['win_idx']):>2}  "
                f"mse={row['mse']:.6f}  z={row['zscore']:.2f}  "
                f"→  {probs_str}  [{flag}]"
            )

            # Per il CSV di output
            r = {
                "dataset":     ds,
                "traj_id":     int(row["traj_id"]),
                "win_idx":     int(row["win_idx"]),
                "mse":         row["mse"],
                "zscore":      round(row["zscore"], 3),
                "pred_class":  row["pred_class"],
                "pred_prob":   round(row["pred_prob"], 3) if row["pred_prob"] else None,
                "label_attesa":row["label_attesa"],
                "correct":     row["correct"],
            }
            for cls in CLASSES:
                r[f"p_{cls}"] = round(row["probs"].get(cls, 0.0), 3)
            all_rows.append(r)

    if out_dir and all_rows:
        pd.DataFrame(all_rows).to_csv(
            out_dir / "07_dettaglio_finestre.csv", index=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisi statistica del classificatore RF da unified_anomaly_log.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("unified_anomaly_log.csv"),
        help="Percorso del file CSV (default: ./unified_anomaly_log.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Cartella di output per le tabelle CSV (opzionale)",
    )
    args = parser.parse_args()

    # Verifica esistenza CSV
    if not args.csv.exists():
        print(f"[ERRORE] File non trovato: {args.csv}")
        sys.exit(1)

    # Crea cartella output se richiesta
    out_dir: Path | None = None
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out_dir = args.out
        print(f"Output CSV salvato in: {out_dir.resolve()}")

    print(f"\nCaricamento: {args.csv.resolve()}")

    # Caricamento e preparazione dati
    df_filt, df_anomale = load_and_prepare(args.csv)

    # ── Esecuzione analisi ────────────────────────────────────────────────────
    analisi_training(args.csv, out_dir)
    analisi_overview(df_filt, out_dir)
    analisi_accuracy(df_anomale, out_dir)
    analisi_prob_per_classe(df_anomale, out_dir)
    analisi_mse_zscore(df_anomale, out_dir)
    analisi_traiettorie(df_filt, df_anomale, out_dir)
    analisi_dettaglio_finestre(df_anomale, out_dir)

    print(f"\n{sep('═')}")
    print("  Analisi completata.")
    print(sep('═'))


if __name__ == "__main__":
    main()
