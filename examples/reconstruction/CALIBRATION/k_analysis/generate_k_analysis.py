"""
generate_k_analysis.py
----------------------
Genera tutte le tabelle (CSV) e i grafici (PNG) per il capitolo sulla
selezione del parametro k della soglia di anomaly detection.

Input
-----
Un CSV con le colonne:
  Joint_ID          : identificatore numerico del giunto (es. 1..6)
  id                : id della traiettoria
  finestra          : indice della finestra dentro la traiettoria
  mse               : MSE della finestra sul validation set
  AVG MSE TRAINING  : mu calcolato sul training set per quel giunto
  AVG STD TRAINING  : sigma calcolato sul training set per quel giunto

Uso
---
  python generate_k_analysis.py --input mse_data.csv [--output ./output] [--fpr-target 1.0]

Output (nella cartella --output)
---------------------------------
  figure_01_distribuzione_mse.png        : istogrammi MSE per giunto
  figure_02_gaussianita_qqplot.png       : QQ-plot per giunto
  figure_03_fpr_vs_k.png                 : curva FPR vs k (tutti i giunti)
  figure_04_finestre_salvate_agg.png     : finestre salvate vs ancora flaggate (aggregato)
  figure_05_finestre_salvate_per_giunto.png : finestre salvate per ogni giunto
  figure_06_densita_intervalli.png       : densità finestre border-line per intervallo
  figure_07_k_ottimali.png               : k ottimale per giunto (bar chart)

  tabella_01_gaussianita.csv             : rapporto p99 empirico / gaussiano
  tabella_02_fpr_per_k_e_giunto.csv      : FPR % per ogni k e giunto
  tabella_03_finestre_salvate_agg.csv    : finestre salvate aggregate
  tabella_04_finestre_salvate_per_giunto.csv : finestre salvate per giunto
  tabella_05_k_ottimali.csv              : k ottimale per-giunto
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats


# ── Costanti di default ────────────────────────────────────────────────────────
KS_COARSE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
KS_FINE_STEP = 0.1          # granularità per k ottimale
FPR_TARGET_DEFAULT = 1.0    # % — soglia FPR per k ottimale

# Palette coerente per i giunti (colorblind-friendly)
JOINT_COLORS = ['#2166ac', '#4dac26', '#d01c8b', '#f1a340', '#998ec3', '#ca0020']
STEP_LABELS  = ['k=1.0→1.5', 'k=1.5→2.0', 'k=2.0→2.5',
                'k=2.5→3.0', 'k=3.0→3.5', 'k=3.5→4.0']

# Stile globale
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      200,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linewidth':   0.5,
})


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento e validazione dati
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal=',')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    required = {'Joint_ID', 'mse', 'AVG MSE TRAINING', 'AVG STD TRAINING'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERRORE] Colonne mancanti nel CSV: {missing}")

    df['Joint_ID'] = df['Joint_ID'].astype(int)
    df['mse'] = pd.to_numeric(df['mse'], errors='coerce')
    df['AVG MSE TRAINING'] = pd.to_numeric(df['AVG MSE TRAINING'], errors='coerce')
    df['AVG STD TRAINING'] = pd.to_numeric(df['AVG STD TRAINING'], errors='coerce')
    df = df.dropna(subset=['mse', 'AVG MSE TRAINING', 'AVG STD TRAINING'])
    return df


def joint_params(df: pd.DataFrame) -> dict:
    """Restituisce {joint_id: {'mu': ..., 'sigma': ..., 'mse': array, 'n': int}}"""
    out = {}
    for jid in sorted(df['Joint_ID'].unique()):
        sub = df[df['Joint_ID'] == jid]
        out[jid] = {
            'mu':    float(sub['AVG MSE TRAINING'].iloc[0]),
            'sigma': float(sub['AVG STD TRAINING'].iloc[0]),
            'mse':   sub['mse'].values.copy(),
            'n':     len(sub),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Calcoli statistici
# ══════════════════════════════════════════════════════════════════════════════

def compute_fpr_table(params: dict, ks=KS_COARSE) -> pd.DataFrame:
    rows = []
    for jid, p in params.items():
        for k in ks:
            thr = p['mu'] + k * p['sigma']
            fpr = float(np.sum(p['mse'] > thr)) / p['n'] * 100
            rows.append({'Joint_ID': jid, 'k': k, 'FPR_%': round(fpr, 2),
                         'threshold': round(thr, 7)})
    return pd.DataFrame(rows)


def compute_gaussianity(params: dict) -> pd.DataFrame:
    rows = []
    for jid, p in params.items():
        mu_emp   = float(np.mean(p['mse']))
        sig_emp  = float(np.std(p['mse']))
        p99_emp  = float(np.percentile(p['mse'], 99))
        p99_gauss = mu_emp + 2.326 * sig_emp
        rows.append({
            'Joint_ID':     jid,
            'mu_test':      round(mu_emp, 7),
            'sigma_test':   round(sig_emp, 7),
            'mu_train':     round(p['mu'], 7),
            'sigma_train':  round(p['sigma'], 7),
            'p99_empirico': round(p99_emp, 7),
            'p99_gaussiano':round(p99_gauss, 7),
            'rapporto':     round(p99_emp / p99_gauss, 3),
        })
    return pd.DataFrame(rows)


def compute_saved_windows(params: dict, ks=KS_COARSE) -> tuple:
    """
    Restituisce:
      df_agg  : aggregato su tutti i giunti
      df_each : per ogni giunto
    """
    agg_rows = []
    each_rows = []
    n_tot = sum(p['n'] for p in params.values())

    for ki in range(1, len(ks)):
        k0, k1 = ks[ki - 1], ks[ki]
        saved_tot = 0
        flag_tot  = 0

        for jid, p in params.items():
            thr0  = p['mu'] + k0 * p['sigma']
            thr1  = p['mu'] + k1 * p['sigma']
            flag0 = int(np.sum(p['mse'] > thr0))
            flag1 = int(np.sum(p['mse'] > thr1))
            saved = flag0 - flag1
            saved_tot += saved
            flag_tot  += flag1

            each_rows.append({
                'Joint_ID':    jid,
                'step':        f'k={k0}→{k1}',
                'k0': k0, 'k1': k1,
                'saved':       saved,
                'saved_%':     round(saved / p['n'] * 100, 1),
                'flagged':     flag1,
                'flagged_%':   round(flag1 / p['n'] * 100, 1),
            })

        agg_rows.append({
            'step':       f'k={k0}→{k1}',
            'k0': k0, 'k1': k1,
            'saved':      saved_tot,
            'saved_%':    round(saved_tot / n_tot * 100, 1),
            'flagged':    flag_tot,
            'flagged_%':  round(flag_tot / n_tot * 100, 1),
        })

    return pd.DataFrame(agg_rows), pd.DataFrame(each_rows)


def compute_k_optimal(params: dict, fpr_target=FPR_TARGET_DEFAULT) -> pd.DataFrame:
    ks_fine = np.arange(1.0, 6.01, KS_FINE_STEP)
    rows = []
    for jid, p in params.items():
        k_opt = fpr_opt = thr_opt = n_flag = None
        for k in ks_fine:
            thr = p['mu'] + k * p['sigma']
            fpr = float(np.sum(p['mse'] > thr)) / p['n'] * 100
            if fpr <= fpr_target:
                k_opt  = round(float(k), 1)
                fpr_opt = round(fpr, 2)
                thr_opt = round(thr, 6)
                n_flag  = int(np.sum(p['mse'] > thr))
                break
        rows.append({
            'Joint_ID':   jid,
            'k_ottimale': k_opt if k_opt else '>6.0',
            'soglia':     thr_opt,
            'FPR_%':      fpr_opt,
            'n_flaggate': n_flag,
            'n_totale':   p['n'],
            'delta_vs_k3': round(k_opt - 3.0, 1) if k_opt else None,
        })
    return pd.DataFrame(rows)


def compute_border_density(params: dict, ks=KS_COARSE) -> pd.DataFrame:
    """Finestre nell'intervallo [thr(k), thr(k+0.5)] per ogni giunto."""
    rows = []
    for ki in range(len(ks) - 1):
        k0, k1 = ks[ki], ks[ki + 1]
        for jid, p in params.items():
            t0 = p['mu'] + k0 * p['sigma']
            t1 = p['mu'] + k1 * p['sigma']
            count = int(np.sum((p['mse'] > t0) & (p['mse'] <= t1)))
            rows.append({
                'Joint_ID': jid,
                'step':     f'k={k0}→{k1}',
                'k0': k0, 'k1': k1,
                'count':    count,
                'count_%':  round(count / p['n'] * 100, 1),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def fig_distribuzione_mse(params: dict, out_dir: Path):
    joints = sorted(params.keys())
    n_j = len(joints)
    ncols = 3
    nrows = int(np.ceil(n_j / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, jid in enumerate(joints):
        ax = axes[idx]
        p  = params[jid]
        mse = p['mse']
        mu_tr, sig_tr = p['mu'], p['sigma']

        # Istogramma
        ax.hist(mse, bins=35, color=JOINT_COLORS[idx % len(JOINT_COLORS)],
                alpha=0.65, edgecolor='white', linewidth=0.4, density=False,
                label='Finestre normali')

        # Gaussiana teorica sovrapposta (scalata al conteggio)
        x = np.linspace(mse.min(), mse.max(), 300)
        n_bins = 35
        bin_w = (mse.max() - mse.min()) / n_bins
        gauss = stats.norm.pdf(x, np.mean(mse), np.std(mse)) * len(mse) * bin_w
        ax.plot(x, gauss, color='#333333', linewidth=1.2, linestyle='--',
                label='Gaussiana teorica')

        # Soglie k=2 e k=3
        for k_ref, ls, lbl in [(2.0, ':', 'k=2'), (3.0, '-', 'k=3')]:
            thr = mu_tr + k_ref * sig_tr
            ax.axvline(thr, color='#c0392b', linewidth=1.1, linestyle=ls,
                       label=f'Soglia {lbl}')

        ax.set_title(f'Giunto {jid}')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Finestre')
        ax.legend(fontsize=7, framealpha=0.7)

    for ax in axes[n_j:]:
        ax.set_visible(False)

    fig.suptitle('Distribuzione MSE sul validation set (dati normali)', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    path = out_dir / 'figure_01_distribuzione_mse.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_qqplot(params: dict, out_dir: Path):
    joints = sorted(params.keys())
    n_j = len(joints)
    ncols = 3
    nrows = int(np.ceil(n_j / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, jid in enumerate(joints):
        ax = axes[idx]
        p  = params[jid]
        mse = p['mse']

        (osm, osr), (slope, intercept, _) = stats.probplot(mse, dist='norm')
        ax.scatter(osm, osr, s=8, alpha=0.5,
                   color=JOINT_COLORS[idx % len(JOINT_COLORS)], label='Dati')
        # Linea teorica gaussiana
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, slope * x_line + intercept, color='#333333',
                linewidth=1.2, linestyle='--', label='Gaussiana teorica')

        p99_emp   = np.percentile(mse, 99)
        p99_gauss = np.mean(mse) + 2.326 * np.std(mse)
        ratio     = p99_emp / p99_gauss

        ax.set_title(f'Giunto {jid}  (p99 ratio={ratio:.2f})')
        ax.set_xlabel('Quantili teorici')
        ax.set_ylabel('Quantili empirici')
        ax.legend(fontsize=7)

    for ax in axes[n_j:]:
        ax.set_visible(False)

    fig.suptitle('QQ-plot: MSE vs distribuzione gaussiana', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    path = out_dir / 'figure_02_gaussianita_qqplot.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_fpr_vs_k(params: dict, fpr_df: pd.DataFrame, out_dir: Path):
    joints = sorted(params.keys())
    ks     = sorted(fpr_df['k'].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, jid in enumerate(joints):
        sub = fpr_df[fpr_df['Joint_ID'] == jid].sort_values('k')
        ax.plot(sub['k'], sub['FPR_%'],
                marker='o', markersize=5, linewidth=1.8,
                color=JOINT_COLORS[idx % len(JOINT_COLORS)],
                label=f'Giunto {jid}')

    # Soglia FPR 1%
    ax.axhline(1.0, color='#c0392b', linewidth=1.0, linestyle='--', alpha=0.7,
               label='FPR = 1%')
    # Evidenzia k=3
    ax.axvline(3.0, color='#7f8c8d', linewidth=1.0, linestyle=':', alpha=0.8,
               label='k = 3 (scelto)')

    ax.set_xlabel('k')
    ax.set_ylabel('FPR (%)')
    ax.set_title('Tasso di falsi positivi su dati normali al variare di k')
    ax.set_xticks(ks)
    ax.legend(loc='upper right')
    fig.tight_layout()
    path = out_dir / 'figure_03_fpr_vs_k.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_finestre_salvate_agg(agg_df: pd.DataFrame, out_dir: Path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x   = np.arange(len(agg_df))
    w   = 0.45

    bars = ax1.bar(x - w / 2, agg_df['saved'], width=w,
                   color='#2ca02c', alpha=0.75, label='Finestre salvate')
    line = ax2.plot(x, agg_df['flagged'], marker='D', markersize=7,
                    color='#d62728', linewidth=2.0, linestyle='-',
                    label='Ancora flaggate')[0]

    # Etichette sulle barre
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     str(int(h)), ha='center', va='bottom', fontsize=8)

    # Etichette sui punti della linea
    for xi, yi in zip(x, agg_df['flagged']):
        ax2.text(xi, yi + 1.5, str(int(yi)), ha='center', va='bottom',
                 fontsize=8, color='#d62728')

    ax1.set_xticks(x)
    ax1.set_xticklabels(agg_df['step'], rotation=20, ha='right')
    ax1.set_ylabel('Finestre salvate (asse sx)', color='#2ca02c')
    ax2.set_ylabel('Finestre ancora flaggate (asse dx)', color='#d62728')
    ax1.set_title('Aggregato — finestre salvate vs ancora flaggate per step Δk')

    # Knee point annotation
    knee_idx = 3   # k=2.5→3.0
    ax1.axvspan(knee_idx - 0.5, knee_idx + 0.5, color='gold', alpha=0.15,
                label='Knee point')

    lines_labels = ([bars, line], ['Finestre salvate', 'Ancora flaggate'])
    ax1.legend(*lines_labels, loc='upper right')
    fig.tight_layout()
    path = out_dir / 'figure_04_finestre_salvate_agg.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_finestre_salvate_per_giunto(each_df: pd.DataFrame, params: dict,
                                     out_dir: Path):
    joints = sorted(params.keys())
    steps  = each_df['step'].unique().tolist()
    n_j    = len(joints)
    ncols  = 3
    nrows  = int(np.ceil(n_j / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows),
                              sharey=False)
    axes = np.array(axes).flatten()

    for idx, jid in enumerate(joints):
        ax  = axes[idx]
        sub = each_df[each_df['Joint_ID'] == jid].sort_values('k0')

        x  = np.arange(len(sub))
        w  = 0.38

        ax.bar(x - w / 2, sub['saved'], width=w,
               color='#2ca02c', alpha=0.75, label='Salvate')
        ax.bar(x + w / 2, sub['flagged'], width=w,
               color='#d62728', alpha=0.65, label='Ancora flaggate')

        ax.set_xticks(x)
        ax.set_xticklabels(sub['step'], rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Finestre')
        ax.set_title(f'Giunto {jid}')
        ax.legend(fontsize=7)

    for ax in axes[n_j:]:
        ax.set_visible(False)

    fig.suptitle('Finestre salvate vs ancora flaggate — per giunto', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    path = out_dir / 'figure_05_finestre_salvate_per_giunto.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_densita_intervalli(border_df: pd.DataFrame, params: dict, out_dir: Path):
    joints = sorted(params.keys())
    steps  = border_df['step'].unique().tolist()
    n_j    = len(joints)
    ncols  = 3
    nrows  = int(np.ceil(n_j / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).flatten()

    bar_colors = ['#2ca02c', '#2ca02c', '#f39c12', '#e74c3c', '#e74c3c', '#e74c3c']

    for idx, jid in enumerate(joints):
        ax  = axes[idx]
        sub = border_df[border_df['Joint_ID'] == jid].sort_values('k0')

        ax.bar(range(len(sub)), sub['count'],
               color=bar_colors[:len(sub)], alpha=0.80, edgecolor='white',
               linewidth=0.5)

        for xi, (cnt, pct) in enumerate(zip(sub['count'], sub['count_%'])):
            if cnt > 0:
                ax.text(xi, cnt + 0.2, f'{cnt}\n({pct}%)',
                        ha='center', va='bottom', fontsize=7)

        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['step'], rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Finestre nell\'intervallo')
        ax.set_title(f'Giunto {jid}')

    for ax in axes[n_j:]:
        ax.set_visible(False)

    fig.suptitle('Densità finestre border-line per intervallo Δk', fontsize=13,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    path = out_dir / 'figure_06_densita_intervalli.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


def fig_k_ottimali(kopt_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    joints = kopt_df['Joint_ID'].tolist()
    kvals  = pd.to_numeric(kopt_df['k_ottimale'], errors='coerce').tolist()
    cols   = []
    for k in kvals:
        if pd.isna(k):
            cols.append('#95a5a6')
        elif k >= 4.0:
            cols.append('#e74c3c')
        elif k <= 2.0:
            cols.append('#2ca02c')
        else:
            cols.append('#2980b9')

    x = np.arange(len(joints))
    bars = ax.bar(x, kvals, color=cols, alpha=0.80, width=0.55,
                  edgecolor='white', linewidth=0.5)

    # Linea k=3 di riferimento
    ax.axhline(3.0, color='#7f8c8d', linewidth=1.2, linestyle='--',
               label='k uniforme = 3')

    # Etichette
    for bar, k in zip(bars, kvals):
        if not pd.isna(k):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    k + 0.05, f'{k:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'J{j}' for j in joints])
    ax.set_ylabel('k ottimale (FPR ≤ 1%)')
    ax.set_title('k ottimale per-giunto che porta FPR ≤ 1% su dati normali')
    ax.set_ylim(0, max(v for v in kvals if not pd.isna(v)) * 1.2)
    ax.legend()

    # Annotation outlier
    for xi, (jid, k) in enumerate(zip(joints, kvals)):
        if not pd.isna(k) and k >= 4.0:
            ax.annotate('outlier', xy=(xi, k),
                        xytext=(xi + 0.35, k + 0.1),
                        fontsize=8, color='#e74c3c',
                        arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                        lw=0.8))

    fig.tight_layout()
    path = out_dir / 'figure_07_k_ottimali.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  [OK] {path.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Tabelle CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_table(df: pd.DataFrame, path: Path, name: str):
    df.to_csv(path, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'  [OK] {path.name}  ({len(df)} righe)')


def save_fpr_pivot(fpr_df: pd.DataFrame, out_dir: Path):
    """Tabella FPR pivotata: righe=k, colonne=giunto."""
    pivot = fpr_df.pivot_table(index='k', columns='Joint_ID',
                               values='FPR_%').round(2)
    pivot.columns = [f'J{c}' for c in pivot.columns]
    pivot['FPR_media'] = pivot.mean(axis=1).round(2)
    pivot['FPR_max']   = pivot.iloc[:, :-1].max(axis=1).round(2)
    pivot = pivot.reset_index()
    save_table(pivot, out_dir / 'tabella_02_fpr_per_k_e_giunto.csv',
               'FPR per k e giunto')


def save_saved_per_giunto_wide(each_df: pd.DataFrame, out_dir: Path):
    """Tabella wide: una riga per giunto x step."""
    save_table(each_df, out_dir / 'tabella_04_finestre_salvate_per_giunto.csv',
               'Finestre salvate per giunto')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Genera tabelle e grafici per la selezione del parametro k.')
    parser.add_argument('--input',  required=True,
                        help='Percorso al CSV di input (sep=; decimal=,)')
    parser.add_argument('--output', default='./output_k_analysis',
                        help='Cartella di output (default: ./output_k_analysis)')
    parser.add_argument('--fpr-target', type=float, default=FPR_TARGET_DEFAULT,
                        help=f'FPR target %% per k ottimale (default: {FPR_TARGET_DEFAULT})')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  generate_k_analysis.py')
    print(f'{"="*60}')
    print(f'  Input  : {args.input}')
    print(f'  Output : {out_dir.resolve()}')
    print(f'  FPR target per k ottimale: {args.fpr_target}%')
    print()

    # ── Caricamento ────────────────────────────────────────────────────────
    print('[ 1/4 ] Caricamento dati...')
    df     = load_data(args.input)
    params = joint_params(df)
    joints = sorted(params.keys())
    n_win_tot = sum(p['n'] for p in params.values())
    print(f'        {len(joints)} giunti, {n_win_tot} finestre totali')

    # ── Calcoli ────────────────────────────────────────────────────────────
    print('[ 2/4 ] Calcoli statistici...')
    fpr_df    = compute_fpr_table(params)
    gauss_df  = compute_gaussianity(params)
    agg_df, each_df = compute_saved_windows(params)
    kopt_df   = compute_k_optimal(params, fpr_target=args.fpr_target)
    border_df = compute_border_density(params)

    # ── Tabelle ────────────────────────────────────────────────────────────
    print('[ 3/4 ] Salvataggio tabelle...')
    save_table(gauss_df,  out_dir / 'tabella_01_gaussianita.csv', 'Gaussianità')
    save_fpr_pivot(fpr_df, out_dir)
    save_table(agg_df,    out_dir / 'tabella_03_finestre_salvate_agg.csv', 'Salvate agg')
    save_saved_per_giunto_wide(each_df, out_dir)
    save_table(kopt_df,   out_dir / 'tabella_05_k_ottimali.csv', 'k ottimali')

    # ── Figure ─────────────────────────────────────────────────────────────
    print('[ 4/4 ] Generazione figure...')
    fig_distribuzione_mse(params, out_dir)
    fig_qqplot(params, out_dir)
    fig_fpr_vs_k(params, fpr_df, out_dir)
    fig_finestre_salvate_agg(agg_df, out_dir)
    fig_finestre_salvate_per_giunto(each_df, params, out_dir)
    fig_densita_intervalli(border_df, params, out_dir)
    fig_k_ottimali(kopt_df, out_dir)

    print(f'\n{"="*60}')
    print(f'  Completato. Output in: {out_dir.resolve()}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
