"""
simulate_backlash.py
=======================
Simula un guasto di tipo backlash (gioco meccanico nel riduttore) sul giunto J2,
a partire da un dataset sano.

Il backlash si manifesta fisicamente in tre modi:
  1. ANGOLO (j2_a): plateau — quando la velocità cambia segno, l'angolo rimane
     fermo per un breve intervallo (il motore recupera il gioco senza muovere l'output).
  2. COPPIA (j2_t): spike impulsivo al momento in cui il gioco viene recuperato
     (il riduttore "batte" contro la parete opposta).
  3. VELOCITÀ (j2_v): lieve distorsione attorno allo zero — rimane a zero
     più a lungo del normale durante il transiente di recupero del gioco.

Rilevamento inversioni: smoothing a blocchi 16x GLOBALE.
  Il segnale j2_v di tutte le traiettorie viene concatenato, mediato a blocchi
  di K campioni, e i sign-change sulla sequenza compressa individuano le inversioni.

Parametri regolabili:
  --backlash_deg   ampiezza del gioco in gradi (default 0.03 → lieve)
  --torque_spike   moltiplicatore dello spike di coppia (default 1.15 → +15%)
  --affected_frac  frazione di traiettorie da modificare (default 1.0 → tutte)
  --seed           seed per riproducibilità
  --plot_traj      quante traiettorie includere nel plot diagnostico (default 30)
  --smooth_k       fattore di compressione per lo smoothing (default 16)

Uso:
python simulate_backlash.py --input  "C:/Users/Carlo/time-series-autoencoder/examples/reconstruction/NUOVI DATI RACCOLTI/QUERY_CSV_SANE2_EXCEL.csv" --output "C:/Users/Carlo/time-series-autoencoder/examples/reconstruction/NUOVI DATI RACCOLTI/QUERY_CSV_j2_EXCEL.csv" --backlash_deg 1.0 --torque_spike 1.15

USARLO CON QUESTI VALORI

Per un guasto più lieve:    --backlash_deg 0.015 --torque_spike 1.08
Per un guasto più visibile: --backlash_deg 0.06  --torque_spike 1.25
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# RILEVAMENTO INVERSIONI — smoothing a blocchi K× GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

def find_inversions_global(v_full: np.ndarray, K: int = 16) -> np.ndarray:
    """
    Trova le inversioni di moto sul segnale globale concatenato.

    Algoritmo:
      1. Divide v_full in blocchi non sovrapposti di K campioni.
      2. Calcola la media di ogni blocco → sequenza compressa.
      3. Applica np.sign() (0 → +1).
      4. Ogni sign-change = un'inversione.
      5. Restituisce gli indici nell'array originale (fine del blocco: c*K + K-1).

    Args:
        v_full: segnale di velocità globale (tutte le traiettorie concatenate)
        K:      fattore di compressione (default 16)

    Returns:
        Array di indici globali dove avviene l'inversione.
    """
    n_blocks = len(v_full) // K
    v_comp   = np.array([v_full[i * K:(i + 1) * K].mean() for i in range(n_blocks)])
    sign_comp = np.sign(v_comp)
    sign_comp[sign_comp == 0] = 1
    cross_blocks = np.where(np.diff(sign_comp) != 0)[0]
    return cross_blocks * K + (K - 1)


# ─────────────────────────────────────────────────────────────────────────────
# CORE: applica backlash a una singola traiettoria
# ─────────────────────────────────────────────────────────────────────────────

def apply_backlash(v: np.ndarray,
                   a: np.ndarray,
                   t: np.ndarray,
                   crossings_global: list,
                   backlash_rad: float,
                   torque_spike_mult: float,
                   rng: np.random.Generator) -> tuple:
    """
    Applica il backlash ai segnali di una singola traiettoria.

    Args:
        v, a, t:           segnali velocità, angolo, coppia (N,)
        crossings_global:  indici globali delle inversioni su tutto il segnale
        backlash_rad:      ampiezza del gioco in radianti
        torque_spike_mult: moltiplicatore dello spike di coppia
        rng:               generatore numpy

    Returns:
        v_mod, a_mod, t_mod
    """
    v_mod = v.copy().astype(float)
    a_mod = a.copy().astype(float)
    t_mod = t.copy().astype(float)
    N = len(v)

    if len(crossings_global) == 0:
        # Nessuna inversione: piccolo drift elastico sull'angolo
        drift = np.linspace(0, backlash_rad * 0.3, N) * rng.choice([-1, 1])
        a_mod += drift
        return v_mod, a_mod, t_mod

    for idx in crossings_global:
        if idx >= N:
            continue

        # ── 1. Durata del plateau ─────────────────────────────────────────
        v_pre = abs(v[max(0, idx - 9):idx + 1].mean())   # 10 campioni

        if v_pre < 1e-6:
            v_pre = 1e-6
        n_plateau   = int(np.clip(backlash_rad / (v_pre + 1e-6) * 0.5, 10, 20))
        plateau_end = min(idx + 1 + n_plateau, N)

        # ── 2. ANGOLO: freeze durante il plateau ──────────────────────────
        a_freeze = a_mod[idx]
        a_mod[idx + 1: plateau_end] = a_freeze

        ramp_len = min(5, N - plateau_end)
        if ramp_len > 0 and plateau_end < N:
            target  = a[plateau_end]
            current = a_mod[plateau_end - 1]
            ramp = np.linspace(current, target, ramp_len + 2)[1:-1]
            a_mod[plateau_end: plateau_end + ramp_len] = ramp

        # ── 3. VELOCITÀ: zero durante il plateau ──────────────────────────
        v_mod[idx + 1: plateau_end] = 0.0

        # ── 4. COPPIA: spike al termine del plateau ───────────────────────
        spike_start = plateau_end
        spike_len   = rng.integers(2, 4)
        spike_end   = min(spike_start + spike_len, N)
        if spike_start < N:
            jitter = rng.uniform(0.97, 1.03)
            t_mod[spike_start: spike_end] *= torque_spike_mult * jitter

    return v_mod, a_mod, t_mod


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICA: plot delle prime N traiettorie su asse temporale continuo
# ─────────────────────────────────────────────────────────────────────────────

def plot_inversions(df: pd.DataFrame,
                    n_traj: int,
                    smooth_k: int,
                    out_path: Path) -> None:
    """
    Plotta le prime n_traj traiettorie di j2_v su un unico asse temporale
    continuo, marcando le inversioni trovate con find_inversions_global().
    """
    traj_ids = df['trajectory_id'].unique()[:n_traj]

    # Costruisce asse continuo
    segments    = []
    traj_bounds = []
    x_cursor    = 0
    for tid in traj_ids:
        v = df[df['trajectory_id'] == tid]['j2_v'].values
        traj_bounds.append((tid, x_cursor, x_cursor + len(v)))
        segments.append((tid, v, x_cursor))
        x_cursor += len(v)

    total_len = x_cursor
    v_full    = np.concatenate([v for _, v, _ in segments])
    x_full    = np.arange(total_len)

    # Inversioni globali
    crossings = find_inversions_global(v_full, K=smooth_k)

    # Segnale smoothed per visualizzazione
    n_blocks = total_len // smooth_k
    v_comp   = np.array([v_full[i * smooth_k:(i + 1) * smooth_k].mean()
                         for i in range(n_blocks)])
    x_comp   = np.arange(n_blocks) * smooth_k + smooth_k // 2

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(26, 5))

    ax.plot(x_full, v_full,
            color='#adb5bd', lw=0.7, alpha=0.75, zorder=2, label='j2_v originale')
    ax.plot(x_comp, v_comp,
            color='#1d3557', lw=1.3, alpha=0.9, zorder=3,
            label=f'Media {smooth_k}× (smoothed)')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.5, zorder=1)

    ymin, ymax = ax.get_ylim()

    for tid, x_start, x_end in traj_bounds:
        ax.axvline(x_start, color='#457b9d', lw=0.9, ls=':', alpha=0.5, zorder=1)
        ax.text((x_start + x_end) / 2, ymax * 0.88,
                str(tid), ha='center', va='top', fontsize=6.5, color='#457b9d')

    for cx in crossings:
        if cx < total_len:
            ax.axvline(cx, color='#e63946', lw=1.4, alpha=0.9, zorder=4)
            ax.plot(cx, v_full[cx], 'o', color='#e63946', ms=5, zorder=5)

    n_inv = len(crossings)
    ax.set_title(
        f"j2_v — prime {n_traj} traiettorie, asse temporale continuo\n"
        f"Inversioni rilevate (smoothing globale {smooth_k}×): N={n_inv}",
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel("Campione (asse continuo)", fontsize=10)
    ax.set_ylabel("j2_v", fontsize=10)
    ax.set_xlim(0, total_len)
    ax.grid(axis='y', alpha=0.2)

    patch_orig   = mpatches.Patch(color='#adb5bd', label='j2_v originale')
    patch_smooth = mpatches.Patch(color='#1d3557', label=f'Media {smooth_k}×')
    patch_bound  = mpatches.Patch(color='#457b9d', label='Boundary traiettoria', alpha=0.6)
    patch_inv    = mpatches.Patch(color='#e63946', label=f'Inversione ({n_inv} totali)')
    ax.legend(handles=[patch_orig, patch_smooth, patch_bound, patch_inv],
              fontsize=8, loc='upper right')

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot salvato: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simula backlash su j2")
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--backlash_deg",  type=float, default=0.03)
    parser.add_argument("--torque_spike",  type=float, default=1.15)
    parser.add_argument("--affected_frac", type=float, default=1.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--plot_traj",     type=int,   default=30)
    parser.add_argument("--smooth_k",      type=int,   default=16)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Caricamento: {args.input}")
    df     = pd.read_csv(args.input, sep=';', decimal=',')
    df_out = df.copy()

    backlash_rad = np.deg2rad(args.backlash_deg)
    print(f"Backlash:     {args.backlash_deg}° = {backlash_rad:.6f} rad")
    print(f"Spike coppia: ×{args.torque_spike}")
    print(f"Smoothing:    {args.smooth_k}× (globale)")

    # ── Calcolo inversioni globali su tutto il dataset ────────────────────
    traj_ids   = df['trajectory_id'].unique()
    v_full_all = df['j2_v'].values

    # Mappa: per ogni traiettoria, l'offset globale dove inizia
    x_offsets = {}
    x_cursor  = 0
    for tid in traj_ids:
        x_offsets[tid] = x_cursor
        x_cursor += (df['trajectory_id'] == tid).sum()

    crossings_global = find_inversions_global(v_full_all, K=args.smooth_k)
    print(f"Inversioni trovate (globale {args.smooth_k}×): {len(crossings_global)}")

    # Assegna ogni crossing alla traiettoria di appartenenza
    # e converte in indice locale
    traj_crossings: dict = {tid: [] for tid in traj_ids}
    traj_lengths   = {tid: (df['trajectory_id'] == tid).sum() for tid in traj_ids}

    for cx in crossings_global:
        for tid in traj_ids:
            x0  = x_offsets[tid]
            x1  = x0 + traj_lengths[tid]
            if x0 <= cx < x1:
                traj_crossings[tid].append(cx - x0)  # indice locale
                break

    # ── Plot diagnostico ──────────────────────────────────────────────────
    out_path  = Path(args.output)
    plot_path = out_path.with_name(out_path.stem + "_inversions.png")
    plot_inversions(df, n_traj=args.plot_traj, smooth_k=args.smooth_k,
                    out_path=plot_path)

    # ── Selezione traiettorie da modificare ───────────────────────────────
    n_affected   = max(1, int(len(traj_ids) * args.affected_frac))
    affected_ids = rng.choice(traj_ids, size=n_affected, replace=False)
    print(f"Traiettorie totali: {len(traj_ids)}  |  Modificate: {n_affected}")
    

    # DOPO
    v_global = df['j2_v'].values.copy().astype(float)
    a_global = df['j2_a'].values.copy().astype(float)
    t_global = df['j2_t'].values.copy().astype(float)
    
    v_mod_g, a_mod_g, t_mod_g = apply_backlash(
        v_global, a_global, t_global,
        crossings_global,
        backlash_rad, args.torque_spike, rng
    )
    
    df_out['j2_v'] = v_mod_g
    df_out['j2_a'] = a_mod_g
    df_out['j2_t'] = t_mod_g
    
    # ── Stats per traiettoria ─────────────────────────────────────────
    stats = []
    for tid in traj_ids:
        mask  = df['trajectory_id'] == tid
        idx   = df.index[mask]
        n_inv = len(traj_crossings[tid])
    
        a_orig = df.loc[idx, 'j2_a'].values
        t_orig = df.loc[idx, 'j2_t'].values
        a_new  = df_out.loc[idx, 'j2_a'].values
        t_new  = df_out.loc[idx, 'j2_t'].values
    
        delta_a = np.max(np.abs(a_new - a_orig))
        delta_t = np.max(np.abs(t_new - t_orig))
    
        modificata = tid in affected_ids
        stats.append({'traj_id': tid, 'modificata': modificata,
                      'n_inversioni': n_inv,
                      'delta_a_max': round(delta_a, 6),
                      'delta_t_max': round(delta_t, 4)})
    
        if modificata:
            print(f"  Traj {tid}: {n_inv} inv "
                  f"| Δa_max={delta_a:.6f} rad | Δt_max={delta_t:.3f} Nm")



    # ── Salvataggio ───────────────────────────────────────────────────────
    df_out.to_csv(out_path, sep=';', decimal=',', index=False)
    print(f"\nSalvato: {out_path}  ({len(df_out)} righe)")

    df_stats = pd.DataFrame(stats)
    mod = df_stats[df_stats['modificata']]
    print(f"\n── Riepilogo modifiche ──────────────────────────────")
    print(f"Traiettorie con inversioni: {(mod['n_inversioni'] > 0).sum()}/{len(mod)}")
    if len(mod) > 0 and mod['delta_a_max'].max() > 0:
        print(f"Δ angolo max medio:  {mod['delta_a_max'].mean():.6f} rad")
        print(f"Δ coppia max medio:  {mod['delta_t_max'].mean():.3f} Nm")
        print(f"Δ angolo max totale: {mod['delta_a_max'].max():.6f} rad")
        print(f"Δ coppia max totale: {mod['delta_t_max'].max():.3f} Nm")

    stats_path = out_path.with_name(out_path.stem + "_stats.csv")
    df_stats.to_csv(stats_path, sep=';', decimal=',', index=False)
    print(f"Stats salvate: {stats_path}")


if __name__ == "__main__":
    main()
