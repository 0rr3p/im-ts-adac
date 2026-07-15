import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import matplotlib.pyplot as plt
import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.gridspec as gridspec
import random
from scipy.stats import skew, kurtosis

import faiss
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint
import sys
import subprocess

from pathlib import Path

FAISS_SIMILARITY_THRESHOLD= 0.167
RF_THRESHOLD= 0.99
AUTO_FALSE_POSITIVE = False
AUTO_ENTER = True

def focus_terminal():
    """Riporta il focus sulla finestra del terminale."""
    if sys.platform == "win32":
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    elif sys.platform == "darwin":
        subprocess.run(["osascript", "-e",
            'tell application "Terminal" to activate'], check=False)
    else:  # Linux
        subprocess.run(["wmctrl", "-a", ":ACTIVE:"], check=False)
        

def find_latest_artifacts(joint_id):
    # Path relativo: ci troviamo in 'reconstruction', cerchiamo la cartella 'outputs'
    base_path = Path(__file__).resolve().parent / "multirun"
    
    if not base_path.exists():
        raise FileNotFoundError(f"Directory multirun non trovata in: {base_path}")

    # Scansione cronologica inversa (Date decrescenti)
    for date_folder in sorted(base_path.iterdir(), reverse=True):
        if not date_folder.is_dir(): continue
        
        # Scansione cronologica inversa (Ore decrescenti)
        for time_folder in sorted(date_folder.iterdir(), reverse=True):
            if not time_folder.is_dir(): continue

            target_dir = time_folder / str(joint_id)

            if not target_dir.exists() or not target_dir.is_dir():
                continue # Se in questo run non c'è questo giunto, passa a quello precedente
             
            scaler_path = target_dir / f"scaler_joint{joint_id}.pkl"
            model_path = target_dir / "output" / "best_model.ckpt"

            if scaler_path.exists() and model_path.exists():
                return str(scaler_path), str(model_path), target_dir
    
    raise FileNotFoundError(f"Nessun file trovato per il giunto {joint_id} nelle cartelle di output.")


def extract_statistical_features(residual_Tx3):
    features = []
    for j in range(3):
        ch = residual_Tx3[:, j]
        features.append(np.mean(ch))                           # 1. Mean
        features.append(np.std(ch))                            # 2. Std
        features.append(float(skew(ch)))                       # 3. Skewness
        features.append(float(kurtosis(ch)))                   # 4. Kurtosis
        features.append(np.max(ch) - np.min(ch))              # 5. Range
        features.append(np.sum(np.diff(np.sign(ch)) != 0))    # 6. Zero crossing rate
        features.append(float(np.corrcoef(ch[:-1], ch[1:])[0,1])) # 7. Autocorr lag-1
        features.append(np.percentile(np.abs(ch), 95))        # 8. P95
        features.append(np.sum(np.abs(np.diff(ch))))           # 9. Total variation
        features.append(np.max(np.abs(ch)))                    # 10. Peak absolute
    return np.array(features)  # shape: (30,)

    
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── DUAL-INDEX FAISS ────────────────────────────────────────────────────────

def load_dual_index(faiss_dir, joint_id, enc_h):
    raw_path       = str(faiss_dir / f"raw_index_joint{joint_id}.index")
    centroids_path = str(faiss_dir / f"centroids_index_joint{joint_id}.index")
    meta_path      = faiss_dir / f"fp_metadata_joint{joint_id}.joblib"

    raw_index = faiss.read_index(raw_path) if Path(raw_path).exists() \
                else faiss.IndexFlatL2(enc_h)
    centroids_index = faiss.read_index(centroids_path) if Path(centroids_path).exists() \
                      else faiss.IndexFlatL2(enc_h)
    metadata = joblib.load(meta_path) if meta_path.exists() \
               else {'cluster_labels': [], 'cluster_counts': []}

    print(f"FAISS dual-index: {raw_index.ntotal} raw, {centroids_index.ntotal} centroidi")
    return raw_index, centroids_index, metadata, raw_path, centroids_path, meta_path


def add_fp_dual_index(emb, raw_index, centroids_index, metadata,
                       raw_path, centroids_path, meta_path, enc_h):
    emb_f32 = emb.reshape(1, -1).astype('float32')
    
    centroids_index.add(emb_f32)
    assigned_cluster = len(metadata['cluster_counts'])
    
    metadata['cluster_counts'].append(1)
    metadata['cluster_labels'].append(assigned_cluster)
    raw_index.add(emb_f32)

    faiss.write_index(raw_index, raw_path)
    faiss.write_index(centroids_index, centroids_path)
    joblib.dump(metadata, meta_path)
    print(f"   ➕ Nuovo cluster #{assigned_cluster}")
    print(f"   💾 {raw_index.ntotal} raw → {centroids_index.ntotal} centroidi")

    return raw_index, centroids_index, metadata

# ────────────────────────────────────────────────────────────────────────────    
 
@hydra.main(version_base="1.1",config_path="./", config_name="config")
def run_detection(cfg):

    set_seed(42)

    SELECTED_JOINT = cfg.joint_id
    N = cfg.training.first_n_ignore

    FAISS_DIR = Path(__file__).resolve().parent  / "faiss"
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    enc_h = cfg.training.hidden_size_encoder
    raw_index, centroids_index, fp_metadata, \
        raw_path, centroids_path, meta_path = load_dual_index(FAISS_DIR, SELECTED_JOINT, enc_h)
    
    RF_PATH = FAISS_DIR / f"rf_model_joint.joblib"
    if RF_PATH.exists():
        rf_model = joblib.load(RF_PATH)
        print(f"✅ Modello RF caricato: {RF_PATH}")
    else:
        rf_model = None
        
    DATA_XY_PATH = FAISS_DIR / f"rf_data_joint.joblib"
    if DATA_XY_PATH.exists():
        history = joblib.load(DATA_XY_PATH)
        print(f"✅ history caricato: {DATA_XY_PATH}")
    else:
        history = {"X": [], "y": []}
        
    LABEL_MAP_PATH = FAISS_DIR / f"label_map_joint.joblib"
    if LABEL_MAP_PATH.exists():
        label_map = joblib.load(LABEL_MAP_PATH)
        print(f"✅ Label map caricata: {LABEL_MAP_PATH}")
    else:
        label_map = {}
        
    print(f"DEBUG: Backend in uso -> {plt.get_backend()}") 
    plt.switch_backend('TkAgg') 
    print(f"Backend forzato a: {plt.get_backend()}")
    
    try:
        PATH_SCALER, PATH_CKPT, FOLDER = find_latest_artifacts(SELECTED_JOINT)
        print(f"--- Rilevamento per Giunto {SELECTED_JOINT} ---")
        print(f"Sorgente: {FOLDER}")
    except Exception as e:
        print(f"Errore: {e}")
        return

    # 2. Preparazione Dataset
    cfg.data.data_path = cfg.path_ad
    ts = instantiate(cfg.data)
    
    ts.load_scaler(PATH_SCALER)
    _, test_scaled, malati_ids = ts.preprocess_with_loaded_scaler()
    test_dataset = ts.frame_series(test_scaled, malati_ids)
    
    from tsa.dataset import seed_worker 
    
    g = torch.Generator()
    g.manual_seed(42)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
    )

    # 3. Caricamento Modello
    nb_features = len(cfg.data.feature_cols)
    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    model, _, _, _ = load_checkpoint(PATH_CKPT, model, optimizer, device)
    model.eval()
    
    checkpoint = torch.load(PATH_CKPT)
    mu = checkpoint.get('mu', 0.0)
    if torch.is_tensor(mu): mu = mu.item()
    sigma = checkpoint.get('sigma', 0.0)
    if torch.is_tensor(sigma): sigma = sigma.item()

    print("\n" + "="*30)
    print("🔎 VERIFICA CARICAMENTO CHECKPOINT")
    print(f"Media Errore (mu):    {mu:.6f}")
    print(f"Deviazione (sigma):   {sigma:.6f}")
    
    if sigma != 0:
        THRESHOLD = mu + (cfg.data.k * sigma)
        print(f"Soglia Dinamica:      {THRESHOLD:.6f} (μ + {cfg.data.k}σ)")
    else:
        THRESHOLD = 0.05 
        print(f"Soglia Statistica:    {THRESHOLD:.6f} (Default manuale)")
    print("="*30 + "\n")
    
    criterion = nn.MSELoss(reduction='none')

    if isinstance(history['X'], np.ndarray):
        history['X'] = history['X'].tolist()
    if isinstance(history['y'], np.ndarray):
        history['y'] = history['y'].tolist()

    # 4. Inferenza per traj_id
    from collections import OrderedDict

    all_errors   = []
    all_traj_ids = []

    session_labels = {}       
    all_windows = []  
    _win_counter = {}  

    print("Inizio analisi dati anomali...")
    with torch.no_grad():
        for batch in test_loader:
            features, y_hist, target, batch_ids, batch_starts = [b.to(device) for b in batch]

            _, latent_seq = model.encoder(features)
            if model.temporal_pool is not None:
                latent_avg = model.temporal_pool(latent_seq).detach().cpu().numpy().astype('float32')
            else:
                latent_avg = latent_seq.mean(dim=1).cpu().numpy().astype('float32')
            output = model(features, y_hist)

            loss = criterion(output[:, N:, :], target[:, N:, :])
            error_torch = loss.mean(dim=(1, 2)).cpu().numpy()
            delta = (target - output).cpu().numpy()

            all_errors.extend(error_torch.tolist())
            all_traj_ids.extend(batch_ids.cpu().numpy().tolist())

            for i in range(len(error_torch)):
                tid = int(batch_ids[i].item())

                win_idx = _win_counter.get(tid, 0)
                _win_counter[tid] = win_idx + 1
                all_windows.append({
                    'id':       tid,
                    'win_idx':  win_idx,   
                    'mse':      float(error_torch[i]),
                    'start': int(batch_starts[i].item()),
                    'emb':      latent_avg[i],
                    'residual': delta[i],
                    'target':   target[i].cpu().numpy(),
                    'output':   output[i].cpu().numpy(),
                })

    traj_windows: dict = OrderedDict()
    for w in all_windows:
        traj_windows.setdefault(w['id'], []).append(w)

    # 5. Plot riepilogativo MSE
    all_anom_windows_recap = [w for ws in traj_windows.values() for w in ws if w['mse'] > THRESHOLD]

    if all_anom_windows_recap:
            df_riepilogo = pd.DataFrame([{'id': w['id'], 'finestra': w['win_idx'], 'mse': w['mse']} for w in all_anom_windows_recap])
            
            x_labels = [f"{int(row['id'])}-w{int(row['finestra'])}" for _, row in df_riepilogo.iterrows()]
            plt.figure(figsize=(12, 4))
            plt.plot(range(len(df_riepilogo)), df_riepilogo['mse'], color='steelblue', linestyle='-', alpha=0.6, label='Andamento MSE')
            plt.stem(range(len(df_riepilogo)), df_riepilogo['mse'], linefmt='steelblue', markerfmt='o', basefmt=" ")
            plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Soglia Anomalia')
            plt.title(f"MSE finestre anomale - Giunto {SELECTED_JOINT}")
            
            plt.xticks(range(len(df_riepilogo)), x_labels, rotation=45, fontsize=8)
            
            plt.ylabel("MSE")
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            focus_terminal()  

            print("\n--- ELENCO FINESTRE ANOMALE RILEVATE ---")
            print(df_riepilogo[['id', 'finestra', 'mse']].to_string(index=False))
            
            n_traiettorie_anomale = df_riepilogo['id'].nunique()
            n_finestre_anomale = len(all_anom_windows_recap)
            
            print(f"\nTraiettorie da analizzare: {len(traj_windows)}")
            print(f"Traiettorie anomale da analizzare: {n_traiettorie_anomale}")
            print(f"\nFinestre totali da analizzare: {len(all_windows)}")
            print(f"Finestre anomale totali da analizzare: {n_finestre_anomale}")
            print(f"{len(traj_windows)},{n_traiettorie_anomale},{len(all_windows)},{n_finestre_anomale}")
        
    else:
        print("Nessuna finestra anomala rilevata.")

    # 6. Loop per traj_id: check FAISS → feedback
    confirmed_anomaly_infos = []   
    
    # 🌟 NUOVO: Lista unica per tutti i record anomali
    unified_records = []
    dataset_name = Path(cfg.data.data_path).stem

    for traj_id, windows in traj_windows.items():
        n_finestre_totali_traj = len(windows)

        embs = np.array([w['emb'] for w in windows]).astype('float32')

        if centroids_index.ntotal > 0:
            distances, _ = centroids_index.search(embs, 1)
            is_known_fp_per_win = distances.flatten() < FAISS_SIMILARITY_THRESHOLD
        else:
            is_known_fp_per_win = np.zeros(len(windows), dtype=bool)

        for w, is_fp in zip(windows, is_known_fp_per_win):
            w['is_known_fp'] = bool(is_fp)

        # 🌟 NOVITÀ: Salvataggio delle finestre sotto soglia (Sane)
        normal_windows = [w for w in windows if w['mse'] <= THRESHOLD]
        for w in normal_windows:
            unified_records.append({
                'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                'win_idx': w['win_idx'], 'mse': w['mse'], 'mu_training': mu,
                'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'normale',
                'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
            })

        anom_windows = [
            w for w in windows
            if w['mse'] > THRESHOLD and not w['is_known_fp']
        ]

        fp_skipped = [w for w in windows if w['mse'] > THRESHOLD and w['is_known_fp']]
        for w in fp_skipped:
            print(f"⏭️  ID {traj_id} finestra {w['win_idx']}: riconosciuta come FP da FAISS, ignorata.")
            
            emb_f32 = w['emb'].reshape(1, -1).astype('float32')
            distances, indices = centroids_index.search(emb_f32, 1)
            nearest_idx = int(indices[0, 0])
            n = fp_metadata['cluster_counts'][nearest_idx]
            old_centroid = centroids_index.reconstruct(nearest_idx).copy()
            new_centroid = old_centroid + (emb_f32[0] - old_centroid) / (n + 1)
            
            all_centroids = np.stack([
                new_centroid if i == nearest_idx else centroids_index.reconstruct(i)
                for i in range(centroids_index.ntotal)
            ]).astype('float32')
            centroids_index = faiss.IndexFlatL2(enc_h)
            centroids_index.add(all_centroids)
            fp_metadata['cluster_counts'][nearest_idx] = n + 1
            
            raw_index.add(emb_f32)
            fp_metadata['cluster_labels'].append(nearest_idx)
            faiss.write_index(raw_index, raw_path)        

            faiss.write_index(centroids_index, centroids_path)
            joblib.dump(fp_metadata, meta_path)
            
            # 🌟 NUOVO: Aggiunta al log unificato per FP noto
            unified_records.append({
                'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                'win_idx': w['win_idx'], 'mse': w['mse'], 'mu_training': mu,
                'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'FP_noto',
                'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
            })
    
        if not anom_windows:
            continue  

        feature_names = cfg.data.feature_cols
        all_wins_sorted = sorted(windows, key=lambda w: w['win_idx'])
        seq_len = anom_windows[0]['target'].shape[0]

        T_total = max(w['start'] + seq_len for w in all_wins_sorted)
        full_target = np.zeros((T_total, all_wins_sorted[0]['target'].shape[1]), dtype=np.float32)
        for w in all_wins_sorted:
            full_target[w['start']: w['start'] + seq_len] = w['target']

        win_starts = {w['win_idx']: w['start'] for w in all_wins_sorted}

        full_output_sum   = np.zeros((T_total, full_target.shape[1]), dtype=np.float64)
        full_output_count = np.zeros((T_total, 1), dtype=np.float64)
        
        for w in all_wins_sorted:
            t0 = win_starts[w['win_idx']]
            t1 = min(t0 + seq_len, T_total)
            actual_len = t1 - t0
            full_output_sum[t0:t1]   += w['output'][:actual_len]
            full_output_count[t0:t1] += 1
        
        full_output_count = np.where(full_output_count == 0, 1, full_output_count)
        full_output = (full_output_sum / full_output_count).astype(np.float32)
        
        all_anom_intervals = [(win_starts[w['win_idx']], win_starts[w['win_idx']] + seq_len)
                              for w in anom_windows]

        n_anom_total = len(anom_windows)
        print(f"\n{'='*50}")
        print(f"ID {traj_id}: {n_anom_total} finestre anomale da analizzare.")
        print(f"{'='*50}")

        for win_num, w in enumerate(anom_windows):
            win_idx = w['win_idx']
            print(f"\n--- ID {traj_id} | Finestra {win_num + 1}/{n_anom_total} (win_idx={win_idx}) | MSE={w['mse']:.6f} ---")

            stat_features = extract_statistical_features(w['residual']).reshape(1, -1)

            win_key = (traj_id, win_idx)
            classified_by_rf = False             
            if win_key not in session_labels and rf_model:
                probs      = rf_model.predict_proba(stat_features)[0]
                best_idx   = np.argmax(probs)
                best_class = rf_model.classes_[best_idx]
                best_prob  = probs[best_idx]
                best_label = label_map.get(best_class, f"Classe_{best_class}")

                print(f" Probabilità predette:")
                for class_idx, prob in sorted(zip(rf_model.classes_, probs), key=lambda x: -x[1]):
                    nome = label_map.get(class_idx, f"Classe_{class_idx}")
                    print(f"   {nome}: {prob*100:.1f}%")

                if best_prob >= RF_THRESHOLD:
                    print(f"✅ Classificazione automatica: '{best_label}' ({best_prob*100:.1f}% >= {RF_THRESHOLD*100:.0f}%)")
                    session_labels[win_key] = best_label
                    classified_by_rf = True          


            elif win_key not in session_labels:
                print(f" Nessun modello RF disponibile.")

            fig = plt.figure(figsize=(14, 11), layout='tight')
            gs = gridspec.GridSpec(7, 1, figure=fig,
                                   height_ratios=[1, 1, 1, 0.3, 2, 2, 2],
                                   hspace=0.05)

            ax_mini = [fig.add_subplot(gs[i]) for i in range(3)]
            t_axis = np.arange(T_total)
            for i in range(3):
                ax_mini[i].plot(t_axis, full_target[:, i], color='tab:blue', lw=0.8)
                ax_mini[i].plot(t_axis, full_output[:, i], color='tab:orange',lw=0.8, linestyle='--', alpha=0.75, label='Ricostruito')
                for (t0, t1) in all_anom_intervals:
                    ax_mini[i].axvspan(t0, min(t1, T_total), color='red', alpha=0.15)
                t0_curr = win_starts[win_idx]
                ax_mini[i].axvspan(t0_curr, min(t0_curr + seq_len, T_total), color='orange', alpha=0.45)
                ax_mini[i].set_ylabel(feature_names[i], fontsize=7)
                ax_mini[i].tick_params(labelsize=6)
                ax_mini[i].grid(True, alpha=0.2)
                if i < 2:
                    ax_mini[i].set_xticklabels([])
            ax_mini[0].set_title(
                f"ID: {traj_id} — Traiettoria completa (blu=reale, arancione=ricostruito) |  "
                f"Finestra {win_num + 1}/{n_anom_total} (arancione)", fontsize=9)
            ax_mini[2].set_xlabel("Campione", fontsize=7)

            fig.add_subplot(gs[3]).axis('off')

            ax_det = [fig.add_subplot(gs[4 + i]) for i in range(3)]
            x_axis_det = np.arange(seq_len) + win_starts[win_idx]

            for i in range(3):
                ax_det[i].plot(x_axis_det, w['target'][:, i], color='tab:blue',   label='Input reale')
                ax_det[i].plot(x_axis_det[N:], w['output'][N:, i], color='tab:orange', linestyle='--', label='Ricostruito')
                
                ax_det[i].set_ylabel(feature_names[i], fontsize=8)
                ax_det[i].legend(loc='upper right', fontsize=7)
                ax_det[i].grid(True, alpha=0.3)
                if i < 2:
                    ax_det[i].set_xticklabels([])
                    
            ax_det[0].set_title(
                f"Dettaglio finestra {win_num + 1} (win_idx={win_idx}) | MSE={w['mse']:.6f} | Inizio t={x_axis_det[0]}", fontsize=9)

            plt.show(block=False)
            plt.pause(0.1)
            focus_terminal()  

            if win_key not in session_labels:
             
                if AUTO_FALSE_POSITIVE:
                    print(f" AUTO_FALSE_POSITIVE attivo: finestra {win_idx} di ID {traj_id} classificata come FP.")
                    raw_index, centroids_index, fp_metadata = add_fp_dual_index(
                        w['emb'], raw_index, centroids_index, fp_metadata,
                        raw_path, centroids_path, meta_path,
                        enc_h=enc_h
                    )
                    plt.close(fig)
                    
                    # 🌟 NUOVO: Log per auto FP
                    unified_records.append({
                        'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                        'win_idx': win_idx, 'mse': w['mse'], 'mu_training': mu,
                        'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'FP_nuovo',
                        'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
                    })
                    continue 
                elif AUTO_ENTER:
                    print(f" AUTO_ENTER attivo: finestra {win_idx} di ID {traj_id} saltata (come Invio).")
                    plt.close(fig)
                    unified_records.append({
                        'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                        'win_idx': win_idx, 'mse': w['mse'], 'mu_training': mu,
                        'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'saltata',
                        'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
                    })
                    continue
                
                print(f"\n Guarda il grafico e le probabilità per la finestra {win_num + 1}.")
                ans = input(
                    f"Etichetta per ID {traj_id} finestra {win_idx} "
                    f"(classe RF, 'false' = falso positivo, 's' per saltare): "
                )
                print(f"\n ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
                plt.close(fig)
                
                if ans.strip().lower() == 'false':
                    raw_index, centroids_index, fp_metadata = add_fp_dual_index(
                        w['emb'], raw_index, centroids_index, fp_metadata,
                        raw_path, centroids_path, meta_path,
                        enc_h=enc_h
                    )
                    print(f"✅ FAISS aggiornato: {raw_index.ntotal} raw → {centroids_index.ntotal} centroidi")

                    # 🌟 NUOVO: Log per FP inserito a mano
                    unified_records.append({
                        'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                        'win_idx': win_idx, 'mse': w['mse'], 'mu_training': mu,
                        'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'FP_nuovo',
                        'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
                    })
                    continue 

                if ans.lower() != 's' and ans.strip():
                    session_labels[win_key] = ans
                else:
                    # 🌟 NUOVO: Log per finestra saltata
                    unified_records.append({
                        'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                        'win_idx': win_idx, 'mse': w['mse'], 'mu_training': mu,
                        'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': 'saltata',
                        'label': '', 'n_finestre_totali_traj': n_finestre_totali_traj
                    })
            else:
                plt.close(fig)

            if win_key in session_labels:
                ans = session_labels[win_key]
                if ans.isdigit():
                    l_id = int(ans)
                    if l_id not in label_map:
                        label_map[l_id] = f"Guasto_{l_id}"
                else:
                    l_id = next((k for k, v in label_map.items() if v == ans), len(label_map))
                    label_map[l_id] = ans

                esito = 'auto_RF' if classified_by_rf else 'confermata'
                
                # 🌟 NUOVO: Log per finestra etichettata
                unified_records.append({
                    'dataset': dataset_name, 'joint_id': SELECTED_JOINT, 'traj_id': traj_id,
                    'win_idx': win_idx, 'mse': w['mse'], 'mu_training': mu,
                    'sigma_training': sigma, 'threshold': THRESHOLD, 'esito': esito,
                    'label': label_map.get(l_id, ans), 'n_finestre_totali_traj': n_finestre_totali_traj
                })

                feat = extract_statistical_features(w['residual']).flatten()
                history['X'].append(feat)
                history['y'].append(l_id)
                confirmed_anomaly_infos.append(w)

    print(f"\nDetection completata.")

    # 7. Ri-addestramento RF
    if history['X'] and len(set(history['y'])) > 1:
        new_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        new_rf.fit(history['X'], history['y'])
        joblib.dump(new_rf, RF_PATH)
        joblib.dump(history, DATA_XY_PATH)
        joblib.dump(label_map, LABEL_MAP_PATH)
        print("✅ Random Forest aggiornato con le nuove feature statistiche.")

    # 8. Salvataggio .npy anomalie confermate
    if confirmed_anomaly_infos:
        final_residuals  = np.stack([w['residual'] for w in confirmed_anomaly_infos])
        final_embeddings = np.stack([w['emb']      for w in confirmed_anomaly_infos])
        np.save(f"residuals_joint{SELECTED_JOINT}.npy",  final_residuals)
        np.save(f"embeddings_joint{SELECTED_JOINT}.npy", final_embeddings)
        print(f"📂 File .npy salvati con {len(confirmed_anomaly_infos)} finestre anomale confermate.")
    else:
        print("ℹ️ Nessuna anomalia confermata, i file .npy non sono stati aggiornati.")

    # 🌟 NUOVO: 9. SALVATAGGIO CSV UNIFICATO
    if unified_records:
        unified_csv_path = Path(__file__).resolve().parent / "unified_anomaly_log.csv"
        file_exists_unified = os.path.exists(unified_csv_path)
        pd.DataFrame(unified_records).to_csv(
            unified_csv_path, mode='a', index=False,
            header=not file_exists_unified, sep=';', decimal=','
        )
        print(f"📊 Dati consolidati salvati in 'unified_anomaly_log.csv' ({len(unified_records)} anomalie processate).")

if __name__ == "__main__":
    run_detection()