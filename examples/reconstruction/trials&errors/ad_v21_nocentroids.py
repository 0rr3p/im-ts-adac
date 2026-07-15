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


import faiss
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint
import sys
import subprocess

from pathlib import Path



FAISS_SIMILARITY_THRESHOLD= 0.15
RF_THRESHOLD= 0.90
AUTO_FALSE_POSITIVE = False 



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



def extract_statistical_features(residual_60x3):
    """
    Estrae 10 feature per ognuno dei 3 canali = 30 feature totali.
    Input: array (60, 3)
    """
    features = []
    for j in range(3): # Per ogni giunto/canale
        channel = residual_60x3[:, j]
        
        # 10 Statistiche core
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(np.max(channel))
        features.append(np.min(channel))
        features.append(np.max(channel) - np.min(channel)) # Range
        features.append(np.sqrt(np.mean(channel**2)))      # RMS
        features.append(np.median(channel))
        features.append(np.var(channel))
        features.append(np.sum(np.abs(np.diff(channel)))) # Total variation (vibrazioni)
        features.append(np.max(np.abs(channel)))          # Peak absolute
        
    return np.array(features)
    
def set_seed(seed: int = 42):
    # Python native
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # per multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # ← AGGIUNTO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
 
@hydra.main(version_base="1.1",config_path="./", config_name="config")
def run_detection(cfg):

    set_seed(42)

    SELECTED_JOINT = cfg.joint_id

    N = cfg.training.first_n_ignore

    # Puntiamo alla cartella 'faiss' specifica che desideri
    FAISS_DIR = Path(__file__).resolve().parent  / "faiss"
    
    # Creiamo la cartella se non esiste ancora
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Definiamo il path completo dell'indice
    index_path = str(FAISS_DIR / f"faiss_index_joint{SELECTED_JOINT}.index")
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"✅ Database FAISS caricato da: {index_path}")
    else:
        index = faiss.IndexFlatL2(cfg.training.hidden_size_encoder)
        print(f"🆕 Nuovo database FAISS creato in: {index_path}")
    
    # --- Configurazione Percorsi ---
    # Carichiamo modello e dati storici se esistono
    # C. Classificazione Residui con Random Forest
    
    
    RF_PATH = FAISS_DIR / f"rf_model_joint{SELECTED_JOINT}.joblib"
    if RF_PATH.exists():
        rf_model = joblib.load(RF_PATH)
        print(f"✅ Modello RF caricato: {RF_PATH}")
    else:
        rf_model = None
        
    DATA_XY_PATH = FAISS_DIR / f"rf_data_joint{SELECTED_JOINT}.joblib"
    if DATA_XY_PATH.exists():
        history = joblib.load(DATA_XY_PATH)
        print(f"✅ history caricato: {DATA_XY_PATH}")
    else:
        history = {"X": [], "y": []}
        
    LABEL_MAP_PATH = FAISS_DIR / f"label_map_joint{SELECTED_JOINT}.joblib"
    if LABEL_MAP_PATH.exists():
        label_map = joblib.load(LABEL_MAP_PATH)
        print(f"✅ Label map caricata: {LABEL_MAP_PATH}")
    else:
        label_map = {}
        
    print(f"DEBUG: Backend in uso -> {plt.get_backend()}") 
    # Se qui stampa ancora 'agg', il comando .use() è stato ignorato.

    # Forza il cambio di backend a runtime
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
    
    # Carichiamo lo scaler del training ed eseguiamo il preprocessing senza rifare il fit
    ts.load_scaler(PATH_SCALER)

    _, test_scaled, malati_ids = ts.preprocess_with_loaded_scaler()

    test_dataset = ts.frame_series(test_scaled, malati_ids)
    
    from tsa.dataset import seed_worker  # già definita in dataset.py
    
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
    
    # Carichiamo il checkpoint trovato
    model, _, _, _ = load_checkpoint(PATH_CKPT, model, optimizer, device)
    model.eval()
    
    #caricamento per threshold dinamica
    checkpoint = torch.load(PATH_CKPT)
    mu = checkpoint.get('mu', 0.0)
    if torch.is_tensor(mu): mu = mu.item()
    sigma = checkpoint.get('sigma', 0.0)
    if torch.is_tensor(sigma): sigma = sigma.item()

    print("\n" + "="*30)
    print("🔎 VERIFICA CARICAMENTO CHECKPOINT")
    print(f"Media Errore (mu):    {mu:.6f}")
    print(f"Deviazione (sigma):   {sigma:.6f}")
    
    # Calcolo della soglia dinamica Z-Score
    if sigma != 0:
        THRESHOLD = mu + (cfg.data.k * sigma)
        print(f"Soglia Dinamica:      {THRESHOLD:.6f} (μ + {cfg.data.k}σ)")
    else:
        THRESHOLD = 0.05 # Fallback manuale se il file è vecchio
        print(f"Soglia Statistica:    {THRESHOLD:.6f} (Default manuale)")
    print("="*30 + "\n")   
	
    criterion = nn.MSELoss(reduction='none')

    # --- Sicurezza tipi history ---
    if isinstance(history['X'], np.ndarray):
        history['X'] = history['X'].tolist()
    if isinstance(history['y'], np.ndarray):
        history['y'] = history['y'].tolist()

    # -----------------------------------------------------------------------
    # 4. Inferenza per traj_id
    # Invece di processare tutti i batch in una volta e poi fare il feedback,
    # processiamo tutte le finestre di un traj_id, facciamo subito il check
    # FAISS, e se anomalo chiediamo il feedback e aggiorniamo FAISS prima
    # di passare al traj_id successivo.
    # -----------------------------------------------------------------------
    from collections import OrderedDict

    all_errors   = []
    all_traj_ids = []

    # Stato sessione
    session_labels = {}       # (traj_id, win_idx) -> etichetta RF confermata

    # Raccogliamo prima TUTTE le finestre con inferenza (senza feedback)
    # poi le raggruppiamo per traj_id e processiamo una alla volta.
    # Questo approccio mantiene il DataLoader invariato e gestisce batch misti.
    all_windows = []  # lista di dict con tutti i dati per finestra
    _win_counter = {}  # contatore finestre per traj_id, per assegnare win_idx

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
                    'win_idx':  win_idx,   # indice progressivo della finestra dentro il traj_id
                    'mse':      float(error_torch[i]),
                    'start': int(batch_starts[i].item()),
                    'emb':      latent_avg[i],
                    'residual': delta[i],
                    'target':   target[i].cpu().numpy(),
                    'output':   output[i].cpu().numpy(),
                })

    # Raggruppiamo per traj_id mantenendo l'ordine di arrivo
    traj_windows: dict = OrderedDict()
    for w in all_windows:
        traj_windows.setdefault(w['id'], []).append(w)


    # -----------------------------------------------------------------------
    # 5. Plot riepilogativo MSE (tutte le finestre anomale, prima del feedback)
    # -----------------------------------------------------------------------
    all_anom_windows_recap = [w for ws in traj_windows.values() for w in ws if w['mse'] > THRESHOLD]

    
        
    if all_anom_windows_recap:
            # 1. Aggiungiamo 'finestra' (presa da 'win_idx') nella creazione del DataFrame
            df_riepilogo = pd.DataFrame([{'id': w['id'], 'finestra': w['win_idx'], 'mse': w['mse']} for w in all_anom_windows_recap])
            
            # Creiamo un'etichetta testuale combinata per l'asse X del grafico (es: "3-w0", "3-w1", "8-w0")
           
            x_labels = [f"{int(row['id'])}-w{int(row['finestra'])}" for _, row in df_riepilogo.iterrows()]
            plt.figure(figsize=(12, 4))
            plt.plot(range(len(df_riepilogo)), df_riepilogo['mse'], color='steelblue', linestyle='-', alpha=0.6, label='Andamento MSE')
            plt.stem(range(len(df_riepilogo)), df_riepilogo['mse'], linefmt='steelblue', markerfmt='o', basefmt=" ")
            plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Soglia Anomalia')
            plt.title(f"MSE finestre anomale - Giunto {SELECTED_JOINT}")
            
            # 2. Usiamo le nuove etichette combinate per l'asse X
            plt.xticks(range(len(df_riepilogo)), x_labels, rotation=45, fontsize=8)
            
            plt.ylabel("MSE")
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            focus_terminal()  # <--- aggiungi qui

    
            print("\n--- ELENCO FINESTRE ANOMALE RILEVATE ---")
            # 3. Aggiungiamo la colonna 'finestra' alla stampa testuale
            print(df_riepilogo[['id', 'finestra', 'mse']].to_string(index=False))



          
            ############################################################################
            # SALVATAGGIO DEI RISULTATI IN APPEND (mse_data.csv)
            # -----------------------------------------------------------------------
            csv_filename = Path(__file__).resolve().parent / "mse_data.csv" # <--- L'UNICA MODIFICA
            
            # Costruiamo il dizionario con l'esatta struttura richiesta
            # Usiamo all_anom_windows_recap se vuoi solo le anomalie, oppure all_windows per tutte le finestre
            records_mse = []
            for w in all_anom_windows_recap:
                records_mse.append({
                    'Joint_ID':           SELECTED_JOINT,
                    'id':                 w['id'],
                    'finestra':           w['win_idx'],
                    'mse':                w['mse'],
                    'AVG MSE TRAINING':   mu,     # Media estratta dal checkpoint
                    'AVG STD TRAINING':   sigma   # Deviazione standard estratta dal checkpoint
                })
            
            df_export = pd.DataFrame(records_mse)
            
            # Verifica se il file esiste già per gestire l'intestazione (header)
            file_exists = os.path.exists(csv_filename)
            
            # Salviamo in modalità append ('a')
            df_export.to_csv(
                csv_filename, 
                mode='a', 
                index=False, 
                header=not file_exists,  # Scrive i titoli solo la prima volta in assoluto
                sep=';',                 
                decimal=','
            )
            print(f"📊 Risultati di Joint {SELECTED_JOINT} salvati/accodati correttamente in '{csv_filename}'.")
        
            #################################################################################
            
            n_traiettorie_anomale = df_riepilogo['id'].nunique()
            n_finestre_anomale = len(all_anom_windows_recap)
            
            print(f"\nTraiettorie da analizzare: {len(traj_windows)}")
            print(f"Traiettorie anomale da analizzare: {n_traiettorie_anomale}")
            print(f"\nFinestre totali da analizzare: {len(all_windows)}")
            print(f"Finestre anomale totali da analizzare: {n_finestre_anomale}")
            print(f"{len(traj_windows)},{n_traiettorie_anomale},{len(all_windows)},{n_finestre_anomale}")
        
    else:
        print("Nessuna finestra anomala rilevata.")

    # -----------------------------------------------------------------------
    # 6. Loop per traj_id: check FAISS (per finestra) → feedback per finestra
    # -----------------------------------------------------------------------
    confirmed_anomaly_infos = []   # finestre anomale confermate (per salvataggio .npy)
    confirmed_non_fp_windows = []

    for traj_id, windows in traj_windows.items():

        # --- 6a. Filtraggio FAISS per singola finestra (NON per intera traiettoria) ---
        ##!! NOTA BENE: distances < FAISS_SIMILARITY_THRESHOLD è la similarity threshold !!##
        embs = np.array([w['emb'] for w in windows]).astype('float32')

        if index.ntotal > 0:
            distances, _ = index.search(embs, 1)
            is_known_fp_per_win = distances.flatten() < FAISS_SIMILARITY_THRESHOLD
        else:
            is_known_fp_per_win = np.zeros(len(windows), dtype=bool)

        # Annotiamo su ogni finestra se è un FP noto, ma NON scartiamo l'intera traiettoria
        for w, is_fp in zip(windows, is_known_fp_per_win):
            w['is_known_fp'] = bool(is_fp)

        # Finestre anomale al netto dei FP noti PER QUELLA SINGOLA FINESTRA
        anom_windows = [
            w for w in windows
            if w['mse'] > THRESHOLD and not w['is_known_fp']
        ]

        # Log delle finestre saltate per FP
        fp_skipped = [w for w in windows if w['mse'] > THRESHOLD and w['is_known_fp']]
        for w in fp_skipped:
            print(f"⏭️  ID {traj_id} finestra {w['win_idx']}: riconosciuta come FP da FAISS, ignorata.")

        if not anom_windows:
            continue  # Nessuna finestra anomala rimasta dopo il filtro FAISS

        # --- 6b. Ricostruzione traiettoria completa (per il mini-overview) ---
        feature_names = cfg.data.feature_cols
        all_wins_sorted = sorted(windows, key=lambda w: w['win_idx'])
        seq_len = anom_windows[0]['target'].shape[0]

       # DOPO
        T_total = max(w['start'] + seq_len for w in all_wins_sorted)
        full_target = np.zeros((T_total, all_wins_sorted[0]['target'].shape[1]), dtype=np.float32)
        for w in all_wins_sorted:
            full_target[w['start']: w['start'] + seq_len] = w['target']

       
        win_starts = {w['win_idx']: w['start'] for w in all_wins_sorted}


                # Costruzione full_output con win_starts (già calcolato sul target)
        # Dove le finestre si sovrappongono, si fa la media pesata per conteggio
        full_output_sum   = np.zeros((T_total, full_target.shape[1]), dtype=np.float64)
        full_output_count = np.zeros((T_total, 1), dtype=np.float64)
        
        for w in all_wins_sorted:
            t0 = win_starts[w['win_idx']]
            t1 = min(t0 + seq_len, T_total)
            actual_len = t1 - t0
            full_output_sum[t0:t1]   += w['output'][:actual_len]
            full_output_count[t0:t1] += 1
        
        # Evita divisione per zero (non dovrebbe succedere, ma per sicurezza)
        full_output_count = np.where(full_output_count == 0, 1, full_output_count)
        full_output = (full_output_sum / full_output_count).astype(np.float32)
        

        # Tutti gli intervalli anomali (per il mini-overview, evidenziamo tutte le finestre anomale)
        all_anom_intervals = [(win_starts[w['win_idx']], win_starts[w['win_idx']] + seq_len)
                              for w in anom_windows]

        # -----------------------------------------------------------------------
        # 6c. Loop per singola finestra anomala: RF + plot + feedback
        # -----------------------------------------------------------------------
        n_anom_total = len(anom_windows)
        print(f"\n{'='*50}")
        print(f"ID {traj_id}: {n_anom_total} finestre anomale da analizzare.")
        print(f"{'='*50}")


        for win_num, w in enumerate(anom_windows):
            win_idx = w['win_idx']
            print(f"\n--- ID {traj_id} | Finestra {win_num + 1}/{n_anom_total} (win_idx={win_idx}) | MSE={w['mse']:.6f} ---")

            # RF sulla singola finestra
            stat_features = extract_statistical_features(w['residual']).reshape(1, -1)

            win_key = (traj_id, win_idx)
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

            elif win_key not in session_labels:
                print(f" Nessun modello RF disponibile.")

            # --- Grafico: mini-overview (tutte le anomalie) + dettaglio finestra corrente ---
            fig = plt.figure(figsize=(14, 11), layout='tight')
            gs = gridspec.GridSpec(7, 1, figure=fig,
                                   height_ratios=[1, 1, 1, 0.3, 2, 2, 2],
                                   hspace=0.05)

            # Mini overview (righe 0-2): traiettoria completa con TUTTE le finestre anomale
            ax_mini = [fig.add_subplot(gs[i]) for i in range(3)]
            t_axis = np.arange(T_total)
            for i in range(3):
                ax_mini[i].plot(t_axis, full_target[:, i], color='tab:blue', lw=0.8)
                ax_mini[i].plot(t_axis, full_output[:, i], color='tab:orange',lw=0.8, linestyle='--', alpha=0.75, label='Ricostruito')
                # Tutte le finestre anomale in rosso chiaro
                for (t0, t1) in all_anom_intervals:
                    ax_mini[i].axvspan(t0, min(t1, T_total), color='red', alpha=0.15)
                # Finestra corrente evidenziata in arancione
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

            # Separatore visivo (riga 3)
            fig.add_subplot(gs[3]).axis('off')

     
            # Dettaglio finestra corrente (righe 4-6): target vs ricostruito
            ax_det = [fig.add_subplot(gs[4 + i]) for i in range(3)]
            
            # Calcolo dell'asse temporale tenendo conto dell'overlap configurato
            x_axis_det = np.arange(seq_len) + win_starts[win_idx]

            for i in range(3):
                # Usiamo x_axis_det per allineare temporalmente i dati
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
            focus_terminal()  # <--- aggiungi qui

            if win_key not in session_labels:
                if AUTO_FALSE_POSITIVE:
                    print(f" AUTO_FALSE_POSITIVE attivo: finestra {win_idx} di ID {traj_id} classificata come FP.")
                    emb_fp = np.array([w['emb']]).astype('float32')
                    index.add(emb_fp)
                    faiss.write_index(index, index_path)
                    plt.close(fig)
                    continue  # Passa alla finestra successiva
                print(f"\n Guarda il grafico e le probabilità per la finestra {win_num + 1}.")
                ans = input(
                    f"Etichetta per ID {traj_id} finestra {win_idx} "
                    f"(classe RF, 'false' = falso positivo, 's' per saltare): "
                )
                print(f"\n ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
                plt.close(fig)

                if ans.strip().lower() == 'false':
                    # Aggiunge a FAISS solo l'embedding di QUESTA finestra
                    emb_fp = np.array([w['emb']]).astype('float32')
                    index.add(emb_fp)
                    faiss.write_index(index, index_path)
                    print(f"✅ FAISS aggiornato: embedding finestra {win_idx} di ID {traj_id} aggiunto come FP. "
                          f"(FAISS ora contiene {index.ntotal} vettori)")
                    continue  # Passa alla finestra successiva

                confirmed_non_fp_windows.append({'Joint_ID': SELECTED_JOINT, 'id': traj_id, 'finestra': w['win_idx'], 'mse': w['mse'], 'AVG MSE TRAINING': mu, 'AVG STD TRAINING': sigma})

                if ans.lower() != 's' and ans.strip():
                    session_labels[win_key] = ans
            else:
                plt.close(fig)

            # --- 6d. Aggiornamento storia RF per questa finestra ---
            if win_key in session_labels:
                ans = session_labels[win_key]
                if ans.isdigit():
                    l_id = int(ans)
                    if l_id not in label_map:
                        label_map[l_id] = f"Guasto_{l_id}"
                else:
                    l_id = next((k for k, v in label_map.items() if v == ans), len(label_map))
                    label_map[l_id] = ans

                feat = extract_statistical_features(w['residual']).flatten()
                history['X'].append(feat)
                history['y'].append(l_id)
                confirmed_anomaly_infos.append(w)

    print(f"\nDetection completata.")

    # -----------------------------------------------------------------------
    # 7. Ri-addestramento RF
    # -----------------------------------------------------------------------
    if history['X'] and len(set(history['y'])) > 1:
        new_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        new_rf.fit(history['X'], history['y'])
        joblib.dump(new_rf, RF_PATH)
        joblib.dump(history, DATA_XY_PATH)
        joblib.dump(label_map, LABEL_MAP_PATH)
        print("✅ Random Forest aggiornato con le nuove feature statistiche.")

    # -----------------------------------------------------------------------
    # 8. Salvataggio .npy anomalie confermate
    # -----------------------------------------------------------------------
    if confirmed_anomaly_infos:
        final_residuals  = np.stack([w['residual'] for w in confirmed_anomaly_infos])
        final_embeddings = np.stack([w['emb']      for w in confirmed_anomaly_infos])
        np.save(f"residuals_joint{SELECTED_JOINT}.npy",  final_residuals)
        np.save(f"embeddings_joint{SELECTED_JOINT}.npy", final_embeddings)
        print(f"📂 File .npy salvati con {len(confirmed_anomaly_infos)} finestre anomale confermate.")
    else:
        print("ℹ️ Nessuna anomalia confermata, i file .npy non sono stati aggiornati.")
        
    # Il CSV dei risultati totali rimane invariato (serve per avere il log di tutto)
    pd.DataFrame({'traj_id': all_traj_ids, 'mse': all_errors}).to_csv("detection_results.csv", index=False)
    if confirmed_non_fp_windows:
        csv_confirmed_path = Path(__file__).resolve().parent / "mse_data_confirmed.csv"
        file_exists_confirmed = os.path.exists(csv_confirmed_path)
        pd.DataFrame(confirmed_non_fp_windows).to_csv(csv_confirmed_path, mode='a', index=False, header=not file_exists_confirmed, sep=';', decimal=',')
        print(f"📊 Anomalie confermate (non-FP) salvate in 'mse_data_confirmed.csv'.")

if __name__ == "__main__":
    run_detection()