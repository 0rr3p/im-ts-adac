import matplotlib.pyplot as plt
import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

import faiss
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint
import os
import sys
from pathlib import Path


FAISS_SIMILARITY_THRESHOLD= 0.1
RF_THRESHOLD= 0.90
hidden_size_encoder= 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




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

 




@hydra.main(version_base="1.1",config_path="./", config_name="config")
def run_detection(cfg):

    SELECTED_JOINT = cfg.joint_id

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
        index = faiss.IndexFlatL2(hidden_size_encoder)
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
    _, test_scaled = ts.preprocess_with_loaded_scaler()
    
    # Creazione del dataset e loader per la detection
    malati_ids = ts.data[ts.traj_col].unique()
    test_dataset = ts.frame_series(test_scaled, malati_ids)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

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
        THRESHOLD = mu + (3 * sigma)
        print(f"Soglia Dinamica:      {THRESHOLD:.6f} (μ + 3σ)")
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
    ids_to_ignore  = set()   # falsi positivi confermati in questa sessione
    session_labels = {}       # traj_id -> etichetta RF confermata

    # Raccogliamo prima TUTTE le finestre con inferenza (senza feedback)
    # poi le raggruppiamo per traj_id e processiamo una alla volta.
    # Questo approccio mantiene il DataLoader invariato e gestisce batch misti.
    all_windows = []  # lista di dict con tutti i dati per finestra

    print("Inizio analisi dati anomali...")
    with torch.no_grad():
        for batch in test_loader:
            features, y_hist, target, batch_ids = [b.to(device) for b in batch]

            _, latent_seq = model.encoder(features)
            latent_avg = latent_seq.mean(dim=1).cpu().numpy().astype('float32')

            output = model(features, y_hist)
            loss   = criterion(output, target)
            error_torch = loss.mean(dim=(1, 2)).cpu().numpy()

            delta = (target - output).cpu().numpy()

            all_errors.extend(error_torch.tolist())
            all_traj_ids.extend(batch_ids.cpu().numpy().tolist())

            for i in range(len(error_torch)):
                all_windows.append({
                    'id':       int(batch_ids[i].item()),
                    'mse':      float(error_torch[i]),
                    'emb':      latent_avg[i],          # (hidden_size,)
                    'residual': delta[i],               # (seq_len, features)
                    'target':   target[i].cpu().numpy(),
                    'output':   output[i].cpu().numpy(),
                })

    # Raggruppiamo per traj_id mantenendo l'ordine di arrivo
    traj_windows: dict = OrderedDict()
    for w in all_windows:
        traj_windows.setdefault(w['id'], []).append(w)

    print(f"\nTraiettorie da analizzare: {len(traj_windows)}")

    # -----------------------------------------------------------------------
    # 5. Plot riepilogativo MSE (tutte le finestre anomale, prima del feedback)
    # -----------------------------------------------------------------------
    all_anom_windows_recap = [w for ws in traj_windows.values() for w in ws if w['mse'] > THRESHOLD]

    if all_anom_windows_recap:
        df_riepilogo = pd.DataFrame([{'id': w['id'], 'mse': w['mse']} for w in all_anom_windows_recap])
        plt.figure(figsize=(12, 4))
        plt.plot(range(len(df_riepilogo)), df_riepilogo['mse'], color='steelblue', linestyle='-', alpha=0.6, label='Andamento MSE')
        plt.stem(range(len(df_riepilogo)), df_riepilogo['mse'], linefmt='steelblue', markerfmt='o', basefmt=" ")
        plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Soglia Anomalia')
        plt.title(f"MSE finestre anomale - Giunto {SELECTED_JOINT}")
        plt.xticks(range(len(df_riepilogo)), df_riepilogo['id'].astype(int), rotation=45)
        plt.ylabel("MSE")
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        print("\n--- ELENCO FINESTRE ANOMALE RILEVATE ---")
        print(df_riepilogo[['id', 'mse']].to_string(index=False))
    else:
        print("Nessuna finestra anomala rilevata.")

    # -----------------------------------------------------------------------
    # 6. Loop per traj_id: check FAISS → anomalia → feedback → aggiorna FAISS
    # -----------------------------------------------------------------------
    confirmed_anomaly_infos = []   # finestre anomale confermate (per salvataggio .npy)

    for traj_id, windows in traj_windows.items():

        # --- 6a. Check FAISS su TUTTE le finestre di questo traj_id ---
        ##!! NOTA BENE: distances < FAISS_SIMILARITY_THRESHOLD è la similarity threshold !!##
        embs = np.array([w['emb'] for w in windows]).astype('float32')

        if index.ntotal > 0:
            distances, _ = index.search(embs, 1)
            # Se ALMENO UNA finestra è simile a un falso positivo noto, ignoriamo l'ID
            any_known_fp = np.any(distances.flatten() < FAISS_SIMILARITY_THRESHOLD)
        else:
            any_known_fp = False

        # Calcolo MSE medio della traiettoria per decidere se è anomala
        mse_values   = np.array([w['mse'] for w in windows])
        is_anomalous = np.any(mse_values > THRESHOLD)

        if any_known_fp:
            print(f"⏭️  ID {traj_id}: riconosciuto come falso positivo da FAISS, ignorato.")
            continue

        if not is_anomalous:
            continue  # Nessuna finestra sopra soglia, traiettoria sana

        # --- 6b. Traiettoria anomala: RF + plot + feedback ---

        # RF sulla prima finestra anomala
        first_anom = next(w for w in windows if w['mse'] > THRESHOLD)
        stat_features = extract_statistical_features(first_anom['residual']).reshape(1, -1)

        if traj_id not in session_labels and rf_model:
            probs      = rf_model.predict_proba(stat_features)[0]
            best_idx   = np.argmax(probs)
            best_class = rf_model.classes_[best_idx]
            best_prob  = probs[best_idx]
            best_label = label_map.get(best_class, f"Classe_{best_class}")

            print(f"\n Probabilità predette per ID {traj_id}:")
            for class_idx, prob in sorted(zip(rf_model.classes_, probs), key=lambda x: -x[1]):
                nome = label_map.get(class_idx, f"Classe_{class_idx}")
                print(f"   {nome}: {prob*100:.1f}%")

            if best_prob >= RF_THRESHOLD:
                print(f"✅ Classificazione automatica: '{best_label}' ({best_prob*100:.1f}% >= {RF_THRESHOLD*100:.0f}%)")
                session_labels[traj_id] = best_label

        elif traj_id not in session_labels:
            print(f"\n Nessun modello RF disponibile per ID {traj_id}.")

        # Plot concatenato (solo finestre anomale, per chiarezza)
        anom_windows = [w for w in windows if w['mse'] > THRESHOLD]
        target_concat = np.concatenate([w['target'] for w in anom_windows], axis=0)
        output_concat = np.concatenate([w['output'] for w in anom_windows], axis=0)
        n_windows = len(anom_windows)
        seq_len   = anom_windows[0]['target'].shape[0]

        fig, axs = plt.subplots(3, 1, figsize=(max(10, 4 * n_windows), 8), sharex=True)
        feature_names = cfg.data.feature_cols

        for i in range(3):
            axs[i].plot(target_concat[:, i], color='tab:blue',   label='Input reale')
            axs[i].plot(output_concat[:, i], color='tab:orange', linestyle='--', label='Ricostruito')
            for w in range(1, n_windows):
                axs[i].axvline(x=w * seq_len - 0.5, color='gray', linestyle=':', alpha=0.6)
            axs[i].set_ylabel(feature_names[i])
            axs[i].legend(loc='upper right', fontsize=8)
            axs[i].grid(True, alpha=0.3)

        title_suffix = f"({n_windows} finestre anomale)" if n_windows > 1 else ""
        axs[0].set_title(f"ID: {traj_id} {title_suffix}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        # --- 6c. Feedback utente ---
        if traj_id not in session_labels:
            print(f"\n Ho analizzato ID {traj_id}. Guarda il grafico e le probabilità e")
            ans = input(
                f"inserisci etichetta per ID {traj_id} "
                f"(classe RF, 'false' = falso positivo, 's' per saltare): "
            )
            print(f"\n ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
            plt.close(fig)

            if ans.strip().lower() == 'false':
                # Aggiorna FAISS con gli embedding delle sole finestre ANOMALE (come V8)
                ids_to_ignore.add(traj_id)
                embs_anom = np.array([w['emb'] for w in anom_windows]).astype('float32')
                index.add(embs_anom)
                faiss.write_index(index, index_path)
                print(f"✅ FAISS aggiornato: {len(embs_anom)} embedding anomali aggiunti per ID {traj_id}. "
                      f"(FAISS ora contiene {index.ntotal} vettori)")
                continue  # Niente RF, passa alla prossima traiettoria

            if ans.lower() != 's' and ans.strip():
                session_labels[traj_id] = ans
        else:
            plt.close(fig)

        # --- 6d. Aggiornamento storia RF ---
        if traj_id in session_labels:
            ans = session_labels[traj_id]
            if ans.isdigit():
                l_id = int(ans)
                if l_id not in label_map:
                    label_map[l_id] = f"Guasto_{l_id}"
            else:
                l_id = next((k for k, v in label_map.items() if v == ans), len(label_map))
                label_map[l_id] = ans

            for w in anom_windows:
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

if __name__ == "__main__":
    run_detection()