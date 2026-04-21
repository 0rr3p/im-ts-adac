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
    
    
    criterion = nn.MSELoss(reduction='none') # 'none' per avere l'errore per singolo sample

    # 4. Loop di Inferenza
    all_errors = []
    all_traj_ids = []
    anomaly_residuals_list = []
    anomaly_embeddings_list = []
    anomaly_info = [] # Per il feedback utente
    ids_to_ignore = [] 
    
    print("Inizio analisi dati anomali...")
    with torch.no_grad():
        for batch in test_loader:
            # Ora il batch ha 4 elementi
            features, y_hist, target, batch_ids = [b.to(device) for b in batch]       


            # Estrazione Embedding (Latent Space)
            _, latent_seq = model.encoder(features) 
            # Media temporale per ottenere vettore 1D per finestra
            latent_avg = latent_seq.mean(dim=1).cpu().numpy().astype('float32')

            
            output = model(features, y_hist)           
            loss = criterion(output, target)   
            
            error_torch = loss.mean(dim=(1, 2)).cpu().numpy()

            ##!! NOTA BENE, distances.flatten() < 0.001 <---QUESTA è LA SIMILARITY THRESHOLD PER FAISS!!##

    
            
            # Controllo Similarità con FAISS
            if index.ntotal > 0:
                distances, _ = index.search(latent_avg, 1)
                is_known_false_positive = distances.flatten() < FAISS_SIMILARITY_THRESHOLD
            else:
                is_known_false_positive = np.zeros(len(error_torch), dtype=bool)

            
            # FILTRO: quali finestre di questo batch sono anomale?
            is_anomalous = (error_torch > THRESHOLD)  & (~is_known_false_positive)

            # Se ci sono anomalie, salviamo il residuo intero (il "delta")
            if is_anomalous.any():
                delta = target - output # [batch, seq_len, features]
                emb_anomali = latent_avg[is_anomalous]
                ids_anomali = batch_ids.cpu().numpy()[is_anomalous]
                err_anomali = error_torch[is_anomalous]
                # Prendiamo solo le fette "True" della maschera
                anomaly_residuals_list.append(delta[is_anomalous].cpu().numpy())
                anomaly_embeddings_list.append(emb_anomali)

                delta_np = delta[is_anomalous].cpu().numpy()  # già calcolato sopra
                for i_id, i_err, i_emb, i_res in zip(ids_anomali, err_anomali, emb_anomali, delta_np):
                    anomaly_info.append({
                        'id':      i_id,
                        'mse':     i_err,
                        'emb':     i_emb,
                        'residual': i_res   # shape (seq_len, features) — già allineato
                    })
                        
            all_errors.extend(error_torch)
            all_traj_ids.extend(batch_ids.cpu().numpy())

        # 5. Analisi Risultati
        # 5. Feedback Utente e Aggiornamento FAISS
        print(f"\nDetection completata. Anomalie rilevate: {len(anomaly_info)}")

        df_anomalies = pd.DataFrame(columns=['id', 'mse'])

        
        if anomaly_info:
            df_anomalies = pd.DataFrame(anomaly_info)
    
            plt.figure(figsize=(12, 4))
            
            # 1. Aggiungiamo la linea continua che collega i punti
            plt.plot(range(len(df_anomalies)), df_anomalies['mse'], color='steelblue', linestyle='-', alpha=0.6, label='Andamento MSE')
            
            # 2. Mantieni lo stem per evidenziare i singoli segmenti
            plt.stem(range(len(df_anomalies)), df_anomalies['mse'], linefmt='steelblue', markerfmt='o', basefmt=" ")
            
            # 3. Resto delle configurazioni
            plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Soglia Anomalia')
            plt.title(f"MSE delle finestre anomale - Giunto {SELECTED_JOINT}")
            plt.xticks(range(len(df_anomalies)), df_anomalies['id'].astype(int), rotation=45)
            plt.ylabel("MSE")
            plt.grid(axis='y', alpha=0.3) # Un po' di griglia aiuta a leggere i valori
            plt.legend()
            
            plt.show(block=False)
            plt.pause(0.1) # Diamo tempo al backend di renderizzare la finestra

                        
        print("\n--- ELENCO ANOMALIE RILEVATE ---")
        print(df_anomalies[['id', 'mse']].to_string(index=False))
        
        # B. Input per FAISS (Traiettorie Sane)
        user_sane = input("\nInserisci gli ID delle traiettorie SANE da ignorare (es: 10,15) [Invio per saltare]: ")
        ids_to_ignore = [int(i.strip()) for i in user_sane.split(",")] if user_sane.strip() else []

        plt.close('all')
        
        # Aggiornamento FAISS
        if ids_to_ignore:
            embs_sani = [info['emb'] for info in anomaly_info if info['id'] in ids_to_ignore]
            index.add(np.array(embs_sani).astype('float32'))
            faiss.write_index(index, index_path)
            print("✅ FAISS aggiornato.")

       

        # --- AGGIUNGI QUESTE RIGHE PER SICUREZZA ---
        if isinstance(history['X'], np.ndarray):
            history['X'] = history['X'].tolist()
        if isinstance(history['y'], np.ndarray):
            history['y'] = history['y'].tolist()
        
        
        true_anomalies_indices = [i for i, info in enumerate(anomaly_info) if info['id'] not in ids_to_ignore]


        if true_anomalies_indices:
            print("\n--- Analisi Residui (Classificazione Guasti) ---")
            session_labels = {}

            for idx in true_anomalies_indices:
                info = anomaly_info[idx]
                residuo_singolo = info['residual']  # Shape (60, 3)
                traj_id = int(info['id'])
                # Estrazione 30 Feature Statistiche dal residuo
                stat_features = extract_statistical_features(residuo_singolo).reshape(1, -1)
                
                # 1. Verifichiamo se abbiamo già etichettato questo ID in questa sessione
                if traj_id in session_labels:
                    ans = session_labels[traj_id]
                    print(f"ID {traj_id}: Applicazione automatica etichetta '{ans}'")
                    print(f"\n ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
                else:
                    # Procedura standard: Mostra grafico e chiedi input
                    if rf_model:
                        probs = rf_model.predict_proba(stat_features)[0]
                        best_idx = np.argmax(probs)
                        best_class = rf_model.classes_[best_idx]
                        best_prob = probs[best_idx]
                        best_label = label_map.get(best_class, f"Classe_{best_class}")
                        
                        print(f"\n Probabilità predette per ID {traj_id}:")
                        
                        for class_idx, prob in sorted(zip(rf_model.classes_, probs), key=lambda x: -x[1]):
                            nome = label_map.get(class_idx, f"Classe_{class_idx}")
                            print(f"   {nome}: {prob*100:.1f}%")

                         # Classificazione automatica se supera la soglia
                        if best_prob >= RF_THRESHOLD:
                            print(f"✅ Classificazione automatica: '{best_label}' ({best_prob*100:.1f}% >= {RF_THRESHOLD*100:.0f}%)")
                            session_labels[traj_id] = best_label
                            
                        
                    else:
                        print(f"\n Nessun modello RF disponibile per ID {traj_id}.")
                    
                    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                    feature_names = cfg.data.feature_cols
                    for i in range(3):
                        axs[i].plot(residuo_singolo[:, i], color='tab:blue')
                        axs[i].set_ylabel(feature_names[i])
                        axs[i].grid(True, alpha=0.3)
                    
                    axs[0].set_title(f"ID: {traj_id}")  # rimosso Pred dal titolo
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.1)
                        
                    print(f"\n Ho analizzato ID {traj_id}. Guarda il grafico e le probabilità e")
                    ans = input(f"inserisci etichetta per ID {traj_id} (scrivi la classe come appare nelle probabilità, 's' per saltare): ")
                    print(f"\n ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
                    plt.close(fig)
                    
                    # Se l'utente ha inserito un'etichetta valida, la salviamo per i prossimi segmenti
                    if ans.lower() != 's' and ans.strip():
                        session_labels[traj_id] = ans

                # 2. Aggiornamento Storia e Mappa (se abbiamo un'etichetta valida)
                if traj_id in session_labels:
                    ans = session_labels[traj_id]
                    if ans.isdigit():
                        l_id = int(ans)
                        if l_id not in label_map: label_map[l_id] = f"Guasto_{l_id}"
                    else:
                        # Recupera ID esistente per quel nome o ne crea uno nuovo
                        l_id = next((k for k, v in label_map.items() if v == ans), len(label_map))
                        label_map[l_id] = ans
                    
                    history['X'].append(stat_features.flatten())
                    history['y'].append(l_id)

        # D. Ri-addestramento (se sono stati aggiunti nuovi dati)
        if true_anomalies_indices and len(set(history['y'])) > 1:
            new_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            new_rf.fit(history['X'], history['y'])
            joblib.dump(new_rf, RF_PATH)
            joblib.dump(history, DATA_XY_PATH)
            joblib.dump(label_map, LABEL_MAP_PATH)
            print("✅ Random Forest aggiornato con le nuove feature statistiche.")



        # --- 5.3 Salvataggio Finale (Dopo il feedback operatore) ---
        if anomaly_info:
            # Filtriamo solo i dati che NON sono stati ignorati dall'operatore
            indices_da_salvare = [i for i, info in enumerate(anomaly_info) if info['id'] not in ids_to_ignore]
            
            if indices_da_salvare:
                # Recuperiamo solo i residui e gli embedding "confermati"
                all_residuals_concat = np.stack([info['residual'] for info in anomaly_info])
                all_embeddings_concat = np.stack([info['emb'] for info in anomaly_info])

                final_residuals = all_residuals_concat[indices_da_salvare]
                final_embeddings = all_embeddings_concat[indices_da_salvare]
                
                # Salvataggio su disco
                np.save(f"residuals_joint{SELECTED_JOINT}.npy", final_residuals)
                np.save(f"embeddings_joint{SELECTED_JOINT}.npy", final_embeddings)
                
                print(f"📂 File .npy salvati con {len(indices_da_salvare)} anomalie confermate.")
            else:
                print("ℹ️ Nessuna anomalia confermata, i file .npy non sono stati aggiornati.")
        
    # Il CSV dei risultati totali rimane invariato (serve per avere il log di tutto)
    pd.DataFrame({'traj_id': all_traj_ids, 'mse': all_errors}).to_csv("detection_results.csv", index=False)

if __name__ == "__main__":
    run_detection()