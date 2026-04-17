import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import faiss
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint
import os
import sys
from pathlib import Path


FAISS_SIMILARITY_THRESHOLD= 0.1
hidden_size_encoder= 32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- LOGICA DI PULIZIA ARGOMENTI PER HYDRA ---
# Estraiamo il numero del giunto e PULIAMO sys.argv prima che Hydra lo veda
def get_joint_and_clean_argv():
    joint = "0" # Default
    filtered_argv = []
    for arg in sys.argv:
        if arg.startswith("--j") and arg[3:].isdigit():
            joint = arg[3:]
        else:
            filtered_argv.append(arg)
    # Sovrascriviamo sys.argv rimuovendo i flag personalizzati
    sys.argv = filtered_argv 
    return joint

# Variabile globale catturata all'avvio
SELECTED_JOINT = get_joint_and_clean_argv()


def find_latest_artifacts(joint_id):
    # Path relativo: ci troviamo in 'reconstruction', cerchiamo la cartella 'outputs'
    base_path = Path(__file__).resolve().parent / "outputs"
    
    if not base_path.exists():
        raise FileNotFoundError(f"Directory outputs non trovata in: {base_path}")

    # Scansione cronologica inversa (Date decrescenti)
    for date_folder in sorted(base_path.iterdir(), reverse=True):
        if not date_folder.is_dir(): continue
        
        # Scansione cronologica inversa (Ore decrescenti)
        for time_folder in sorted(date_folder.iterdir(), reverse=True):
            if not time_folder.is_dir(): continue
            
            scaler_name = f"scaler_joint{joint_id}.pkl"
            scaler_path = time_folder / scaler_name
            model_path = time_folder / "output" / "best_model.ckpt"

            if scaler_path.exists() and model_path.exists():
                return str(scaler_path), str(model_path), time_folder
    
    raise FileNotFoundError(f"Nessun file trovato per il giunto {joint_id} nelle cartelle di output.")

 
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



@hydra.main(config_path="./", config_name="config")
def run_detection(cfg):


    
    # 1. Configurazione Path e Parametri
    PATH_MALATI = "C:\\Users\\Carlo\\time-series-autoencoder\\examples\\reconstruction\\QUERY_CSV_TOTALE_EXCEL_FATTO_1MALATA.csv" # Inserisci il path reale
    
    try:
        PATH_SCALER, PATH_CKPT, FOLDER = find_latest_artifacts(SELECTED_JOINT)
        print(f"--- Rilevamento per Giunto {SELECTED_JOINT} ---")
        print(f"Sorgente: {FOLDER}")
    except Exception as e:
        print(f"Errore: {e}")
        return

    

    # 2. Preparazione Dataset
    cfg.data.data_path = PATH_MALATI
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
    sigma = checkpoint.get('sigma', 0.05)
    if torch.is_tensor(sigma): sigma = sigma.item()

    print("\n" + "="*30)
    print("🔎 VERIFICA CARICAMENTO CHECKPOINT")
    print(f"Media Errore (mu):    {mu:.6f}")
    print(f"Deviazione (sigma):   {sigma:.6f}")
    
    # Calcolo della soglia dinamica Z-Score
    if mu > 0:
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
            
            error_torch = loss.mean(dim=(1, 2))

            ##!! NOTA BENE, distances.flatten() < 0.001 <---QUESTA è LA SIMILARITY THRESHOLD PER FAISS!!##

    
            
            # Controllo Similarità con FAISS
            if index.ntotal > 0:
                distances, _ = index.search(latent_avg, 1)
                is_known_false_positive = distances.flatten() < FAISS_SIMILARITY_THRESHOLD
            else:
                is_known_false_positive = np.zeros(len(error_torch), dtype=bool)

            
            # FILTRO: quali finestre di questo batch sono anomale?
            is_anomalous = (error_torch.cpu().numpy() > THRESHOLD)  & (~is_known_false_positive)

            # Se ci sono anomalie, salviamo il residuo intero (il "delta")
            if is_anomalous.any():
                delta = target - output # [batch, seq_len, features]
                emb_anomali = latent_avg[is_anomalous]
                ids_anomali = batch_ids.cpu().numpy()[is_anomalous]
                err_anomali = error_torch[is_anomalous]
                # Prendiamo solo le fette "True" della maschera
                anomaly_residuals_list.append(delta[is_anomalous].cpu().numpy())
                anomaly_embeddings_list.append(emb_anomali)

                for i_id, i_err, i_emb in zip(ids_anomali, err_anomali, emb_anomali):
                    anomaly_info.append({'id': i_id, 'mse': i_err, 'emb': i_emb})
                
            all_errors.extend(error_torch.cpu().numpy())
            all_traj_ids.extend(batch_ids.cpu().numpy())

    # 5. Analisi Risultati
    # 5. Feedback Utente e Aggiornamento FAISS
    print(f"\nDetection completata. Anomalie rilevate: {len(anomaly_info)}")
    
    if anomaly_info:
        df_anomalies = pd.DataFrame(anomaly_info)
        print("\n--- ELENCO ANOMALIE RILEVATE ---")
        print(df_anomalies[['id', 'mse']].to_string(index=False))
        
        user_input = input("\nInserisci gli ID delle traiettorie SANE (da ignorare in futuro), separati da virgola [Premi Invio per saltare]: ")
        
        if user_input.strip():
            ids_to_ignore = [int(i.strip()) for i in user_input.split(",")]
            embeddings_to_add = []
            
            for info in anomaly_info:
                if info['id'] in ids_to_ignore:
                    embeddings_to_add.append(info['emb'])
            
            if embeddings_to_add:
                new_embs = np.array(embeddings_to_add).astype('float32')
                index.add(new_embs)
                faiss.write_index(index, index_path)
                print(f"✅ Database FAISS aggiornato con {len(new_embs)} nuovi pattern sani.")

        # Salvataggio Residui Finale (solo quelli rimasti anomali dopo il filtro FAISS)
        final_residuals = np.concatenate(anomaly_residuals_list, axis=0)
        np.save(f"residuals_joint{SELECTED_JOINT}.npy", final_residuals)
        
        # Salvataggio Embedding (per archivio o future analisi)
        final_embeddings = np.concatenate(anomaly_embeddings_list, axis=0)
        np.save(f"embeddings_joint{SELECTED_JOINT}.npy", final_embeddings)
        print(f"📂 File .npy aggiornati (Residuals: {final_residuals.shape}, Embeddings: {final_embeddings.shape})")

    pd.DataFrame({'traj_id': all_traj_ids, 'mse': all_errors}).to_csv("detection_results.csv")

if __name__ == "__main__":
    run_detection()