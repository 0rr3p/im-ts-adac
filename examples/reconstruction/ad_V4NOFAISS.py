import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint
import os
import sys
from pathlib import Path



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
    
    
    print("Inizio analisi dati anomali...")
    with torch.no_grad():
        for batch in test_loader:
            # Ora il batch ha 4 elementi
            features, y_hist, target, batch_ids = [b.to(device) for b in batch]       
            
            output = model(features, y_hist)           
            loss = criterion(output, target)   
            
            error_torch = loss.mean(dim=(1, 2))

            # FILTRO: quali finestre di questo batch sono anomale?
            is_anomalous = error_torch > THRESHOLD

            # Se ci sono anomalie, salviamo il residuo intero (il "delta")
            if is_anomalous.any():
                delta = target - output # [batch, seq_len, features]
                # Prendiamo solo le fette "True" della maschera
                anomaly_residuals_list.append(delta[is_anomalous].cpu().numpy())
            
            all_errors.extend(error_torch.cpu().numpy())
            all_traj_ids.extend(batch_ids.cpu().numpy())

    # 5. Analisi Risultati
    errors_np = np.array(all_errors)
    anomalies = errors_np > THRESHOLD
    
    print("-" * 30)
    print(f"Risultati Detection:")
    print(f"Finestre totali: {len(errors_np)}")
    print(f"Anomalie rilevate: {np.sum(anomalies)}")
    print(f"Percentuale anomalie: {100 * np.sum(anomalies) / len(errors_np):.2f}%")
    print(f"Errore Max: {np.max(errors_np):.6f} | Errore Medio: {np.mean(errors_np):.6f}")

    # Salvataggio opzionale per analisi grafica
    # Creiamo un DataFrame organizzato
    pd.DataFrame({'traj_id': all_traj_ids,'mse': all_errors}).to_csv("detection_results.csv")
  
    # --- 5.1 Salvataggio residui per Classificazione Supervisionata ---
    if anomaly_residuals_list:
        # Unisce i batch in un unico array [N, 60, 3]
        final_residuals = np.concatenate(anomaly_residuals_list, axis=0)
        
        # Definiamo un nome file che includa il giunto selezionato
        output_filename = f"residuals_joint{SELECTED_JOINT}.npy"
        
        # Salvataggio in formato binario NumPy
        np.save(output_filename, final_residuals)
        
        print(f"\n✅ Dataset salvato con successo!")
        print(f"File: {output_filename}")
        print(f"Shape: {final_residuals.shape} (Anomalie, Seq_Len, Features)")
    else:
        print("\nℹ️ Nessun residuo estratto (nessuna anomalia sopra soglia).")
        
if __name__ == "__main__":
    run_detection()