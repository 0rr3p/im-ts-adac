import hydra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from hydra.utils import instantiate
from tsa import AutoEncForecast
from tsa.utils import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="./", config_name="config")
def run_detection(cfg):
    # 1. Configurazione Path e Parametri
    PATH_MALATI = "C:\\Users\\Carlo\\time-series-autoencoder\\examples\\reconstruction\\QUERY_CSV_TOTALE_EXCEL_FATTO_1MALATA.csv" # Inserisci il path reale
    PATH_SCALER = "C:\\Users\\Carlo\\time-series-autoencoder\\examples\\reconstruction\\outputs\\2026-04-13\\20-06-29\\scaler_joint1.pkl"
  
    THRESHOLD = 0.05 # Da regolare in base ai test sui dati sani
    
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
    
    if cfg.general.get("ckpt", False):
        model, _, _, _ = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
    else:
        print("ATTENZIONE: Nessun checkpoint specificato nel config!")
    
    model.eval()
    criterion = nn.MSELoss(reduction='none') # 'none' per avere l'errore per singolo sample

    # 4. Loop di Inferenza
    all_errors = []
    all_traj_ids = []
    
    print("Inizio analisi dati anomali...")
    with torch.no_grad():
        for batch in test_loader:
            # Ora il batch ha 4 elementi
            features, y_hist, target, batch_ids = [b.to(device) for b in batch]
            
            output = model(features, y_hist)
            loss = criterion(output, target)
            
            error_per_window = loss.mean(dim=-1).cpu().numpy()
            
            all_errors.extend(error_per_window)
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
  

if __name__ == "__main__":
    run_detection()