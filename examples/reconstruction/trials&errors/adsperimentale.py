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


# ===========================================================================
# COSTANTI
# ===========================================================================
FAISS_SIMILARITY_THRESHOLD = 0.1
RF_THRESHOLD = 0.90
HIDDEN_SIZE_ENCODER = 32          # dimensione latente per singolo giunto
N_JOINTS = 6                      # numero totale di giunti
FEATURES_PER_JOINT = 3            # j_v, j_a, j_t
GLOBAL_EMBEDDING_DIM = HIDDEN_SIZE_ENCODER * N_JOINTS   # 32 * 6 = 192

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Pulizia argv per Hydra (invariata)
# ---------------------------------------------------------------------------
def get_joint_and_clean_argv():
    joint = "0"
    filtered_argv = []
    for arg in sys.argv:
        if arg.startswith("--j") and arg[3:].isdigit():
            joint = arg[3:]
        else:
            filtered_argv.append(arg)
    sys.argv = filtered_argv
    return joint

SELECTED_JOINT = get_joint_and_clean_argv()


# ---------------------------------------------------------------------------
# find_latest_artifacts — invariata, usata 6 volte in fase di init
# ---------------------------------------------------------------------------
def find_latest_artifacts(joint_id):
    base_path = Path(__file__).resolve().parent / "multirun"
    if not base_path.exists():
        raise FileNotFoundError(f"Directory multirun non trovata in: {base_path}")

    for date_folder in sorted(base_path.iterdir(), reverse=True):
        if not date_folder.is_dir():
            continue
        for time_folder in sorted(date_folder.iterdir(), reverse=True):
            if not time_folder.is_dir():
                continue
            target_dir = time_folder / str(joint_id)
            if not target_dir.exists() or not target_dir.is_dir():
                continue
            scaler_path = target_dir / f"scaler_joint{joint_id}.pkl"
            model_path  = target_dir / "output" / "best_model.ckpt"
            if scaler_path.exists() and model_path.exists():
                return str(scaler_path), str(model_path), target_dir

    raise FileNotFoundError(
        f"Nessun file trovato per il giunto {joint_id} nelle cartelle di output."
    )


# ---------------------------------------------------------------------------
# Estrazione feature statistiche dal residuo (invariata)
# ---------------------------------------------------------------------------
def extract_statistical_features(residual_60x3):
    """
    Estrae 10 feature per ognuno dei 3 canali = 30 feature totali.
    Input: array (seq_len, 3)
    """
    features = []
    for j in range(FEATURES_PER_JOINT):
        channel = residual_60x3[:, j]
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(np.max(channel))
        features.append(np.min(channel))
        features.append(np.max(channel) - np.min(channel))
        features.append(np.sqrt(np.mean(channel ** 2)))
        features.append(np.median(channel))
        features.append(np.var(channel))
        features.append(np.sum(np.abs(np.diff(channel))))
        features.append(np.max(np.abs(channel)))
    return np.array(features)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
@hydra.main(config_path="./", config_name="config")
def run_detection(cfg):

    # -----------------------------------------------------------------------
    # 0. Percorsi FAISS — un unico indice globale a 192 dim
    # -----------------------------------------------------------------------
    FAISS_DIR = Path(__file__).resolve().parent / "faiss"
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    index_path = str(FAISS_DIR / "faiss_index_global.index")

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"✅ Database FAISS globale caricato da: {index_path}")
    else:
        index = faiss.IndexFlatL2(GLOBAL_EMBEDDING_DIM)
        print(f"🆕 Nuovo database FAISS globale creato ({GLOBAL_EMBEDDING_DIM} dim)")

    RF_PATH        = FAISS_DIR / "rf_model_global.joblib"
    DATA_XY_PATH   = FAISS_DIR / "rf_data_global.joblib"
    LABEL_MAP_PATH = FAISS_DIR / "label_map_global.joblib"

    rf_model  = joblib.load(RF_PATH)        if RF_PATH.exists()        else None
    history   = joblib.load(DATA_XY_PATH)   if DATA_XY_PATH.exists()   else {"X": [], "y": []}
    label_map = joblib.load(LABEL_MAP_PATH) if LABEL_MAP_PATH.exists() else {}

    plt.switch_backend('TkAgg')

    # -----------------------------------------------------------------------
    # 1. Caricamento multi-giunto: modelli, scaler, soglie
    # -----------------------------------------------------------------------
    models     = {}   # {1: AutoEncForecast, ..., 6: AutoEncForecast}
    scalers    = {}   # {1: StandardScaler,  ..., 6: StandardScaler}
    thresholds = {}   # {1: float,           ..., 6: float}

    print("\n--- Caricamento artefatti per tutti i giunti ---")
    for jid in range(1, N_JOINTS + 1):
        try:
            path_scaler, path_ckpt, folder = find_latest_artifacts(jid)
        except FileNotFoundError as e:
            print(f"  ⚠️  Giunto {jid}: {e}")
            continue

        scalers[jid] = joblib.load(path_scaler)

        ckpt  = torch.load(path_ckpt, map_location=device)
        mu    = float(ckpt.get('mu',    0.0))
        sigma = float(ckpt.get('sigma', 0.0))
        thresholds[jid] = mu + 3 * sigma if sigma != 0 else 0.05

        m = AutoEncForecast(cfg.training, input_size=FEATURES_PER_JOINT).to(device)
        opt_tmp = torch.optim.Adam(m.parameters(), lr=cfg.training.lr)
        m, _, _, _ = load_checkpoint(path_ckpt, m, opt_tmp, device)
        m.eval()
        models[jid] = m

        print(f"  ✅ Giunto {jid} | μ={mu:.4f} σ={sigma:.4f} → soglia={thresholds[jid]:.4f}")

    loaded_joints = sorted(models.keys())
    print(f"\nGiunti caricati: {loaded_joints}")

    # -----------------------------------------------------------------------
    # 2. Dataset e DataLoader — tutte e 18 le feature in un unico loader
    # -----------------------------------------------------------------------
    all_feature_cols = [
        f"j{jid}_{feat}"
        for jid in range(1, N_JOINTS + 1)
        for feat in ["v", "a", "t"]
    ]

    cfg.data.data_path    = cfg.path_ad
    cfg.data.feature_cols = all_feature_cols

    ts = instantiate(cfg.data)

    malati_ids = ts.data[ts.traj_col].unique()

    def scale_multijoints(raw_array):
        """
        raw_array: (n_obs, 18)
        Scala ogni blocco di 3 feature con lo scaler del relativo giunto.
        Restituisce array (n_obs, 18).
        """
        scaled_blocks = []
        for idx, jid in enumerate(loaded_joints):
            col_start = idx * FEATURES_PER_JOINT
            col_end   = col_start + FEATURES_PER_JOINT
            block = raw_array[:, col_start:col_end]
            scaled_blocks.append(scalers[jid].transform(block))
        return np.hstack(scaled_blocks)

    raw_arrays = [
        scale_multijoints(
            ts.data[ts.data[ts.traj_col] == tid][all_feature_cols].values
        )
        for tid in malati_ids
    ]

    test_dataset = ts.frame_series(raw_arrays, malati_ids)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )

    criterion = nn.MSELoss(reduction='none')

    # -----------------------------------------------------------------------
    # 3. Loop di Inferenza Multi-Giunto con Late Fusion
    # -----------------------------------------------------------------------
    all_errors              = []
    all_traj_ids            = []
    anomaly_residuals_list  = []   # residuo del giunto peggiore, shape (1, seq, 3)
    anomaly_embeddings_list = []   # global_embedding 192-dim,   shape (1, 192)
    anomaly_info            = []

    print("\nInizio analisi dati anomali (multi-giunto)...")
    with torch.no_grad():
        for batch in test_loader:
            features_18, y_hist_18, target_18, batch_ids = [b.to(device) for b in batch]
            # shapes: [B, seq_len, 18]

            B = features_18.size(0)

            joint_errors     = np.zeros((B, len(loaded_joints)))
            joint_embeddings = []

            for idx, jid in enumerate(loaded_joints):
                col_start = idx * FEATURES_PER_JOINT
                col_end   = col_start + FEATURES_PER_JOINT

                feat_j   = features_18[:, :, col_start:col_end]   # (B, seq, 3)
                yhist_j  = y_hist_18[:, :, col_start:col_end]
                target_j = target_18[:, :, col_start:col_end]

                output_j = models[jid](feat_j, yhist_j)            # (B, seq, 3)

                loss_j = criterion(output_j, target_j)
                joint_errors[:, idx] = loss_j.mean(dim=(1, 2)).cpu().numpy()

                _, latent_seq = models[jid].encoder(feat_j)        # (B, seq, 32)
                latent_avg_j  = latent_seq.mean(dim=1)             # (B, 32)
                joint_embeddings.append(latent_avg_j.cpu().numpy())

            # Late Fusion: [B, 32] × 6 → [B, 192]
            global_embedding = np.hstack(joint_embeddings).astype('float32')

            # Anomalia a silos: basta che un giunto superi la sua soglia
            per_joint_anomalous = np.column_stack([
                joint_errors[:, idx] > thresholds[jid]
                for idx, jid in enumerate(loaded_joints)
            ])  # (B, 6)
            is_raw_anomalous = per_joint_anomalous.any(axis=1)

            error_global = joint_errors.mean(axis=1)   # MSE medio 6 giunti

            # Filtro FAISS su embedding 192-dim
            if index.ntotal > 0:
                distances, _ = index.search(global_embedding, 1)
                is_known_fp  = distances.flatten() < FAISS_SIMILARITY_THRESHOLD
            else:
                is_known_fp = np.zeros(B, dtype=bool)

            is_anomalous = is_raw_anomalous & (~is_known_fp)

            if is_anomalous.any():
                worst_joint_idx = np.argmax(joint_errors, axis=1)  # (B,)

                for b in np.where(is_anomalous)[0]:
                    wj   = worst_joint_idx[b]
                    wjid = loaded_joints[wj]
                    cs   = wj * FEATURES_PER_JOINT
                    ce   = cs + FEATURES_PER_JOINT

                    feat_w  = features_18[b, :, cs:ce].unsqueeze(0)
                    yhist_w = y_hist_18[b, :, cs:ce].unsqueeze(0)
                    tgt_w   = target_18[b, :, cs:ce].unsqueeze(0)
                    out_w   = models[wjid](feat_w, yhist_w)
                    delta   = (tgt_w - out_w).squeeze(0).cpu().numpy()  # (seq, 3)

                    anomaly_residuals_list.append(delta[np.newaxis])
                    anomaly_embeddings_list.append(global_embedding[b:b+1])
                    anomaly_info.append({
                        'id':          int(batch_ids[b].cpu().item()),
                        'mse':         float(error_global[b]),
                        'emb':         global_embedding[b],
                        'worst_joint': wjid,
                    })

            all_errors.extend(error_global.tolist())
            all_traj_ids.extend(batch_ids.cpu().numpy().tolist())

    # -----------------------------------------------------------------------
    # 4. Analisi Risultati e Feedback Operatore
    # -----------------------------------------------------------------------
    print(f"\nDetection completata. Anomalie rilevate: {len(anomaly_info)}")

    df_anomalies = pd.DataFrame(columns=['id', 'mse', 'worst_joint'])

    if anomaly_info:
        df_anomalies = pd.DataFrame(anomaly_info)

        avg_threshold = np.mean([thresholds[j] for j in loaded_joints])

        plt.figure(figsize=(12, 4))
        plt.plot(range(len(df_anomalies)), df_anomalies['mse'],
                 color='steelblue', linestyle='-', alpha=0.6, label='Andamento MSE')
        plt.stem(range(len(df_anomalies)), df_anomalies['mse'],
                 linefmt='steelblue', markerfmt='o', basefmt=" ")
        plt.axhline(y=avg_threshold, color='r', linestyle='--', label='Soglia media')
        plt.title("MSE delle finestre anomale — Multi-Giunto (Late Fusion)")
        plt.xticks(range(len(df_anomalies)), df_anomalies['id'].astype(int), rotation=45)
        plt.ylabel("MSE medio (6 giunti)")
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)

    print("\n--- ELENCO ANOMALIE RILEVATE ---")
    print(df_anomalies[['id', 'mse', 'worst_joint']].to_string(index=False))

    # Feedback operatore — embedding 192-dim salvato in FAISS
    user_sane = input(
        "\nInserisci gli ID delle traiettorie SANE da ignorare (es: 10,15) "
        "[Invio per saltare]: "
    )
    ids_to_ignore = (
        [int(i.strip()) for i in user_sane.split(",")]
        if user_sane.strip() else []
    )
    plt.close('all')

    if ids_to_ignore:
        embs_sani = np.array(
            [info['emb'] for info in anomaly_info if info['id'] in ids_to_ignore],
            dtype='float32'
        )
        if len(embs_sani) > 0:
            index.add(embs_sani)
            faiss.write_index(index, index_path)
            print(f"✅ FAISS aggiornato con {len(embs_sani)} embedding globali ({GLOBAL_EMBEDDING_DIM}-dim).")

    # -----------------------------------------------------------------------
    # 5. Classificazione Residui con Random Forest (invariata)
    # -----------------------------------------------------------------------
    if isinstance(history['X'], np.ndarray):
        history['X'] = history['X'].tolist()
    if isinstance(history['y'], np.ndarray):
        history['y'] = history['y'].tolist()

    true_anomalies_indices = [
        i for i, info in enumerate(anomaly_info) if info['id'] not in ids_to_ignore
    ]

    if true_anomalies_indices:
        print("\n--- Analisi Residui (Classificazione Guasti) ---")
        session_labels = {}
        all_residuals  = np.concatenate(anomaly_residuals_list, axis=0)

        for idx in true_anomalies_indices:
            info        = anomaly_info[idx]
            residuo     = all_residuals[idx]
            traj_id     = int(info['id'])
            worst_joint = info['worst_joint']
            stat_features = extract_statistical_features(residuo).reshape(1, -1)

            if traj_id in session_labels:
                ans = session_labels[traj_id]
                print(f"ID {traj_id}: Applicazione automatica etichetta '{ans}'")
                print(f"\n {'-'*55}")
            else:
                if rf_model:
                    probs      = rf_model.predict_proba(stat_features)[0]
                    best_idx   = np.argmax(probs)
                    best_class = rf_model.classes_[best_idx]
                    best_prob  = probs[best_idx]
                    best_label = label_map.get(best_class, f"Classe_{best_class}")

                    print(f"\nProbabilità predette per ID {traj_id} (giunto peggiore: J{worst_joint}):")
                    for class_idx, prob in sorted(zip(rf_model.classes_, probs), key=lambda x: -x[1]):
                        nome = label_map.get(class_idx, f"Classe_{class_idx}")
                        print(f"   {nome}: {prob*100:.1f}%")

                    if best_prob >= RF_THRESHOLD:
                        print(
                            f"✅ Classificazione automatica: '{best_label}' "
                            f"({best_prob*100:.1f}% >= {RF_THRESHOLD*100:.0f}%)"
                        )
                        session_labels[traj_id] = best_label
                        continue
                else:
                    print(f"\nNessun modello RF disponibile per ID {traj_id}.")

                # Grafico residuo del giunto peggiore
                wj_start = (worst_joint - 1) * FEATURES_PER_JOINT
                wj_cols  = all_feature_cols[wj_start: wj_start + FEATURES_PER_JOINT]

                fig, axs = plt.subplots(FEATURES_PER_JOINT, 1, figsize=(10, 8), sharex=True)
                for i in range(FEATURES_PER_JOINT):
                    axs[i].plot(residuo[:, i], color='tab:blue')
                    axs[i].set_ylabel(wj_cols[i] if i < len(wj_cols) else f"feat_{i}")
                    axs[i].grid(True, alpha=0.3)
                axs[0].set_title(f"Residuo ID {traj_id} — Giunto peggiore: J{worst_joint}")
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)

                ans = input(
                    f"\nEtichetta per ID {traj_id} "
                    f"(classe come appare nelle probabilità, 's' per saltare): "
                )
                print(f"\n {'-'*55}")
                plt.close(fig)

                if ans.lower() != 's' and ans.strip():
                    session_labels[traj_id] = ans

            if traj_id in session_labels:
                ans = session_labels[traj_id]
                if ans.isdigit():
                    l_id = int(ans)
                    if l_id not in label_map:
                        label_map[l_id] = f"Guasto_{l_id}"
                else:
                    l_id = next(
                        (k for k, v in label_map.items() if v == ans),
                        len(label_map)
                    )
                    label_map[l_id] = ans

                history['X'].append(stat_features.flatten())
                history['y'].append(l_id)

    if true_anomalies_indices and len(set(history['y'])) > 1:
        new_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        new_rf.fit(history['X'], history['y'])
        joblib.dump(new_rf,    RF_PATH)
        joblib.dump(history,   DATA_XY_PATH)
        joblib.dump(label_map, LABEL_MAP_PATH)
        print("✅ Random Forest aggiornato.")

    # -----------------------------------------------------------------------
    # 6. Salvataggio Finale
    # -----------------------------------------------------------------------
    if anomaly_info:
        indices_da_salvare = [
            i for i, info in enumerate(anomaly_info) if info['id'] not in ids_to_ignore
        ]
        if indices_da_salvare:
            all_res_concat = np.concatenate(anomaly_residuals_list, axis=0)
            all_emb_concat = np.concatenate(anomaly_embeddings_list, axis=0)
            np.save("residuals_global.npy", all_res_concat[indices_da_salvare])
            np.save("embeddings_global.npy", all_emb_concat[indices_da_salvare])
            print(f"📂 File .npy salvati con {len(indices_da_salvare)} anomalie confermate.")
        else:
            print("ℹ️  Nessuna anomalia confermata.")

    pd.DataFrame({'traj_id': all_traj_ids, 'mse': all_errors}).to_csv(
        "detection_results.csv", index=False
    )


if __name__ == "__main__":
    run_detection()