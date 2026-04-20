from enum import Enum
from typing import List, Optional
import math
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Tasks(Enum):
    prediction = "prediction"
    reconstruction = "reconstruction"


# Columns of interest for joint 1
J1_COLS = ["j1_v", "j1_a", "j1_t"]


class TimeSeriesDataset(object):
    def __init__(
        self,
        task: Tasks,
        data_path: str,
        index_col: str,
        traj_col: str,
        feature_cols: List[str],
        seq_length: int,
        batch_size: int,
        train_size: float = 0.8,
        prediction_window: int = 1,
        target_col: Optional[str] = None,
    ):
        """
        :param task:             name of the task (prediction or reconstruction)
        :param data_path:        path to the CSV file
        :param index_col:        column to use as index (e.g. "Timestamp")
        :param traj_col:         column identifying each trajectory (e.g. "trajectory_id")
        :param feature_cols:     list of feature columns to use (e.g. ["j1_v", "j1_a", "j1_t"])
        :param seq_length:       sliding-window length
        :param batch_size:       batch size for DataLoaders
        :param train_size:       fraction of trajectories used for training (split is per-trajectory)
        :param prediction_window: steps ahead to predict (only used for prediction task)
        :param target_col:       target column for prediction task (None for reconstruction)
        """
        self.task = task.value
        self.traj_col = traj_col
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.target_col = target_col

        # --- Load CSV (Italian locale: semicolon sep, comma decimals) ---
        self.data = pd.read_csv(
            data_path,
            sep=";",
            decimal=",",
            index_col=index_col,
        )

        # Keep only the trajectory id and the selected feature columns
        keep_cols = [traj_col] + feature_cols
        if target_col and target_col not in feature_cols:
            keep_cols.append(target_col)
        self.data = self.data[keep_cols]

        # Identify unique trajectories and split train/test at trajectory level
        traj_ids = self.data[traj_col].unique()
        n_train = max(1, int(len(traj_ids) * train_size))
        self.train_ids = traj_ids[:n_train]
        self.test_ids = traj_ids[n_train:]

        # Scaler fitted on training data only
        self.scaler = StandardScaler()

        if self.task == "prediction":
            self.y_scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def save_scaler(self, path: str):
        """Salva lo scaler fittato in un file .pkl."""
        joblib.dump(self.scaler, path)
        print(f"Scaler salvato in: {path}")

    def load_scaler(self, path: str):
        """Carica uno scaler salvato e lo assegna all'istanza."""
        self.scaler = joblib.load(path)
        print(f"Scaler caricato da: {path}")

    def preprocess_with_loaded_scaler(self):
        """
        Versione alternativa di preprocess_data che NON fa il fit,
        ma usa lo scaler caricato per trasformare i dati.
        """
        # Trattiamo tutto il dataset attuale come "test" (dati da valutare)
        traj_ids = self.data[self.traj_col].unique()
        arrays = self._get_traj_arrays(traj_ids)
        
        # Trasformazione usando lo scaler già esistente (caricato)
        scaled_data = [self.scaler.transform(a) for a in arrays]
        
        # Restituiamo una lista vuota per il train e i dati scalati per il test
        return [], scaled_data

    def _get_traj_arrays(self, traj_ids):
        """Return list of (X_array, y_array) per trajectory (already scaled)."""
        arrays = []
        for tid in traj_ids:
            subset = self.data[self.data[self.traj_col] == tid][self.feature_cols].values
            arrays.append(subset)
        return arrays

    def preprocess_data(self):
        """Fit scaler on training trajectories, transform all splits."""
        train_arrays = self._get_traj_arrays(self.train_ids)
        test_arrays = self._get_traj_arrays(self.test_ids)

        # Fit on concatenated training data
        train_concat = np.vstack(train_arrays)
        self.scaler.fit(train_concat)

        train_scaled = [self.scaler.transform(a) for a in train_arrays]
        test_scaled = [self.scaler.transform(a) for a in test_arrays]

        return train_scaled, test_scaled

    def frame_series(self, traj_list,traj_ids_list, min_overlap: float = 0.4):
        """
        Build sliding windows across all trajectories with dynamic stride.
        
        The stride is computed per-trajectory to guarantee at least
        min_overlap overlap between consecutive windows, while minimising
        the total number of windows.
        
        Windows are distributed uniformly along each trajectory.
        """
        features, y_hist, targets, ids = [], [], [], []
    
        max_stride = int(self.seq_length * (1 - min_overlap))  # e.g. 80 for seq=100, overlap=0.2
    
        for X, tid in zip(traj_list, traj_ids_list):
            nb_obs = X.shape[0]
            nb_features = X.shape[1]
        
            min_len = self.seq_length + self.prediction_window
            if nb_obs < min_len:
                continue
        
            usable_len = nb_obs - self.prediction_window
        
            n_windows = math.ceil((usable_len - self.seq_length) / max_stride) + 1
        
            if n_windows > 1:
                actual_stride = (usable_len - self.seq_length) / (n_windows - 1)
            else:
                actual_stride = 0
        
            # ✅ Niente +1: gli indici partono da 0
            start_indices = [round(k * actual_stride) for k in range(n_windows)]
        
            for i in start_indices:
                feat = torch.FloatTensor(X[i: i + self.seq_length, :])
                features.append(feat.unsqueeze(0))
                ids.append(tid)
        
                if self.task == "reconstruction":
                    if i == 0:
                        # ✅ Primo step: padding con zero-vector (nessun sample precedente)
                        pad   = torch.zeros(1, nb_features)
                        rest  = torch.FloatTensor(X[0: self.seq_length - 1, :])
                        yh    = torch.cat([pad, rest], dim=0)
                    else:
                        # ✅ Caso normale: shift di -1, completamente sicuro
                        yh = torch.FloatTensor(X[i - 1: i + self.seq_length - 1, :])
        
                    y_hist.append(yh.unsqueeze(0))
                    tgt = torch.FloatTensor(X[i: i + self.seq_length, :])
                    targets.append(tgt.unsqueeze(0))
        
        return TensorDataset(torch.cat(features), torch.cat(y_hist), torch.cat(targets), torch.tensor(ids))
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_loaders(self):
        """
        Preprocess and frame the dataset.

        :return: (train_iter, test_iter, nb_features)
        """
        train_scaled, test_scaled = self.preprocess_data()
        nb_features = len(self.feature_cols)

        train_dataset = self.frame_series(train_scaled, self.train_ids)
        test_dataset = self.frame_series(test_scaled, self.test_ids)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """Invert standardisation on model outputs."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()

        original_shape = predictions.shape
        # Flatten to 2-D for inverse_transform, then restore shape
        predictions_2d = predictions.reshape(-1, len(self.feature_cols))
        unscaled = self.scaler.inverse_transform(predictions_2d)
        return torch.Tensor(unscaled.reshape(original_shape))
