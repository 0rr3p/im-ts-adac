## Industrial Multijoint Time-Series Anomaly Detection And Classification (IM-TS-ADAC)
This project is a heavily modified and extended fork of the [time-series-autoencoder](https://github.com/JulesBelveze/time-series-autoencoder)
originally developed by [Jules Belveze](https://github.com/JulesBelveze).

Major modifications from the original repository include:

1.Implementation of a complete post-processing inference pipeline (ad.py).

2.Addition of FAISS to store latent embeddings and filter false positives.

3.Addition of a Random Forest layer to classify the reconstruction residuals.

4.Overhaul of the dataset and DataLoader to support multi-feature synchronization and post-training static scaling.

## Description IM-TS-ADAC

An end-to-end pipeline for anomaly detection and fault classification in industrial multi-joint robots. This project leverages an **LSTM-based AutoEncoder** to reconstruct time-series data, integrating a **FAISS vector database** to reduce false positives and a **Random Forest** classifier to automatically categorize mechanical faults based on reconstruction residuals.

## 🚀 Key Features

* **LSTM AutoEncoder with Attention:** Learns the normal mechanical behavior of robot joints (Velocity, Acceleration, Torque) and computes reconstruction errors.
* **Dynamic Thresholding:** Automatically computes the anomaly threshold (Z-Score: $\mu + 3\sigma$) during the training phase.
* **FAISS Vector Database (Memory):** Extracts the 32-dimensional latent representation of trajectories. When a "sane" but rare trajectory triggers a false positive, it is saved in FAISS. Future identical anomalies are instantly filtered out.
* **Random Forest Fault Classification:** Extracts 30 statistical features from the reconstruction residuals (MSE delta) to classify the specific type of mechanical fault.
* **Human-in-the-Loop:** An interactive CLI allows operators to review anomalies via Matplotlib, discard false positives (updating FAISS), and manually label new faults (retraining the Random Forest on the fly).

## 📁 Project Structure

* `ad.py`: The core anomaly detection and classification pipeline (Inference, FAISS query, RF classification).
* `run_reconstruction.py`: Training script for the AutoEncoder model.
* `run_ad.py`: Convenience script that calls `ad.py` in sequence for joints 1–6.
* `model.py`: PyTorch implementation of the Encoder/Decoder with optional Temporal Attention.
* `dataset.py`: Data preprocessing, scaling, and sliding-window generation.
* `train.py`: Training loop, LR scheduler, checkpoint/best-model saving.
* `eval.py`: Evaluation loop, per-window MSE stats, plots.
* `utils.py`: Checkpoint loading helper.
* `config.yaml`: Hydra configuration file to manage hyperparameters and file paths.

## ⚙️ Installation
The code supports CUDA gpu accelleration. So if extra performance is needed, once every library is installed, torch & torch vision can be upgraded to the version that supports gpu accelleration.

0. Remember to use the correct paths (data_path=training data, path_ad=execution data) in the examples/reconstruction/config.yaml file!

1. Install miniconda and create python 3.11 environment from "anaconda prompt" terminal (the name cobot it's used just as an example):
   ```bash
   conda create --name cobot python=3.11
   ```
1. Activate the environment and install git:
   ```bash
   conda activate cobot
   conda install git
   ```
1. Clone the repository:
   ```bash
   git clone https://github.com/0rr3p/im-ts-adac/
   cd im-ts-adac
   pip install -e .
   pip install -r requirements.txt
   ```

## 🛠️ Usage

This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration management: every
parameter can be changed directly in `config.yaml` or overridden from the CLI. This section
walks through how the pipeline works end-to-end, what every config parameter does, and how to
run training and anomaly detection.

### How the pipeline works

The project is a two-stage system for detecting anomalies in robot joint signals
(velocity, current/acceleration, temperature — or whatever three signals you feed it
per joint).

**Stage 1 — Training (`run_reconstruction.py`)**
For a given joint, an LSTM-based **autoencoder** is trained to reconstruct short
windows of the joint's signals. It learns what "normal" behaviour looks like. At
the end of training you get a checkpoint, a fitted feature scaler, and a
reconstruction-error statistic (mean `mu` and standard deviation `sigma`) computed
on the validation trajectories.

**Stage 2 — Anomaly detection (`ad.py` / `run_ad.py`)**
The trained model is loaded and run on new data. Any window whose reconstruction
error (MSE) exceeds `mu + k·sigma` is flagged as a candidate anomaly. For each
candidate:
- A **FAISS** similarity index checks whether this "shape" of anomaly has already
  been marked as a **known false positive** in a previous session — if so it's
  skipped automatically.
- Otherwise, a **Random Forest** classifier (trained on-the-fly from your past
  labels) suggests a probable fault class.
- Two plots are shown and you are asked, in the terminal, to confirm/label the
  window (or mark it as a false positive).
- Every decision is saved, so the FAISS index and the classifier get smarter with
  each session — this is a semi-supervised, human-in-the-loop workflow.

### Understanding `config.yaml`

#### `joint_id`
```yaml
joint_id: 1
```
Selects which joint you are working on. It's used to build the feature column
names dynamically: `feature_cols: ["j${joint_id}_v", "j${joint_id}_a", "j${joint_id}_t"]`
→ with `joint_id=1` this resolves to `j1_v`, `j1_a`, `j1_t`. Change this (or
override it on the command line) to train/detect on a different joint.

#### `data` block
```yaml
data:
  _target_: tsa.dataset.TimeSeriesDataset
  batch_size: 32
  data_path: "...QUERY_CSV_SANE_EXCEL.csv"
  index_col: "Timestamp"
  traj_col: "trajectory_id"
  feature_cols: ["j${joint_id}_v", "j${joint_id}_a", "j${joint_id}_t"]
  target_col: null
  train_size: 0.80
  prediction_window: 1
  seq_length: 128
  overlap: 0.125
  k: 3
  first_n_ignore: 0
  task:
    _target_: tsa.dataset.Tasks
    value: reconstruction
```
| Parameter | Meaning |
|---|---|
| `batch_size` | Batch size for both train and test DataLoaders. |
| `data_path` | CSV used for **training**. Must have `;` as separator and `,` as decimal (Italian locale) — this is hard-coded in `dataset.py`. |
| `index_col` | Column used as the DataFrame index (timestamp). |
| `traj_col` | Column that identifies which "trajectory" (episode/run) each row belongs to. Splitting into train/test happens **per trajectory**, not per row, so an entire trajectory stays together. |
| `feature_cols` | The 3 signal columns for the selected joint (auto-built from `joint_id`). |
| `target_col` | Only used for the `prediction` task; leave `null` for `reconstruction`. |
| `train_size` | Fraction of trajectories used for training (e.g. 0.80 = 80%). The rest become the test/validation set. |
| `prediction_window` | Only relevant for the `prediction` task (unused in reconstruction). |
| `seq_length` | Length (in samples) of the sliding window fed to the model. This is your main "how much history the model sees" knob. |
| `overlap` | Minimum fractional overlap between consecutive windows of the same trajectory. The dataset computes a stride that guarantees at least this much overlap while using as few windows as possible. Higher `overlap` → more windows, more compute, denser coverage. |
| `k` | Anomaly threshold multiplier: `threshold = mu + k·sigma`. Higher `k` → fewer, more severe anomalies flagged. |
| `first_n_ignore` | Number of initial timesteps of each window to exclude from the loss and from the anomaly-error computation (useful if the model always needs a short "warm-up" before its reconstruction becomes reliable). |
| `task` | `reconstruction` (autoencoder reconstructs the window) or `prediction` (not used by these two scripts). |

#### `training` block
```yaml
training:
  denoising: False
  scheduler: "cosine"
  directions: 1
  gradient_accumulation_steps: 1
  from_last_steps: 0.4
  hidden_size_encoder: 48
  hidden_size_decoder: ${training.hidden_size_encoder}
  input_att: False
  lr: 0.01
  lr_decay_every_n_epoch: 65
  num_epochs: 158
  output_size: 3
  reg1: False
  reg2: False
  clip: 1.0
  reg_factor1: 1e-4
  reg_factor2: 1e-5
  weight_decay: 1e-5
  seq_len: ${data.seq_length}
  first_n_ignore: ${data.first_n_ignore}
  max_grad_norm: ${training.clip}
  temporal_att: True
```
| Parameter | Meaning |
|---|---|
| `denoising` | If `True` (and `input_att: True`), injects small Gaussian noise into the input during training — regularization technique. Only implemented in the attention encoder. |
| `scheduler` | Learning-rate schedule: `"cosine"` (single cosine decay over all epochs), `"cosine_restarts"` (cosine with warm restarts every `num_epochs // 3` epochs), or `"step"` (halve LR every `lr_decay_every_n_epoch` epochs). |
| `directions` | Number of LSTM directions used when initializing hidden states (kept at 1 for a unidirectional LSTM). |
| `gradient_accumulation_steps` | Accumulate gradients over N batches before stepping the optimizer — use if you want a larger effective batch size than memory allows. |
| `from_last_steps` | Fraction of the **final** part of training during which the "best model" checkpointing is active (e.g. `0.4` = only in the last 40% of steps). This avoids saving an early, under-trained model as "best". |
| `hidden_size_encoder` / `hidden_size_decoder` | Size of the LSTM/attention hidden state (the bottleneck dimension). The decoder size defaults to the encoder size. |
| `input_att` | `True` → use the attention-based encoder (`AttnEncoder`, learns per-timestep feature attention). `False` → use the plain `Encoder` (simple LSTM). |
| `lr` | Initial learning rate for AdamW. |
| `lr_decay_every_n_epoch` | Only used by the `"step"` scheduler. |
| `num_epochs` | Total training epochs. |
| `output_size` | Number of output features — **must equal `len(feature_cols)`**, i.e. 3. |
| `reg1` / `reg2` | Enable L1 / L2 weight regularization added to the loss. |
| `clip` / `max_grad_norm` | Gradient-clipping norm (same value, `max_grad_norm` just mirrors `clip`). |
| `reg_factor1` / `reg_factor2` | Strength of the L1/L2 penalty when enabled. |
| `weight_decay` | AdamW weight decay. |
| `seq_len` | Mirrors `data.seq_length` (used inside the model). |
| `first_n_ignore` | Mirrors `data.first_n_ignore`. |
| `temporal_att` | `True` → the encoder output is compressed into the bottleneck using a learned attention-pooling layer (`TemporalAttnPooling`). `False` → simple mean-pooling over time. This is independent from `input_att`. |

#### `general` block
```yaml
general:
  do_eval: True
  do_train: True
  logging_steps: 50
  logging_steps_final: 4
  output_dir: "output"
  save_steps: 100
  eval_during_training: True
```
| Parameter | Meaning |
|---|---|
| `do_train` | If `True`, runs the training loop (`train.py`). |
| `do_eval` | If `True` **and** you also pass a `general.ckpt=<path>` override, the script skips training and only evaluates that checkpoint. |
| `eval_during_training` | Run periodic evaluation on the test set while training (needed to compute the running MSE/std used for the "best model" logic and for the anomaly threshold `mu`/`sigma`). |
| `logging_steps` | Evaluate/log every N optimizer steps, during the "normal" phase of training. |
| `logging_steps_final` | Same, but used once you enter the final `from_last_steps` window (finer-grained logging near the end, since that's when the best checkpoint is chosen). |
| `output_dir` | Directory (relative to Hydra's run folder, see the `hydra` block below) where checkpoints, plots and logs are written. |
| `save_steps` | Save a rolling `checkpoint-<step>.ckpt` every N steps (separate from the "best model" checkpoint). |

#### `path_ad`
```yaml
path_ad: "...QUERY_CSV_SANE_EXCEL.csv"
```
The CSV used **only by `ad.py`** for anomaly detection (inference). It can be the
same file as `data.data_path` or a different one (e.g. new/unseen production data
you want to screen for anomalies). Same format requirements (`;` separator, `,`
decimal, same `Timestamp`/`trajectory_id`/feature columns).

#### `hydra` block
```yaml
hydra:
  job:
    chdir: True
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${joint_id}...seq_l=${data.seq_length}...i_att=${training.input_att}...t_att=${training.temporal_att}...h_size=${training.hidden_size_encoder}...ovlap=${data.overlap}...schd=${training.scheduler}...lr=${training.lr}...clip=${training.max_grad_norm}
```
- `job.chdir: True` means Hydra **changes the current working directory** to a
  fresh, timestamped run folder before your script's `run()`/`run_detection()`
  function executes. That's why `output_dir: "output"` and
  `scaler_joint{id}.pkl` end up nested inside that folder rather than in your
  repo root.
- `sweep.dir` / `sweep.subdir` control the folder layout used for **multirun**
  (grid-search) executions. The subdirectory name encodes the joint id
  and all the key hyperparameters, so you can tell runs apart at a glance. This
  same folder structure is what `ad.py`'s `find_latest_artifacts()` function
  scans to locate the most recent trained model for a given joint.

### Stage 1 — Training the AutoEncoder

Train the model on healthy trajectories for a specific joint. The script will automatically
save the scaler, the best model checkpoints, and compute the dynamic threshold.

```bash
cd im-ts-adac/examples/reconstruction
python run_reconstruction.py joint_id=1
```

Any config value can be overridden on the command line, Hydra-style:
```bash
python run_reconstruction.py joint_id=1 training.num_epochs=100 training.lr=0.005 data.seq_length=64
```

**Train all six joints (multirun / grid search).** Hydra's `-m` flag lets you sweep over
comma-separated values. To train all six joints in sequence, each in its own output folder:
```bash
python run_reconstruction.py -m joint_id=1,2,3,4,5,6
```
You can sweep any other parameter the same way, e.g. a small hyperparameter grid:
```bash
python run_reconstruction.py -m joint_id=1,2,3,4,5,6 training.hidden_size_encoder=32,48,64 data.overlap=0.0,0.125
```
Each combination gets its own folder under
`multirun/<date>/<time>/<joint_id>...seq_l=...i_att=...t_att=...h_size=...ovlap=...schd=...lr=...clip=.../`,
matching the `hydra.sweep.subdir` pattern in the config.

> **Note:** Make sure `data_path` in `config.yaml` points to your training dataset.

**What happens under the hood:**
1. The seed is fixed (42) for reproducibility (Python, NumPy, PyTorch, cuDNN deterministic mode).
2. `TimeSeriesDataset` loads the CSV, splits trajectories into train/test, fits a
   `StandardScaler` **on the training data only**, and builds sliding-window DataLoaders.
3. The fitted scaler is saved as `scaler_joint{joint_id}.pkl` (needed later by `ad.py` to
   preprocess new data the same way).
4. It prints diagnostics: steps per epoch, total steps — useful for sanity checking before a
   long run.
5. The `AutoEncForecast` model, an `AdamW` optimizer and `MSELoss` are created.
6. If `general.do_train: True` → training runs (`train.py`); if instead `general.do_eval: True`
   and a `general.ckpt=<path>` override is given, it only evaluates that checkpoint.

**What you get after training**, inside each run's folder (`.../<joint_subdir>/output/`):
- `best_model.ckpt` — the checkpoint with the best `custom_score = MSE + 3·std`,
  only saved once training reaches the last `from_last_steps` fraction of steps.
  It also stores `mu` and `sigma` (mean/std of validation reconstruction error),
  which `ad.py` later uses to compute the anomaly threshold.
- `checkpoint-<step>.ckpt` — periodic rolling checkpoints (every `save_steps`).
- `best_model_results.csv` — one row summarizing the best run's hyperparameters
  and metrics (handy for comparing grid-search runs).
- `eval_results.txt` — latest evaluation report (MSE, std, residual mean, loss).
- `preds.png` — reconstruction plot for the first sequence of the test set.
- `targets.pt`, `predictions.pt`, `attentions.pt` — raw tensors from the last
  evaluation, in case you want to inspect them yourself.
- TensorBoard logs (`SummaryWriter`) — run `tensorboard --logdir <output_dir>`
  to visualize training/eval curves (`train/loss`, `train/lr`, `train/grad_norm`,
  `eval/MSE`, `eval/std`, `eval/custom_score`, …).

And in the run's root folder (one level up from `output/`): `scaler_joint{joint_id}.pkl`.

### Stage 2 — Anomaly Detection & Classification

Run the detection pipeline on unseen/mixed data. The script outputs anomaly charts,
queries the FAISS database for known false positives, and asks the user to label any
unclassified faults.

```bash
cd im-ts-adac/examples/reconstruction
python ad.py joint_id=1
```

Run it for all six joints in sequence:
```bash
python run_ad.py
```
This simply calls `python ad.py joint_id={1..6}` one after another as subprocesses —
convenient for screening every joint in one go.

> **Note:** Make sure `path_ad` in `config.yaml` points to your testing/production dataset,
> and that `joint_id` points to the joint you want to analyze (if calling `ad.py` directly
> without an override).

**What `ad.py` does, step by step:**
1. **Finds the model**: `find_latest_artifacts(joint_id)` scans
   `multirun/<date>/<time>/<joint_id>/` folders, newest first, and returns the
   first one that contains both `scaler_joint{id}.pkl` and
   `output/best_model.ckpt`. Make sure you've trained that joint at least once
   before running detection on it.
2. **Loads/creates support files** (all under a local `faiss/` folder, created
   next to `ad.py` if missing):
   - `rf_model_joint.joblib` — the Random Forest fault classifier.
   - `rf_data_joint.joblib` — the `{X, y}` history used to (re)train it.
   - `label_map_joint.joblib` — maps numeric class ids to human-readable label
     names. On first run, dummy placeholder classes (`Fuffa_Normale` /
     `Fuffa_Anomalo`) are created just so the classifier has something to start
     from; they are automatically removed once you've confirmed at least two
     real fault classes.
   - `raw_index_joint{id}.index` / `centroids_index_joint{id}.index` — FAISS
     indices of embeddings for windows you've previously marked as **false
     positives**, used to auto-skip similar-looking windows in future runs.
3. **Loads the trained autoencoder and scaler**, reads `mu`/`sigma` from the
   checkpoint, and computes the anomaly threshold:
   `threshold = mu + k · sigma` (falls back to a fixed `0.05` if `sigma` is 0).
4. **Runs inference** over `path_ad` (preprocessed with the *loaded* scaler, not
   refit), computing per-window MSE and encoder embeddings for every window of
   every trajectory.
5. **Summary plot**: if any window exceeds the threshold, a bar/stem plot of all
   anomalous windows' MSE is shown, plus a printed table.
6. **Per-trajectory loop**: for each trajectory,
   - windows are checked against the FAISS **centroid** index; a distance below
     `FAISS_SIMILARITY_THRESHOLD` (0.167, hard-coded near the top of `ad.py`)
     marks the window as a **known false positive** and it's skipped, updating
     that FP cluster's running centroid.
   - Windows below the anomaly threshold are logged as `esito = "normale"`.
   - Remaining (genuinely new) anomalous windows are shown one at a time:
     - the Random Forest prints a probability for each known fault class (if a
       model exists yet);
     - two plots pop up: (a) the whole trajectory for the 3 features with all
       anomalous windows shaded and the current one highlighted in black, and
       (b) a detailed zoom of the current window (real vs. reconstructed);
     - you are prompted in the terminal:
       ```
       Etichetta per ID <traj_id> finestra <win_idx> (classe RF, 'false' = falso positivo, 's' per saltare):
       ```
       - type an existing/new **class name or numeric id** → confirms/creates a
         fault label, adds the residual's statistical features to the RF
         training history, and retrains the Random Forest immediately;
       - type **`false`** → marks it as a false positive and updates the FAISS
         index so similar windows are auto-skipped next time;
       - press **Enter** or type **`s`** → skip without labeling (logged as
         `"saltata"`, nothing is learned from it).
7. **Saves outputs** (see table below).

**Semi-automatic mode.** Interactive labeling is convenient the first few times, but tedious
for batch screening. Three flags near the top of `ad.py` let you automate it — **edit the
script directly**, they are not exposed through `config.yaml`:
```python
FAISS_SIMILARITY_THRESHOLD = 0.167   # distance below which a window counts as "known FP"
RF_THRESHOLD = 0.99                  # confidence needed for auto-classification
AUTO_FALSE_POSITIVE = True           # if True: every new anomaly is auto-marked as FP (no prompt, no plots)
AUTO_ENTER = False                   # if True: every new anomaly is auto-skipped (like pressing Enter)
AUTO_CLASSIFY = False                # if True + RF confidence ≥ RF_THRESHOLD: auto-accept the RF's top-1 label
```
- With `AUTO_FALSE_POSITIVE = True` (the shipped default), any window that isn't
  already a known FP is *automatically* treated as a false positive — useful
  for bulk "sanity check" runs on data you already trust to be mostly normal,
  building up the FAISS FP index without manual clicking.
- Set `AUTO_FALSE_POSITIVE = False` (and optionally `AUTO_CLASSIFY = True`) once
  you actually want to review/confirm real anomalies interactively.
- Only one of these should really drive behaviour at a time — they are checked
  in this priority order: `AUTO_FALSE_POSITIVE` → `AUTO_ENTER` → interactive
  prompt (with `AUTO_CLASSIFY` silently pre-filling the label before the prompt
  is even reached, if confidence is high enough).

**Output files from `ad.py`:**

| File | Location | Content |
|---|---|---|
| `unified_anomaly_log.csv` | next to `ad.py` (fixed path, appended across all runs) | One row per analyzed window: dataset, joint, trajectory id, window index, MSE, `mu`/`sigma`/`threshold` used, outcome (`normale`, `FP_noto`, `FP_nuovo`, `saltata`, `confermata`, `auto_RF`), label, RF probabilities. `;`-separated, `,` decimal (Italian format) — open with Excel/pandas accordingly. |
| `residuals_joint{id}.npy` / `embeddings_joint{id}.npy` | current Hydra run folder (chdir'd) | Stacked residuals and encoder embeddings for windows you **confirmed** as real anomalies in this session. |
| `faiss/*.index`, `faiss/*.joblib` | `faiss/` next to `ad.py` | Updated FP index and RF classifier/history — persist and improve across sessions. |

### Quick reference

```bash
# Train joint 1 with default settings
python run_reconstruction.py joint_id=1

# Train joint 1 overriding a couple of hyperparameters
python run_reconstruction.py joint_id=1 training.lr=0.005 training.num_epochs=100

# Train all six joints (multirun)
python run_reconstruction.py -m joint_id=1,2,3,4,5,6

# Small grid search over hidden size and overlap, for every joint
python run_reconstruction.py -m joint_id=1,2,3,4,5,6 training.hidden_size_encoder=32,48,64 data.overlap=0.0,0.125

# Evaluate an existing checkpoint instead of training
python run_reconstruction.py joint_id=1 general.do_train=False general.do_eval=True general.ckpt=path/to/best_model.ckpt

# Run anomaly detection on a single joint
python ad.py joint_id=1

# Run anomaly detection on all six joints
python run_ad.py

# Monitor training curves
tensorboard --logdir multirun/<date>/<time>/<joint_subdir>/output
```

### Practical tips

- **Train before detecting.** `ad.py` will raise a `FileNotFoundError` for any
  joint that hasn't produced a `best_model.ckpt` + `scaler_joint{id}.pkl` yet in
  the `multirun/` tree.
- **`output_size` and `feature_cols` must stay in sync.** If you ever add/remove
  a signal per joint, update both `data.feature_cols` and `training.output_size`.
- **The threshold depends entirely on training data quality.** `k` (in
  `data.k`) controls how conservative detection is; if `ad.py` flags too many
  (or too few) windows, try tuning `k` and retraining, rather than only
  adjusting things at detection time.
- **`path_ad` vs `data_path`.** Keep them distinct if you want to train on a
  curated "known-healthy" dataset but screen a different (e.g. live/production)
  file for anomalies.
- **The Random Forest and FAISS index start "empty".** Expect the first
  detection sessions to require more manual labeling; the system gets faster
  and more automatic as `unified_anomaly_log.csv` and the `faiss/` folder
  accumulate confirmed examples.
- **CSV format.** Both training and detection CSVs must use `;` as the field
  separator and `,` as the decimal separator (this is hard-coded in
  `dataset.py`'s `pd.read_csv(..., sep=";", decimal=",")`).

---

## ORIGINAL README
<h1 align="center">LSTM-autoencoder with attentions for multivariate time series</h1>

<p align="center">
    <img src="https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2FJulesBelveze%2Ftime-series-autoencoder" alt="Hits">
  <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg">
</p>

This repository contains an autoencoder for multivariate time series forecasting.
It features two attention mechanisms described
in *[A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)*
and was inspired by [Seanny123's repository](https://github.com/Seanny123/da-rnn).

![Autoencoder architecture](autoenc_architecture.png)

## Download and dependencies

To clone the repository please run:

```
git clone https://github.com/JulesBelveze/time-series-autoencoder.git
```

<details>

<summary>Use uv</summary>

Then install `uv` 
```shell
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # linux/mac
# or
brew install uv  # mac with homebrew
```

# setup environment and install dependencies
```bash
cd time-series-autoencoder
uv venv
uv pip sync pyproject.toml
```

</details>

<details>
<summary>Install directly from requirements.txt</summary>

```shell
pip install -r requirements.txt
```

</details>

## Usage

The project uses [Hydra](https://hydra.cc/docs/intro/) as a configuration parser. You can simply change the parameters
directly within your `.yaml` file or you can override/set parameter using flags (for a complete guide please refer to
the docs).

```
python3 main.py -cn=[PATH_TO_FOLDER_CONFIG] -cp=[CONFIG_NAME]
```

Optional arguments:

```  
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --output-size OUTPUT_SIZE
                        size of the ouput: default value to 1 for forecasting
  --label-col LABEL_COL
                        name of the target column
  --input-att INPUT_ATT
                        whether or not activate the input attention mechanism
  --temporal-att TEMPORAL_ATT
                        whether or not activate the temporal attention
                        mechanism
  --seq-len SEQ_LEN     window length to use for forecasting
  --hidden-size-encoder HIDDEN_SIZE_ENCODER
                        size of the encoder's hidden states
  --hidden-size-decoder HIDDEN_SIZE_DECODER
                        size of the decoder's hidden states
  --reg-factor1 REG_FACTOR1
                        contribution factor of the L1 regularization if using
                        a sparse autoencoder
  --reg-factor2 REG_FACTOR2
                        contribution factor of the L2 regularization if using
                        a sparse autoencoder
  --reg1 REG1           activate/deactivate L1 regularization
  --reg2 REG2           activate/deactivate L2 regularization
  --denoising DENOISING
                        whether or not to use a denoising autoencoder
  --do-train DO_TRAIN   whether or not to train the model
  --do-eval DO_EVAL     whether or not evaluating the mode
  --data-path DATA_PATH
                        path to data file
  --output-dir OUTPUT_DIR
                        name of folder to output files
  --ckpt CKPT           checkpoint path for evaluation 
  ```

## Features

* handles multivariate time series
* attention mechanisms
* denoising autoencoder
* sparse autoencoder
