
# Industrial Multijoint Time-Series Anomaly Detection (IM-TS-ADAC)
This project is a heavily modified and extended fork of the time-series-autoencoder originally developed by Jules Belveze.

Major modifications from the original repository include:
        
Implementation of a complete post-processing inference pipeline (ad.py).
        
Addition of FAISS to store latent embeddings and filter false positives.
        
Addition of a Random Forest layer to classify the reconstruction residuals.
        
Overhaul of the dataset and DataLoader to support multi-feature synchronization and post-training static scaling.

# Description IM-TS-ADAC

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
* `model.py`: PyTorch implementation of the Encoder/Decoder with optional Temporal Attention.
* `dataset.py`: Data preprocessing, scaling, and sliding-window generation.
* `config.yaml`: Hydra configuration file to manage hyperparameters and file paths.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name

## 🛠️ Usage
This project uses Hydra for configuration management. You can override parameters directly from the CLI.

1. Training the AutoEncoder
Train the model on healthy trajectories for a specific joint (e.g., Joint 1). The script will automatically save the scaler, the best model checkpoints, and compute the dynamic threshold.

Bash
python run_reconstruction.py joint_id=1
# Note: Ensure the path_ad in config.yaml points to your training dataset.


2. Anomaly Detection & Classification
Run the detection pipeline on unseen/mixed data. The script will output anomaly charts, query the FAISS database for known false positives, and ask the user to label any unclassified faults.

Bash
python ad.py --j1
# Note: Ensure the path_ad in config.yaml points to your testing dataset.









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

## Examples

You can find under the `examples` scripts to train the model in both cases:

* reconstruction: the dataset can be found [here](https://gist.github.com/JulesBelveze/99ecdbea62f81ce647b131e7badbb24a)
* forecasting: the dataset can be found [here](https://gist.github.com/JulesBelveze/e9997b9b0b68101029b461baf698bd72)
  
  
