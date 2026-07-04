# AD_Benchmark_MIMII

Classical and Deep Learning Approaches for Industrial Sound Anomaly Detection on the MIMII Dataset

---

## 1. Project Overview

This repository contains the implementation and benchmark results for our comparative study of anomaly detection methods on industrial machine sounds from the MIMII dataset.
We evaluate:

- Classical machine learning baselines using handcrafted acoustic features.
- Reconstruction-based deep learning models (autoencoders).
- Embedding-based deep learning models using pretrained CNN14 features with k-nearest neighbors (kNN).

All methods are trained and evaluated under a common experimental protocol on three machine types: **fan**, **pump**, and **valve**.

---

## 2. Authors and Affiliation

| Name                | Role                  | ORCID                 |
| ------------------- | --------------------- | --------------------- |
| **Simin Mirzadeh**  | Co-author & Developer | `0009-0005-5132-9599` |
| **Muhammad Rahman** | Co-author & Developer | `0009-0006-3456-6004` |

University of Bamberg, Bamberg, Germany  
Supervised by **Prof. Dr. Jakob Abeßer**

This repository accompanies the paper:

> Mirzadeh, S., Rahman, M.  
> _Classical and Deep Learning Approaches for Industrial Sound Anomaly Detection on the MIMII Dataset_ (2026).

---

## 3. Research Question

> Do deep learning methods (autoencoders, pretrained embeddings) significantly outperform classical ML baselines (OCSVM, Isolation Forest) for unsupervised acoustic anomaly detection on industrial machine sounds from the MIMII dataset?

We further analyze:

- How model complexity relates to anomaly detection performance.
- Whether intermediate (truncated) CNN14 embeddings improve kNN‑based detection compared to full CNN14 features.

---

## 4. Methods

Six core models are benchmarked across three machine types (fan, pump, valve):

| #   | Model                          | Category            | Input Features        | Anomaly Score            |
| --- | ------------------------------ | ------------------- | --------------------- | ------------------------ |
| 1   | OCSVM                          | Classical ML        | MFCC + spectral stats | Negative decision value  |
| 2   | Isolation Forest               | Classical ML        | MFCC + spectral stats | Negative score_samples   |
| 3   | Dense Autoencoder              | DL – Reconstruction | Feature vectors       | MSE reconstruction error |
| 4   | CNN Autoencoder                | DL – Reconstruction | Log‑mel spectrograms  | MSE reconstruction error |
| 5   | Transformer Autoencoder        | DL – Reconstruction | Log‑mel spectrograms  | MSE reconstruction error |
| 6   | CNN14 + kNN (full / truncated) | Hybrid (DL Emb.)    | CNN14 embeddings      | Mean kNN distance        |

Classical models operate on handcrafted acoustic feature vectors (MFCCs, deltas, energy and spectral descriptors).
Deep models use Log‑mel spectrograms or CNN14 embeddings as inputs, and anomaly scores are computed from reconstruction errors or nearest‑neighbor distances.

---

## 5. Repository Structure

```text
AD_Benchmark_MIMII_v2/
│
├── run_all.py                          # Run all models, aggregate results to benchmark_summary.csv
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── utils/
│   ├── config.py                       # All hyperparameters and paths
│   ├── features.py                     # Audio feature extraction (MFCC, log-mel, rich features)
│   └── evaluate.py                     # Metrics: AUC, pAUC, Accuracy, Precision, Recall, F1, confusion matrix
│
├── Baseline/
│   ├── SVM/ocsvm.py                    # Model 1: OCSVM
│   └── IsolationForest/iforest.py      # Model 2: Isolation Forest
│
├── Autoencoder/
│   ├── Dense/dense_ae.py               # Model 3: Dense Autoencoder
│   ├── CNN/cnn_ae.py                   # Model 4: CNN Autoencoder
│   └── Transformer/transformer_ae.py   # Model 5: Transformer Autoencoder
│
├── Embedding/
│   └── CNN14_KNN/cnn14_knn.py          # Model 6: CNN14 + kNN (full and truncated variants)
│
├── data/MIMII/                         # Place the MIMII dataset here (not committed)
│   ├── fan/train/normal/
│   ├── fan/test/normal/ + anomaly/
│   ├── pump/...
│   └── valve/...
│
└── results/                            # Auto-generated CSV result files
    ├── dense_ae_all.csv                # Dense AE metrics per machine
    ├── cnn_ae_all.csv                  # CNN AE metrics per machine
    ├── transformer_ae_all.csv          # Transformer AE metrics per machine
    ├── iforest_all.csv                 # Isolation Forest metrics per machine
    ├── truncated_cnn14_pump.csv        # Example truncated CNN14 results (pump)
    ├── benchmark_summary.csv           # Summary comparison table
    └── benchmark_avg.csv               # Averaged metrics across machines
```

The CSV files in `results/` are directly used to generate the tables in the paper.

---

## 6. Setup

### 6.1 Clone

```bash
git clone https://github.com/dev-rahman/AD_Benchmark_MIMII_v2.git
cd AD_Benchmark_MIMII_v2
```

### 6.2 Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 6.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> On M1/M2 Macs, PyTorch can use Metal (MPS) for GPU acceleration if installed with the appropriate options.
> For CUDA‑based GPUs, install PyTorch with the correct CUDA version from https://pytorch.org.

---

## 7. Dataset: MIMII

The MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset contains normal and anomalous sounds from valves, pumps, fans, and slide rails recorded in real factory environments with background noise.
In this project, we use the **fan**, **pump**, and **valve** machines.

### 7.1 Download

1. Visit: https://zenodo.org/record/3384388
2. Download the `fan`, `pump`, and `valve` archives.
3. Extract into `data/MIMII/`:

```text
data/MIMII/
├── fan/
│   ├── train/normal/
│   └── test/
│       ├── normal/
│       └── anomaly/
├── pump/
└── valve/
```

> For fast experiments on laptops, a subset of ~30–40 train clips and ~10 normal + 10 anomalous test clips per machine is sufficient for the benchmark, but the code supports the full dataset.

---

## 8. Running Experiments

All models are trained exclusively on `train/normal` and evaluated on both normal and anomalous test recordings under the same train–test split.

### 8.1 Run all models and generate benchmark tables

```bash
python run_all.py --machine all --models all
```

This command:

- Trains and evaluates all configured models for fan, pump, and valve.
- Writes per‑model CSV files to `results/`.
- Aggregates metrics into `benchmark_summary.csv` and `benchmark_avg.csv`.

### 8.2 Run individual models

Classical baselines:

```bash
python Baseline/SVM/ocsvm.py --machine fan
python Baseline/IsolationForest/iforest.py --machine pump
python Baseline/IsolationForest/iforest.py --machine valve
```

Reconstruction‑based models:

```bash
python Autoencoder/Dense/dense_ae.py --machine valve
python Autoencoder/CNN/cnn_ae.py --machine fan
python Autoencoder/Transformer/transformer_ae.py --machine pump
```

Embedding‑based models (full and truncated CNN14):

```bash
python Embedding/CNN14_KNN/cnn14_knn.py --machine fan --k 5
python Embedding/CNN14_KNN/cnn14_knn.py --machine pump --k 5 --truncated True
```

### 8.3 Filter by paradigm

Run only classical models:

```bash
python run_all.py --models classical
```

Run only deep learning models:

```bash
python run_all.py --models dl
```

---

## 9. Evaluation Metrics

We follow common practice in industrial acoustic anomaly detection and DCASE Task 2 challenges.

Primary metrics:

- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measures ranking quality of anomaly scores.
- **pAUC**: Partial ROC-AUC in a low false‑positive region (e.g. up to 10% FPR), emphasizing performance under strict alarm budgets.

Additional threshold‑based metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Per‑model confusion matrices (TN, FP, FN, TP) and thresholds are stored in the CSV files under `results/`.  
These metrics are used directly in the tables and analysis sections of the paper.

---

## 10. Results

The tables below summarize the main comparison between classical ML and deep learning across machine types. Values are taken from the CSV files in `results/` (rounded to 4 decimals where appropriate).

### 10.1 Classical vs Deep Learning (main comparison)

| Machine | Method           | Category            | ROC-AUC | F1     |
| ------- | ---------------- | ------------------- | ------- | ------ |
| Fan     | OCSVM            | Classical ML        | 0.531   | 0.394  |
|         | Isolation Forest | Classical ML        | 0.5028  | 0.8811 |
|         | Dense AE         | DL – Reconstruction | 0.7105  | 0.8846 |
|         | CNN AE           | DL – Reconstruction | 0.5203  | 0.8806 |
|         | Transformer AE   | DL – Reconstruction | 0.5619  | 0.8824 |
|         | CNN14 + kNN      | Hybrid (DL Emb.)    | 0.5275  | 0.8806 |
|         | TruncCNN14 + kNN | Hybrid (DL Emb.)    | 0.5646  | 0.8806 |

| Machine | Method           | Category            | ROC-AUC | F1     |
| ------- | ---------------- | ------------------- | ------- | ------ |
| Pump    | OCSVM            | Classical ML        | 0.617   | 0.494  |
|         | Isolation Forest | Classical ML        | 0.5969  | 0.7203 |
|         | Dense AE         | DL – Reconstruction | 0.7783  | 0.7339 |
|         | CNN AE           | DL – Reconstruction | 0.5130  | 0.7047 |
|         | Transformer AE   | DL – Reconstruction | 0.6427  | 0.7000 |
|         | CNN14 + kNN      | Hybrid (DL Emb.)    | 0.6985  | 0.7286 |
|         | TruncCNN14 + kNN | Hybrid (DL Emb.)    | 0.7064  | 0.7082 |

| Machine | Method           | Category            | ROC-AUC | F1     |
| ------- | ---------------- | ------------------- | ------- | ------ |
| Valve   | OCSVM            | Classical ML        | 0.653   | 0.516  |
|         | Isolation Forest | Classical ML        | 0.5230  | 0.7054 |
|         | Dense AE         | DL – Reconstruction | 0.7288  | 0.7660 |
|         | CNN AE           | DL – Reconstruction | 0.5109  | 0.7056 |
|         | Transformer AE   | DL – Reconstruction | 0.5443  | 0.7061 |
|         | CNN14 + kNN      | Hybrid (DL Emb.)    | 0.7119  | 0.7412 |
|         | TruncCNN14 + kNN | Hybrid (DL Emb.)    | 0.6669  | 0.7193 |

The full metrics (including pAUC, accuracy, precision, recall, threshold, TN/FP/FN/TP) are available in:

- `results/dense_ae_all.csv` (Dense Autoencoder)
- `results/cnn_ae_all.csv` (CNN Autoencoder)
- `results/transformer_ae_all.csv` (Transformer Autoencoder)
- `results/iforest_all.csv` (Isolation Forest)
- `results/truncated_cnn14_pump.csv` and related truncated CNN14 files.

`benchmark_summary.csv` and `benchmark_avg.csv` aggregate these per‑model results into summary tables used in the paper.

---

## 11. Reproducibility

To reproduce the paper’s results:

1. Use the same MIMII machine categories (fan, pump, valve) and directory layout as described in Section 7.
2. Install dependencies with `requirements.txt`, using Python 3.
3. Run `python run_all.py --machine all --models all` to generate all CSV files under `results/`.
4. Use the CSV files (`*_all.csv`, `benchmark_summary.csv`, `benchmark_avg.csv`) to recreate the tables and plots in the manuscript.

Random seeds, hyperparameters, and evaluation settings are defined in `utils/config.py` and the individual model scripts. This configuration is fixed for all runs in the paper to ensure a consistent comparison.

---

## 12. Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{mirzadeh2026mimii,
  title       = {Classical and Deep Learning Approaches for Industrial Sound Anomaly Detection
                 on the MIMII Dataset},
  author      = {Mirzadeh, Simin and Rahman, Muhammad},
  year        = {2026},
  institution = {University of Bamberg},
  note        = {Supervised by Prof. Dr. Jakob Abe{\ss}er}
}
```

And the MIMII dataset:

```bibtex
@inproceedings{purohit2019mimii,
  author    = {Purohit, Harsh and Tanabe, Ryo and Ichige, Kenji and Endo, Takashi and Nikaido, Yuki and Suefusa, Kaori and Kawaguchi, Yohei},
  title     = {{MIMII} Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop},
  year      = {2019},
  doi       = {10.48550/arXiv.1909.09347},
  url       = {https://arxiv.org/abs/1909.09347}
}
```

---

## 13. License

This project is released under the MIT License.

See `LICENSE` for details.
