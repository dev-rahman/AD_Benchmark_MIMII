# AD_Benchmark_MIMII_v2

**Classical and Deep Learning Approaches for Industrial Sound Anomaly Detection on the MIMII Dataset**

---

## Authors

| Name                | Role                  | ORCID                 |
| ------------------- | --------------------- | --------------------- |
| **Simin Mirzadeh**  | Co-author & Developer | `0009-0005-5132-9599` |
| **Muhammad Rahman** | Co-author & Developer | `0009-0006-3456-6004` |

University of Bamberg, Bamberg, Germany
Supervised by **Prof. Dr. Jakob Abeßer**

---

## Research Question

> _Do deep learning methods (autoencoders, pretrained embeddings) significantly outperform
> classical ML baselines (OCSVM, Isolation Forest) for unsupervised acoustic anomaly detection?_

---

## Methods

Six models are benchmarked across three machine types (fan, pump, valve) from the MIMII dataset.

| #   | Model                       | Category            | Input Features              | Anomaly Score            |
| --- | --------------------------- | ------------------- | --------------------------- | ------------------------ |
| 1   | **OCSVM**                   | Classical ML        | 87-dim MFCC + spectral      | Neg. decision function   |
| 2   | **Isolation Forest**        | Classical ML        | 87-dim MFCC + spectral      | Neg. score_samples       |
| 3   | **Dense Autoencoder**       | DL – Reconstruction | 87-dim feature vector       | MSE reconstruction error |
| 4   | **CNN Autoencoder**         | DL – Reconstruction | Log-mel spectrogram 128×128 | MSE pixel reconstruction |
| 5   | **Transformer Autoencoder** | DL – Reconstruction | Log-mel sequence (T×N_MELS) | MSE token reconstruction |
| 6   | **CNN14 + kNN**             | Hybrid (DL Emb.)    | CNN14 2048-dim embedding    | Mean kNN distance        |

---

## Repository Structure

```
AD_Benchmark_MIMII_v2/
│
├── run_all.py                          ← Run all 6 models, generates benchmark_summary.csv
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── utils/
│   ├── config.py                       ← All hyperparameters and paths
│   ├── features.py                     ← Audio feature extraction (MFCC, log-mel, rich)
│   └── evaluate.py                     ← AUC, pAUC, F1, confusion matrix
│
├── Baseline/
│   ├── SVM/ocsvm.py                    ← Model 1: OCSVM
│   └── IsolationForest/iforest.py      ← Model 2: Isolation Forest
│
├── Autoencoder/
│   ├── Dense/dense_ae.py               ← Model 3: Dense Autoencoder
│   ├── CNN/cnn_ae.py                   ← Model 4: CNN Autoencoder
│   └── Transformer/transformer_ae.py   ← Model 5: Transformer Autoencoder
│
├── Embedding/
│   └── CNN14_KNN/cnn14_knn.py          ← Model 6: CNN14 + kNN (Hybrid)
│
├── data/MIMII/                         ← Place your dataset here (not in Git)
│   ├── fan/train/normal/
│   ├── fan/test/normal/ + anomaly/
│   ├── pump/...
│   └── valve/...
│
└── results/                            ← Auto-generated CSV result files
    ├── ocsvm_fan.csv
    ├── cnn14_knn_k5_fan.csv
    └── benchmark_summary.csv           ← Full comparison table
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/<dev-rahman>/AD_Benchmark_MIMII_v2.git
cd AD_Benchmark_MIMII_v2
```

### 2. Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **M1/M2 Mac:** PyTorch will automatically use Metal (MPS) for GPU acceleration.
> **CUDA:** Install PyTorch with the correct CUDA version from https://pytorch.org

### 4. Download the MIMII Dataset

1. Go to: https://zenodo.org/record/3384388
2. Download `fan`, `pump`, and `valve` zip files
3. Extract into `data/MIMII/`:

```
data/MIMII/
├── fan/
│   ├── train/normal/       ← .wav files for training
│   └── test/normal/        ← normal test clips
│       anomaly/            ← anomalous test clips
├── pump/...
└── valve/...
```

> **Tip (laptop users):** Use ~30–40 train clips + 10 normal + 10 anomaly per machine
> for a fast benchmark. The models work on any size subset.

---

## Running

### Run all 6 models on all machines (produces paper table)

```bash
python run_all.py --machine all --models all
```

### Run a single model on one machine

```bash
python Baseline/SVM/ocsvm.py --machine fan
python Baseline/IsolationForest/iforest.py --machine pump
python Autoencoder/Dense/dense_ae.py --machine valve
python Autoencoder/CNN/cnn_ae.py --machine fan
python Autoencoder/Transformer/transformer_ae.py --machine fan
python Embedding/CNN14_KNN/cnn14_knn.py --machine fan --k 5
```

### Run only classical ML models

```bash
python run_all.py --models classical
```

### Run only DL models

```bash
python run_all.py --models dl
```

---

## Evaluation Metrics

| Metric                            | Description                                 | Standard        |
| --------------------------------- | ------------------------------------------- | --------------- |
| **AUC-ROC**                       | Area under ROC curve — primary metric       | Standard        |
| **pAUC**                          | Partial AUC at 10% FPR, normalised to [0,1] | DCASE Challenge |
| **F1**                            | F1 score at optimal threshold               | Standard        |
| **Accuracy / Precision / Recall** | Threshold-based metrics                     | Standard        |

Results are saved per-model per-machine to `results/` as CSV files.
`run_all.py` additionally outputs `benchmark_summary.csv` and `benchmark_avg.csv`.

---

## Results Table

> Fill in after running all experiments.

| Method           | Category            | Fan AUC | Pump AUC | Valve AUC | Avg AUC |
| ---------------- | ------------------- | ------- | -------- | --------- | ------- |
| OCSVM            | Classical ML        | —       | —        | —         | —       |
| Isolation Forest | Classical ML        | —       | —        | —         | —       |
| Dense AE         | DL – Reconstruction | —       | —        | —         | —       |
| CNN AE           | DL – Reconstruction | —       | —        | —         | —       |
| Transformer AE   | DL – Reconstruction | —       | —        | —         | —       |
| CNN14 + kNN      | Hybrid (DL Emb.)    | —       | —        | —         | —       |

---

## Citation

```bibtex
@misc{mirzadeh2026mimii,
  title     = {Classical and Deep Learning Approaches for Industrial Sound Anomaly Detection
               on the MIMII Dataset},
  author    = {Mirzadeh, Simin and Rahman, Muhammad},
  year      = {2026},
  institution = {University of Bamberg},
  note      = {Supervised by Prof. Dr. Jakob Abe{\ss}er}
}
```

---

## References

- **MIMII Dataset**: Purohit et al., _MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection_, DCASE 2019. https://zenodo.org/record/3384388
- **CNN14 / PANNs**: Kong et al., _PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition_, IEEE/ACM TASLP 2020.
- **DCASE Challenge**: http://dcase.community

---

## License

MIT License — Copyright (c) 2026 Simin Mirzadeh and Muhammad Rahman.
See [LICENSE](LICENSE) for details.
