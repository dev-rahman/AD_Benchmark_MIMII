# CNN14 + kNN for Audio Anomaly Detection

This module implements a **hybrid** anomaly detector for machine-sound data. It uses a pretrained CNN14 model from PANNs as a fixed feature extractor and applies k-nearest neighbours (kNN) on the resulting embeddings to score anomalies. CNN14 in the PANNs ecosystem is commonly used as a pretrained audio model, and CNN14 embeddings are 2048-dimensional.[1][2]

## Method

The pipeline follows these steps:

1. Load a pretrained CNN14 model from `panns-inference`.[1]
2. Extract one 2048-dimensional embedding for each `.wav` file.[1][2]
3. Cache embeddings as `.npy` files to avoid repeated inference.
4. Fit a `StandardScaler` on normal training embeddings only.
5. Train a kNN index on the scaled normal training embeddings.
6. Score each test clip by the mean Euclidean distance to its `k` nearest normal neighbours.

Higher mean distance indicates that a test clip is farther from the normal training distribution and is therefore treated as more anomalous.

## File

- Script: `Embedding/CNN14_KNN/cnn14_knn.py`
- Cache directory: `Embedding/CNN14_KNN/.cache/`
- Default neighbours: `k = 5`

## Requirements

Install the main dependency first:

```bash
pip install panns-inference
```

This script also depends on the project utilities and the following Python packages being available in the environment:

- `numpy`
- `torch`
- `scikit-learn`
- `tqdm`

## Expected project utilities

The script imports the following project-specific helpers:

- `utils.config`: `DATA_ROOT`, `RESULTS_DIR`, `MACHINES`, `SAMPLE_RATE`, `get_device`
- `utils.features`: `wav_paths`, `load_audio`
- `utils.evaluate`: `evaluate`, `save_results`

These utilities are expected to provide:

- Dataset root paths and result paths.
- Supported machine names such as `fan`, `pump`, and `valve`.
- Audio loading at the project sample rate.
- Evaluation and CSV result export.

## Dataset layout

The script expects this folder structure for each machine:

```text
DATA_ROOT/
└── <machine>/
    ├── train/
    │   └── normal/
    └── test/
        ├── normal/
        └── anomaly/
```

Each folder should contain `.wav` files.

## Usage

Run one machine:

```bash
python Embedding/CNN14_KNN/cnn14_knn.py --machine fan --k 5
```

Run all supported machines:

```bash
python Embedding/CNN14_KNN/cnn14_knn.py --all --k 5
```

### Command-line arguments

- `--machine`: One machine from `MACHINES`, default is `fan`.
- `--all`: Run the method on all machines listed in `MACHINES`.
- `--k`: Number of nearest neighbours used for scoring, default is `5`.

## Outputs

For each run, the script:

- Extracts and caches embeddings as `.npy` files in `.cache/`.
- Evaluates anomaly scores on test normal and test anomaly clips.
- Saves per-machine CSV results to `RESULTS_DIR`.

Example result filenames:

```text
cnn14_knn_k5_fan.csv
cnn14_knn_k5_all.csv
```

## Notes on implementation

- The current code loads the CNN14 wrapper through `AudioTagging` from `panns-inference`, which is the package’s documented entry point for inference.[1]
- The script currently forces inference on CPU inside `load_cnn14_model()` and `extract_embeddings()`, even though a `device` value is passed around.
- The project sample rate should match the pretrained PANNs setup used by the wrapper.
- `k` must not exceed the number of normal training samples.

## Interpretation

This method is a strong baseline when labelled anomaly data is limited. The deep model supplies a rich representation learned from large-scale audio data, while kNN provides a simple non-parametric anomaly score in the embedding space.
