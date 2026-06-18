# BNN MIA — Bayesian Neural Networks & Membership Inference Attacks

Master thesis research notebook implementing and evaluating Membership Inference Attacks (MIA) against Bayesian Neural Networks (BNNs) on image classification datasets.

---

## Overview

This notebook benchmarks several MIA strategies — **LiRA**, **rMIA**, and **BASE** — against both deterministic and Bayesian target models across three datasets: **CIFAR-10**, **CIFAR-100**, and **CINIC-10**. BNN variants include MC Dropout (MCDP), Laplace Approximation (LA), Variational Inference (VI), and Stochastic Weight Averaging Gaussian (SWAG).

Results are reported as ROC curves (log-log scale) with AUC, averaged over multiple target models.

---

## Environment

This notebook is designed to run on **Google Colab** with GPU acceleration and **Google Drive** mounted for persistent storage.

### Package Installation

Run the first cell before anything else. It installs a pinned version of `curvlinops` needed for compatibility with `laplace-torch`:

```python
!pip uninstall -y curvlinops-for-pytorch curvlinops laplace-torch
!pip install "curvlinops-for-pytorch<3" laplace-torch
```

### Full dependency list

| Package | Notes |
|---|---|
| `tensorflow` / `keras` | Target/shadow model training |
| `torch`, `torchvision` | Required by Laplace |
| `laplace-torch` | Laplace Approximation BNN |
| `curvlinops-for-pytorch<3` | Pinned for Laplace compatibility |
| `timm` | Model backbone utilities |
| `datasets` (HuggingFace) | Alternative CIFAR loader |
| `scipy`, `numpy` | Statistical computations |
| `scikit-learn` | ROC curve metrics |
| `matplotlib` | Plot generation |

---

## Setup

### 1. Mount Google Drive

The second cell mounts your Drive and must run successfully before anything else:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2. Set `BASE_DIR`

Immediately after the mount, set `BASE_DIR` to the folder where this notebook is saved:

```python
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Master code"
```

Replace the path with the actual location of the notebook in your Drive. All saved models, attack indices, and score outputs will be written into subdirectories inside `BASE_DIR`:

```
BASE_DIR/
├── saved_target_models/
├── saved_models_external/
├── saved_models_with_batch/
├── saved_models_dp/
├── saved_models_la/
├── saved_models_vi/
├── saved_models_swag/
├── attack_data_indices/
└── score_points/
```

These directories are created automatically at runtime.

---

## Notebook Structure

```
[1] Package Installation (pip)
[2] Import Packages + Mount Drive
[3] Set BASE_DIR
[4] Define pre-Parameters         ← global constants
[5] Load ALL Functions            ← all function definitions
[6] MIA > Define Fixed Parameters ← per-dataset training hyperparameters
[7] MIA > Define Changing Parameters ← hyperparameters to modify when switching models
[8] MIA > MIA Functions           ← main functions
[9] RUN MIA                       ← Show main results
[10] Further Studies              ← optional analysis cells
```

---

## Configuration

### Pre-Parameters (Cell 4)

Set once before loading functions. These control global inference behaviour:

```python
n_batch_max = 256       # max batch size for inference decomposition
mia_mode    = "external"  # shadow-model mode: "external" | "normal"

mia_pars: dict = {
    "alpha": 1,   # heuristic for rMIA/BASE (1 = disabled)
    "gamma": 1    # rMIA decision threshold
}
```

### Fixed Training Parameters (Cell 6)

Per-dataset hyperparameters for each BNN type. These are stable across runs and should only be changed if retraining with a different setup:

```python
# Example for CIFAR-10, deterministic shadow models
cifar10_training_pars: dict = {
    "len_train_dataset": 15000,
    "epochs": 17, # 10 for MobileNetV2, 17 for small ResNet
    "batch_size": 128
}

# Number of Laplace posterior samples and SWAG shadow models
n_la_samples = 1000
n_swag       = 30
```

Equivalent dicts exist for `cinic10_training_pars`, `cifar100_training_pars`, and their `_mcdp` (Monte Carlo Dropout) and `_vi` (Variational Inference) variants.

### Changing Parameters (Cell 7)

Select which model architecture to attack:

```python
# Target model for shadow-model MIA
model_function = resnet_model
# Options: resnet_model, small_resnet_model, mobilenetv2_model, efficientnetv2_model

# Target model for MC Dropout MIA
model_function_mcdp = drop_out_resnet_model
# Options: drop_out_resnet_model, drop_out_small_resnet_model,
#          drop_out_mobilenetv2_model, drop_out_efficientnetv2_model

# Target model for Laplace MIA
model_function_la = laplace_resnet_model
# Options: laplace_resnet_model, laplace_small_resnet_model, laplace_mobilenetv2_model
```

---

## Running the Main Experiment

The **RUN MIA** section (Cell 9) is the primary entry point. It contains three `multiple_mia_attacks` calls, one per dataset.

### Step 1 — Set `refreshing_pars`

This dict controls which components are retrained or recomputed. Set flags to `True` on a fresh run, or `False` to reuse previously saved models and scores:

```python
refreshing_pars: dict = {
    "targets_retraining":           True,   # train new target models
    "audit_resampling":             True,   # resample audit points
    "shadows_retraining":           True,   # train shadow models
    "compute_shadows_pred_scores":  True,   # run shadow model inference
    "mcdp_retraining":              True,   # train MC Dropout model
    "compute_mcdp_pred_scores":     True,   # run MCDP inference
    "la_retraining":                False,  # train Laplace model
    "compute_la_pred_scores":       False,  # run LA inference
    "vi_retraining":                False,  # train VI model
    "compute_vi_pred_scores":       False,  # run VI inference
    "swag_retraining":              False,  # train SWAG models
    "compute_swag_pred_scores":     False   # run SWAG inference
}
```

### Step 2 — Run the attack cells

Each dataset has its own call:

```python
# CIFAR-100
multiple_mia_attacks("cifar100", model_function, n_target_models=10,
                     n_shadow_models_list=[1, 2, 3, 5, 10, 30],
                     n_audit_points=30000, members_audit_ratio=0.5,
                     training_pars=cifar100_training_pars,
                     model_function_mcdp=model_function_mcdp,
                     training_pars_mcdp=cifar100_training_pars_mcdp,
                     model_function_la=None, n_la_samples=n_la_samples,
                     training_pars_vi=cifar100_training_pars_vi,
                     n_swag=n_swag,
                     refreshing_pars=refreshing_pars,
                     do_lira=False, do_rmia=False, do_base=True,
                     seed=42)

# CIFAR-10
multiple_mia_attacks("cifar10", model_function, n_target_models=10, ...)

# CINIC-10
multiple_mia_attacks("cinic10", model_function, n_target_models=10, ...)
```

Key parameters:

| Parameter | Description |
|---|---|
| `n_target_models` | Number of target models to train and average over |
| `n_shadow_models_list` | List of shadow model counts to sweep over |
| `n_audit_points` | Total audit set size (members + non-members) |
| `members_audit_ratio` | Fraction of audit points that are members |
| `do_lira` / `do_rmia` / `do_base` | Toggle which attack strategies to run |

### Output

Each run saves ROC TPR/AUC arrays under `BASE_DIR/score_points/` and displays log-log ROC curves comparing all enabled attack strategies and BNN types.

---

## CINIC-10 Dataset

CINIC-10 is not bundled with Keras. The `load_cinic_datasets()` function will:

1. Check for a pre-downloaded `CINIC-10.tar.gz` inside `BASE_DIR`.
2. If absent, attempt to download it from the Edinburgh DataShare.
3. Extract and load the dataset from the `train/`, `valid/`, and `test/` directories.

If the automatic download fails, download it manually from:
```
https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
```
and place it in `BASE_DIR`.

---

## Further Studies (Optional)

The **Further Studies** section below the main RUN MIA cells contains additional diagnostic tools. These are independent of the main pipeline and can be run selectively after the main experiment.
They all use **Standalone dataset loaders**. Use them to re-load CIFAR-10/100 or CINIC-10 individually for custom analysis. `model_datasize` should be set to match previously saved shadow models.

- **MCDP Analysis** — `model_prediction_variation()`: plots histograms of attack score distributions (phi-values) across different numbers of MC Dropout iterations.

> These cells are not required for the main results and contain functions that are not called during `RUN MIA`.

---

## Recommended Run Order

```
1. Cell: pip install (Laplace pinned deps)
2. Cell: imports + drive.mount()
3. Cell: BASE_DIR = "your/path/here"
4. Cell: pre-parameters
5. All cells under "Load ALL Functions"
6. All cells under "MIA > Define Fixed Parameters"
7. Cell: Define Changing Parameters (model selection)
8. All cells under "MIA > MIA Functions"
9. Cell: refreshing_pars
10. Cells: multiple_mia_attacks(...) for each dataset
```

---

## Notes

- High-RAM GPU runtime is strongly recommended; training 10 target models + shadow models per dataset is computationally intensive.
- All random seeds are controlled via the `seed=42` argument. Change this to reproduce different splits.
- The `out_points_n = 8000` parameter in Cell 7 controls how many additional training points are reserved per target model shift. Reducing this increases the overlap between the target models training sets. It can be safely lowered to 2000 data points.
- Models are saved in `.keras` format and can be reloaded by setting the relevant `refreshing_pars` flags to `False`.
