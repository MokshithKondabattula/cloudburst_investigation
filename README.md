<div align="center">

# 🌩️ Cloudburst Investigation
### *CloudburstNet: ConvLSTM + U-Net for Satellite Precipitation Nowcasting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![IIIT Kalyani](https://img.shields.io/badge/IIIT-Kalyani-blue?style=for-the-badge)](https://www.iiitklyani.ac.in)

**Deep learning pipeline that forecasts cloudburst-scale rainfall from INSAT-3DR Water Vapour satellite sequences — no radar, no NWP, sub-minute inference.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training Results](#-training-results)
- [Visual Output](#-visual-output)
- [Quantitative Metrics](#-quantitative-metrics)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🌐 Overview

Cloudbursts — rainfall events exceeding **100 mm/hr** over a localised area — develop within minutes and are virtually impossible to capture with conventional NWP grids. This project presents **CloudburstNet**, a spatiotemporal deep learning model that maps a 6-hour sequence of INSAT-3DR Water Vapour satellite images directly to a predicted GPM IMERG precipitation field at the next time step.

| Property | Value |
|---|---|
| Input | 12 × WV frames (past 6 hours, 320×320) |
| Output | Predicted precipitation map (320×320) |
| Forecast horizon | +30 minutes |
| Region of interest | Indian subcontinent (6°N–32°N, 68°E–92°E) |
| Inference time | < 1 second (GPU) |

---

## 🏗️ Architecture

CloudburstNet is a three-stage sequential pipeline:

```
Input (12 × WV frames)
        │
        ▼
┌───────────────────┐
│   CNN  Encoder    │  ← Extracts spatial features per frame
│ [1→32→64→128→256] │    Double Conv + BN + ReLU + MaxPool ×4
│  320² → 20² feat  │    Skip connections saved at each stage
└────────┬──────────┘
         │  stack 12 bottlenecks → (B, 12, 256, 20, 20)
         ▼
┌───────────────────┐
│  ConvLSTM ×2      │  ← Models temporal evolution
│  [256 → 128 ch]   │    Convolutional gates (3×3 kernels)
│  Forget · Input   │    Tracks cloud motion & dissipation
│  Output · Cell    │
└────────┬──────────┘
         │  final hidden state → (B, 128, 20, 20)
         ▼
┌───────────────────┐
│  U-Net Decoder    │  ← Reconstructs full-resolution field
│  TransposeConv ×4 │    Skip connections from encoder stages
│  20² → 320²       │    1×1 Conv + Sigmoid → [0, 1]
└───────────────────┘
         │
         ▼
  Precipitation Map (1 × 320 × 320)
```

### Loss Function

$$\mathcal{L} = 0.5 \times \frac{1}{2}(\mathcal{L}_{\text{MAE}} + \mathcal{L}_{\text{wMAE}}) + 0.5 \times \mathcal{L}_{\text{SSIM}}$$

Weighted MAE up-weights heavy rainfall pixels (> 0.5 normalised) by **3×** to counteract class imbalance.

---

## 📦 Dataset

| Source | Product | Format | Resolution | Interval |
|---|---|---|---|---|
| ISRO / MOSDAC | INSAT-3DR WV (6.7–7.1 µm) | HDF | ~4 km | 30 min |
| NASA GES DISC | GPM IMERG Late Run | HDF5 | 0.1° | 30 min |

**20 rainfall events** were curated and split at the event level to prevent data leakage:

| Split | Events | Purpose |
|---|---|---|
| Train | 1–14 (14 events) | Weight optimisation |
| Validation | 15–17 (3 events) | Hyperparameter tuning, early stopping |
| Test | 18–20 (3 events) | Final evaluation |

Each event yields multiple overlapping sequences via a **sliding window** (length 12, stride 1), never crossing event boundaries.

### Preprocessing Pipeline

```
HDF / HDF5  →  Lat-Lon crop  →  Temporal sync  →  Bilinear resize (320×320)  →  Min-Max normalise [0, 1]
```

- INSAT brightness temperatures clipped to 200–280 K
- GPM precipitation clipped at 50 mm/hr, then log-scaled

---

## 📈 Training Results

<table>
<tr>
<td align="center" width="50%">

**Training & Validation Loss**

<img src="loss_curve.png" alt="Training and Validation Loss" width="100%"/>

Validation loss stabilised near **0.13** from epoch 18, with minimal train/val gap — no severe overfitting.

</td>
<td align="center" width="50%">

**SSIM & Pearson Correlation**

<img src="correlation_metrics.png" alt="SSIM and Pearson Correlation" width="100%"/>

SSIM climbed from **0.11 → 0.68** over 18 epochs. Most gain occurred in epochs 5–15 as the model began producing spatially coherent outputs.

</td>
</tr>
<tr>
<td align="center" width="50%">

**MAE & RMSE**

<img src="error_metrics.png" alt="MAE and RMSE" width="100%"/>

MAE converged to **~0.015** (normalised), RMSE to **~0.042** — sharp drop in first 4 epochs then gradual refinement.

</td>
<td align="center" width="50%">

**Critical Success Index (Multi-Threshold)**

<img src="csi_curves.png" alt="CSI at multiple thresholds" width="100%"/>

CSI@0.2 trends upward through training. CSI@0.4 and CSI@0.6 remain near zero — reflecting the challenge of predicting intense, localised rainfall cores.

</td>
</tr>
</table>

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 (cosine annealed → 1e-6) |
| Weight decay | 1e-5 |
| Batch size | 2 |
| Max epochs | 30 |
| Gradient clip (max norm) | 1.0 |
| Early stopping patience | 10 epochs |

---

## 🖼️ Visual Output

Prediction panels at Epochs 0, 5, and 10 — showing the model's evolution from near-uniform outputs to spatially structured precipitation fields:

<div align="center">
<img src="WhatsApp_Image_2026-05-15_at_2_32_18_AM__1_.jpeg" alt="Prediction panels at Epoch 0, 5, and 10" width="90%"/>

*Left to right per row: Last WV Input · Ground Truth · Predicted · Absolute Error*
</div>

**Key observations:**
- **Epoch 0:** Nearly uniform near-zero predictions — model has not yet learned rainfall structure
- **Epoch 5:** Broad spatial pattern begins to emerge, roughly matching rainfall zone location
- **Epoch 10:** Improved spatial coherence; model correctly localises the general rainfall region but still underestimates peak intensity in compact convective cores

---

## 📊 Quantitative Metrics

### Pixel-Level Metrics (Test Set)

| Metric | Test Value | Interpretation |
|---|---|---|
| MAE | **0.132** | ~6.6 mm/hr avg per-pixel error (denormalised) |
| RMSE | **0.187** | Higher than MAE → large errors in convective cores |
| PSNR | **14.6 dB** | Consistent with similar-resolution DL nowcasting models |
| Pearson *r* | **0.61** | Moderate spatial agreement |
| SSIM | **0.68** | Captures broad structural features of precipitation field |

### Threshold-Based Metrics (Test Set)

| Threshold | Label | CSI | POD | FAR |
|---|---|---|---|---|
| 0.2 | Light rain | **0.38** | 0.57 | 0.41 |
| 0.4 | Moderate rain | **0.17** | 0.29 | 0.52 |
| 0.6 | Heavy rain | **0.00** | 0.00 | — |

> Performance degrades sharply with intensity threshold. Heavy rainfall CSI = 0 reflects the extreme pixel-level rarity of cloudburst cores and the model's smooth output distribution. This is the primary open challenge.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cloudburst-investigation.git
cd cloudburst-investigation

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

```
torch>=2.0
torchvision
h5py
numpy
scipy
scikit-image
matplotlib
pandas
tqdm
```

---

## 🚀 Usage

### 1. Preprocess Data

```bash
python preprocess.py \
  --insat_dir /data/insat_hdf \
  --gpm_dir   /data/gpm_hdf5 \
  --out_dir   /data/processed \
  --lat_min 6  --lat_max 32 \
  --lon_min 68 --lon_max 92
```

### 2. Train

```bash
python train.py \
  --data_dir  /data/processed \
  --epochs    30 \
  --batch     2 \
  --lr        1e-4 \
  --out_dir   checkpoints/
```

### 3. Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir   /data/processed \
  --split      test \
  --save_plots results/
```

### 4. Inference on a single sequence

```python
from model import CloudburstNet
import torch

model = CloudburstNet()
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# wv_seq: (1, 12, 1, 320, 320) tensor, normalised to [0, 1]
with torch.no_grad():
    pred = model(wv_seq)   # → (1, 1, 320, 320)
```

---

## 📁 Project Structure

```
cloudburst-investigation/
│
├── data/
│   ├── preprocess.py          # HDF → PNG pipeline
│   └── dataset.py             # PyTorch Dataset + sliding window
│
├── model/
│   ├── encoder.py             # CNN encoder (4-stage, skip connections)
│   ├── convlstm.py            # ConvLSTM cell & stacked module
│   ├── decoder.py             # U-Net decoder with transpose convolutions
│   └── cloudburst_net.py      # Full model assembly
│
├── loss/
│   └── combined_loss.py       # MAE + weighted MAE + SSIM
│
├── train.py                   # Training loop, checkpointing, early stopping
├── evaluate.py                # Metrics computation & visualisation
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- **Attention mechanisms** — spatial self-attention + cross-frame temporal attention to focus on active convective cells
- **Transformer temporal encoder** — replace ConvLSTM with a spatiotemporal transformer for longer-range dependencies
- **GAN post-processing** — adversarial sharpening to push predicted intensities beyond the current smoothness bias
- **Multi-channel input** — add INSAT-3DR visible, SWIR, and TIR channels for richer convective information
- **Probabilistic forecasting** — Monte Carlo dropout / ensemble training to produce calibrated uncertainty maps
- **Expanded dataset** — cover more seasons and geographic sub-regions to reduce event-level overfitting

---

## 📚 References

1. Shi, X. et al. (2015) *ConvLSTM Network: A Machine Learning Approach for Precipitation Nowcasting.* NeurIPS.
2. Ronneberger, O., Fischer, P. and Brox, T. (2015) *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
3. Wang, Z. et al. (2004) *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP.
4. Huffman, G.J. et al. (2020) *GPM IMERG Technical Documentation.* NASA/GSFC.
5. Ravuri, S. et al. (2021) *Skilful Precipitation Nowcasting Using Deep Generative Models of Radar.* Nature, 597.
6. Loshchilov, I. and Hutter, F. (2019) *Decoupled Weight Decay Regularization.* ICLR.

---

## 🏛️ Institution

<div align="center">

**Indian Institute of Information Technology Kalyani**
*B.Tech Computer Science and Engineering — 2024*

Supervised by **Dr. Uma Das**, Assistant Professor, Physics

</div>

---

<div align="center">


</div>

