# EEG-Driven Motor Imagery Classification for Gamified Stroke Rehabilitation

## Overview

This repository implements a sophisticated **EEG-driven motor imagery classification system** designed for **gamified stroke rehabilitation**, leveraging a hybrid **CNN-Transformer** architecture. It addresses a Complex Engineering Problem (CEP) under the Department of Biomedical Engineering, Riphah International University, focusing on patient-specific upper limb neuro-rehabilitation.

The system integrates:
- Advanced signal processing
- Machine learning
- Real-time gamified visual feedback

It quantifies neuroplasticity and enhances motor recovery using single-channel EEG data, achieving **78.33% test accuracy** on the **3D-ARM-Gaze** dataset.

---

## Repository Structure

### `1.png` and `2.png`
- Static image assets representing **hand opening and closing**.
- Rendered in the gamified interface.
- Animation speed modulated by intent score:  
  `speed_factor = max(200, 2000 - (current_score * 1800 / 30520))` (ms)

### `Model.py`
- **Implements CNN-Transformer model** in TensorFlow 2.15.
- Input shape: `(500, 1)` → Output: 11-class softmax (range-of-motion classes).
- Architecture:
  - 3 Conv1D layers (128, 256, 256 filters)
  - Multi-head attention (8 heads, 256 dim)
  - Dense layers (L2=0.01)
- ~1.2 million parameters.
- Optimized with Adam (`lr=0.001`) and sparse categorical cross-entropy.
- Class weights used for imbalance handling.

### `app.py`
- Main execution script (Python).
- Initializes full-screen Matplotlib GUI.
- Handles EEG file uploads: `(n, 500, 1)` shape.
- Uses pre-trained `Model.py` for inference (batch size=16).
- Computes:
  ```python
  normalized_score = np.mean(np.argmax(predictions, axis=1)) / 10  
  current_score = 100 + (30520 - 100) * normalized_score
  ```
- Features:
  - Hand animation
  - Tiered medals:  
    - Bronze ≤ 10173  
    - Silver: 10174–20346  
    - Gold ≥ 20347
  - Real-time EEG intensity plots

### `README.md`
- This file. Contains all documentation, usage, and licensing.

---

## EEG Preprocessing Pipeline

- **Input:**  
  Raw EEG from **3D-ARM-Gaze** dataset (14,000 trials, 20 subjects, 100 Hz)

- **Extraction:**  
  Parses JSON files (e.g., `initial_acquisition.json`) using:
  ```python
  eeg_data_keys = ["signal", "timeseries", "data", "eeg"]
  ```

- **Denoising:**  
  Wavelet transform (`db4`) with soft thresholding:
  ```
  τ = σ√(2ln(N)), N=500
  ```
  → SNR gain: 8–12 dB

- **Standardization:**  
  Z-score: `(x - μ) / σ`  
  Clip outliers >3σ  
  Pad/truncate to 500 samples

- **Augmentation:**
  - Gaussian noise: `ε ~ N(0, 0.03 * std(x))`
  - Amplitude scaling: `s = ROM + δ`, `δ ~ U(-0.05, 0.05)`
  - Temporal shifting: `roll(x, τ)`, `τ ~ U(-10, 10)`
  - **Total: ~220,000 samples**

---

## Model Architecture

- **Input:** `(500, 1)` z-scored EEG tensors
- **Convolutional Block:**
  - 3 Conv1D layers (128, 256, 256 filters)
  - Kernel size: 3, ReLU, same padding
  - BatchNorm, Dropout (20%), MaxPool (pool=2)
- **Transformer Block:**
  - 8-head attention, 256-dim keys
  - Residual connection + LayerNorm
- **Output:**
  - Dense (512, ReLU) + GlobalAvgPool
  - Dense (11, softmax)
- **Training:**
  - 2 hours on NVIDIA GPU
  - Loss: Weighted sparse categorical cross-entropy
  - Regularization: L2 (0.01)

---

## Gamification Interface

- **Workflow:**
  1. Upload EEG file
  2. Inference using model
  3. Score computation
  4. Hand animation based on `speed_factor`
  5. Medal update
  6. Intensity graph update

- **Performance Logging:**
  - Peak score saved in local JSON
  - Reset & error handling with try-except

---

## Usage Instructions

### Dependencies
Install required packages:
```bash
pip install tensorflow==2.15 numpy matplotlib pywt
```

### Running the App
```bash
python app.py
```

### Input Format
- EEG data in `.npy` format: shape `(n, 500, 1)`
- Example files in `testfiles/`

### Interaction
- **Select File** → Load EEG
- **Next** → Run inference
- View:
  - Hand animation
  - Intensity plot
  - Score & Medal updates

---

## License

This project is licensed for **non-commercial prototyping and testing only**.

> Redistribution, commercial use, or modification of the code, data, or assets (e.g., 1.png, 2.png) without explicit written permission is strictly prohibited.


---

## Acknowledgments

**Dataset:**  
[3D-ARM-Gaze dataset (Lento et al., 2024)](https://doi.org/10.1038/s41597-024-03765-4)
