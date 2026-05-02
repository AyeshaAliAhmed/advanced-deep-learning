# Stellar Spectra Classification with a Vanilla CNN

## Overview
This project implements a Convolutional Neural Network (CNN) in PyTorch to predict 
three physical properties of stars from their spectra:
- **t_eff** — Effective Temperature (Kelvin)
- **log_g** — Surface Gravity
- **fe_h** — Iron Abundance

## Dataset
The dataset is the GALAH4 dataset from HuggingFace, containing 8914 stars, 
each with a spectrum of 16384 measurements.

🔗 https://huggingface.co/datasets/simbaswe/galah4/tree/main

## Model Architecture
A standard 1D CNN with three convolutional blocks followed by two fully connected layers:

| Layer | Output Channels | Spectrum Length |
|-------|----------------|-----------------|
| Conv1d + ReLU + MaxPool | 16 | 8192 |
| Conv1d + ReLU + MaxPool | 32 | 4096 |
| Conv1d + ReLU + MaxPool | 64 | 2048 |
| Flatten | — | 131072 |
| Linear + ReLU | — | 256 |
| Linear (output) | — | 3 |

**Total trainable parameters: 33,563,299**

## Results
The model was trained for 20 epochs using the Adam optimizer (lr=0.001) and MSE loss.

| Label | Test MSE | Test RMSE |
|-------|----------|-----------|
| t_eff | 8948.93 | ~94.6 K |
| log_g | 7.29 | ~2.70 |
| fe_h | 5.23 | ~2.29 |

The model performed well on t_eff but struggled with log_g and fe_h, 
showing signs of overfitting after epoch 5.

## Requirements
- Python 3
- PyTorch
- NumPy
- Matplotlib
- torchsummary

## Usage
1. Download the dataset from HuggingFace and upload to Google Drive
2. Open the notebook in Google Colab
3. Set the correct DATA_PATH in cell 3
4. Run all cells in order

## Course
Advanced Deep Learning — Assignment B01 (Warm-up Exercise)
