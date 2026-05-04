## Summary: Uncertainty Prediction with CNN

### What I did and how

In this assignment, I built a CNN that predicts three stellar parameters 
— effective temperature (t_eff), surface gravity (log_g), and iron 
abundance (fe_h) — from star spectra, along with the uncertainty of 
each prediction.

I used the same GALAH4 dataset from the previous assignment 
(Vanilla_CNN_Stellar_Spectra), containing 8914 stars each with a 
spectrum of 16384 measurements. The data was split into 70% training, 
15% validation, and 15% test sets.

The model (UncertaintyCNN) is built on top of TinyCNN from the 
professor's solution. Instead of predicting 3 outputs, it predicts 
6 outputs — 3 predictions and 3 uncertainties (σ). The uncertainties 
are predicted as log(σ) to ensure they are always positive, then 
converted back using the exponential function.

Instead of MSE loss used in the previous assignment, I used the 
Negative Log-Likelihood (NLL) loss function, which rewards the model 
for being confident when correct and penalizes it for being confident 
when wrong.

Training used the Adam optimizer with a learning rate of 0.0002 and 
early stopping with patience of 10 epochs. Training stopped 
automatically at epoch 52, with the best model saved at epoch 42.

### What results I obtained

The model showed a massive improvement over Vanilla_CNN_Stellar_Spectra:

| Label | Vanilla RMSE | This Model RMSE | Improvement |
|-------|-------------|-----------------|-------------|
| t_eff | ~94.6 K | 71.90 K |  Better |
| log_g | ~2.70 | 0.13 |  Massively better |
| fe_h | ~2.29 | 0.08 |  Massively better |

The predicted uncertainties (Mean σ) were very close to the actual 
errors for all three labels, meaning the model has a good sense of 
when it is uncertain.

The predicted vs true plots showed all three labels following the 
diagonal closely — a big improvement over Vanilla_CNN_Stellar_Spectra 
where log_g and fe_h completely failed.

### Challenges and what could be improved

The main challenge was understanding the difference between MSE and 
NLL loss, and why we predict log(σ) instead of σ directly.

The uncertainty calibration plot showed that σ is slightly smaller 
than the actual error — meaning the model is slightly overconfident. 
This could be improved by:
- Training for more epochs with a lower learning rate
- Using calibration techniques after training
- Using a larger or deeper model
- Increasing Dropout to further reduce overconfidence
