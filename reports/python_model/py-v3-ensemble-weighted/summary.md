# Python model py-v3-ensemble-weighted

* Algorithm: **Soft-voting ensemble (Histogram Gradient Boosting + Multi-layer Perceptron + XGBoost) + critical-class weighting**
* Trained: 2026-09-24T19:27:16Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "ensemble", "base_params": {"members": ["hist_gradient_boosting", "mlp", "xgboost"], "weights": "equal", "combination": "mean of class probabilities"}, "class_weights": {"Glass Breaking": 2.5, "Gunshot": 2.5, "Panic Scream": 3.0, "Aggression": 1.25}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.8222222222222222, "f1_macro": 0.9201488810736279}, "after": {"critical_recall_min": 0.8888888888888888, "f1_macro": 0.9375072328834249}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.918 | 0.921 | 0.918 | 0.917 | 0.822 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.952 | 0.889 | 0.920 | 45 |
| Glass Breaking | 0.978 | 1.000 | 0.989 | 45 |
| Alarm or Siren | 0.977 | 0.933 | 0.955 | 45 |
| Vehicle Horn | 0.976 | 0.889 | 0.930 | 45 |
| Animal Sound | 0.947 | 0.800 | 0.867 | 45 |
| Gunshot | 0.898 | 0.978 | 0.936 | 45 |
| Panic Scream | 0.917 | 0.978 | 0.946 | 45 |
| Aggression | 0.891 | 0.911 | 0.901 | 45 |
| Person Asking for Help | 0.841 | 0.822 | 0.831 | 45 |
| Background Noise | 0.830 | 0.978 | 0.898 | 45 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Support Vector Machine (RBF kernel) | 0.801 | 0.889 | 0.902 | 0.902 | 0.733 |  |
| Random Forest | 0.771 | 0.878 | 0.871 | 0.872 | 0.756 |  |
| Histogram Gradient Boosting | 0.798 | 0.914 | 0.918 | 0.919 | 0.822 |  |
| Multi-layer Perceptron (neural network) | 0.802 | 0.906 | 0.924 | 0.924 | 0.822 |  |
| XGBoost (gradient-boosted trees) | 0.804 | 0.905 | 0.904 | 0.905 | 0.778 |  |
| Soft-voting ensemble (Histogram Gradient Boosting + Multi-layer Perceptron + XGBoost) | 0.802 | 0.920 | 0.924 | 0.924 | 0.822 |  |
| Soft-voting ensemble (Histogram Gradient Boosting + Multi-layer Perceptron + XGBoost) + critical-class weighting | 0.802 | 0.926 | 0.918 | 0.917 | 0.822 | yes |

## Noise robustness (test accuracy with added background noise)

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| svm | 0.893 | 0.831 | 0.711 | 0.484 |
| random_forest | 0.880 | 0.807 | 0.638 | 0.329 |
| hist_gradient_boosting | 0.916 | 0.858 | 0.738 | 0.429 |
| mlp | 0.929 | 0.856 | 0.722 | 0.509 |
| xgboost | 0.913 | 0.860 | 0.740 | 0.404 |
| ensemble | 0.922 | 0.867 | 0.753 | 0.480 |
| ensemble_weighted | 0.907 | 0.864 | 0.742 | 0.516 |
