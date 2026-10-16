# Python model py-v6-ensemble-weighted

* Algorithm: **Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + XGBoost) + critical-class weighting**
* Trained: 2026-09-27T12:40:36Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "ensemble", "base_params": {"members": ["mlp", "hist_gradient_boosting", "xgboost"], "weights": "equal", "combination": "mean of class probabilities"}, "class_weights": {"Glass Breaking": 2.0, "Gunshot": 1.1, "Panic Scream": 1.75, "Aggression": 1.5, "Person Asking for Help": 1.1, "Explosion": 1.1}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.7333333333333333, "f1_macro": 0.8945518764330223}, "after": {"critical_recall_min": 0.8222222222222222, "f1_macro": 0.8985221410034666}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.862 | 0.880 | 0.885 | 0.880 | 0.778 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.784 | 0.889 | 0.833 | 45 |
| Glass Breaking | 0.978 | 1.000 | 0.989 | 45 |
| Alarm or Siren | 0.916 | 0.826 | 0.869 | 92 |
| Vehicle Horn | 0.896 | 0.860 | 0.878 | 100 |
| Animal Sound | 0.705 | 0.883 | 0.784 | 111 |
| Gunshot | 0.925 | 0.860 | 0.891 | 100 |
| Panic Scream | 0.957 | 1.000 | 0.978 | 45 |
| Aggression | 0.860 | 0.822 | 0.841 | 45 |
| Person Asking for Help | 0.814 | 0.778 | 0.795 | 45 |
| Drone Sound | 1.000 | 1.000 | 1.000 | 45 |
| Drilling or Grinder Sound | 0.909 | 0.865 | 0.887 | 104 |
| Fireworks | 1.000 | 0.956 | 0.977 | 45 |
| Vehicle Backfire | 0.800 | 0.889 | 0.842 | 45 |
| Explosion | 0.778 | 0.933 | 0.848 | 45 |
| Door Impact | 0.881 | 0.740 | 0.804 | 50 |
| Graffiti Spray | 0.957 | 1.000 | 0.978 | 45 |
| Background Noise | 0.800 | 0.736 | 0.766 | 174 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Support Vector Machine (RBF kernel) | 0.762 | 0.864 | 0.844 | 0.864 | 0.689 |  |
| Random Forest | 0.712 | 0.844 | 0.821 | 0.840 | 0.467 |  |
| Histogram Gradient Boosting | 0.740 | 0.886 | 0.854 | 0.875 | 0.733 |  |
| Multi-layer Perceptron (neural network) | 0.746 | 0.888 | 0.843 | 0.859 | 0.778 |  |
| XGBoost (gradient-boosted trees) | 0.742 | 0.865 | 0.859 | 0.880 | 0.711 |  |
| Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + XGBoost) | 0.743 | 0.895 | 0.865 | 0.883 | 0.778 |  |
| Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + XGBoost) + critical-class weighting | 0.743 | 0.897 | 0.862 | 0.880 | 0.778 | yes |

## Noise robustness (test accuracy with added background noise)

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| svm | 0.820 | 0.760 | 0.655 | 0.458 |
| random_forest | 0.800 | 0.721 | 0.566 | 0.338 |
| hist_gradient_boosting | 0.831 | 0.779 | 0.634 | 0.401 |
| mlp | 0.810 | 0.758 | 0.652 | 0.465 |
| xgboost | 0.817 | 0.758 | 0.615 | 0.383 |
| ensemble | 0.848 | 0.797 | 0.667 | 0.430 |
| ensemble_weighted | 0.848 | 0.798 | 0.667 | 0.444 |
