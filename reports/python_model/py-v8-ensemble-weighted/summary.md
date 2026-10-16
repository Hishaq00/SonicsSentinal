# Python model py-v8-ensemble-weighted

* Algorithm: **Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + Support Vector Machine) + critical-class weighting**
* Trained: 2026-09-27T15:14:58Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "ensemble", "base_params": {"members": ["mlp", "hist_gradient_boosting", "svm"], "weights": "equal", "combination": "mean of class probabilities"}, "class_weights": {"Glass Breaking": 1.25, "Aggression": 1.25, "Person Asking for Help": 1.1, "Explosion": 1.75}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.8222222222222222, "f1_macro": 0.9098895251330746}, "after": {"critical_recall_min": 0.8666666666666667, "f1_macro": 0.915581140900028}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.860 | 0.889 | 0.881 | 0.881 | 0.800 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.930 | 0.889 | 0.909 | 45 |
| Glass Breaking | 1.000 | 1.000 | 1.000 | 45 |
| Alarm or Siren | 0.910 | 0.880 | 0.895 | 92 |
| Vehicle Horn | 0.868 | 0.920 | 0.893 | 100 |
| Animal Sound | 0.681 | 0.883 | 0.769 | 111 |
| Gunshot | 0.880 | 0.880 | 0.880 | 100 |
| Panic Scream | 0.977 | 0.933 | 0.955 | 45 |
| Aggression | 0.902 | 0.822 | 0.860 | 45 |
| Person Asking for Help | 0.800 | 0.800 | 0.800 | 45 |
| Drone Sound | 1.000 | 1.000 | 1.000 | 45 |
| Drilling or Grinder Sound | 0.954 | 0.798 | 0.869 | 104 |
| Fireworks | 1.000 | 0.978 | 0.989 | 45 |
| Vehicle Backfire | 0.921 | 0.778 | 0.843 | 45 |
| Explosion | 0.672 | 0.956 | 0.789 | 45 |
| Door Impact | 0.841 | 0.740 | 0.787 | 50 |
| Graffiti Spray | 0.978 | 1.000 | 0.989 | 45 |
| Background Noise | 0.796 | 0.718 | 0.755 | 174 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Support Vector Machine (RBF kernel) | 0.692 | 0.866 | 0.851 | 0.875 | 0.667 |  |
| Random Forest | 0.660 | 0.855 | 0.818 | 0.837 | 0.467 |  |
| Histogram Gradient Boosting | 0.672 | 0.884 | 0.867 | 0.883 | 0.689 |  |
| Multi-layer Perceptron (neural network) | 0.677 | 0.891 | 0.854 | 0.877 | 0.867 |  |
| XGBoost (gradient-boosted trees) | 0.674 | 0.859 | 0.852 | 0.873 | 0.711 |  |
| Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + Support Vector Machine) | 0.680 | 0.910 | 0.862 | 0.884 | 0.756 |  |
| Soft-voting ensemble (Multi-layer Perceptron + Histogram Gradient Boosting + Support Vector Machine) + critical-class weighting | 0.680 | 0.915 | 0.860 | 0.881 | 0.800 | yes |

## Noise robustness (test accuracy with added background noise)

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| svm | 0.824 | 0.755 | 0.655 | 0.472 |
| random_forest | 0.808 | 0.727 | 0.568 | 0.337 |
| hist_gradient_boosting | 0.838 | 0.781 | 0.635 | 0.395 |
| mlp | 0.819 | 0.765 | 0.673 | 0.460 |
| xgboost | 0.815 | 0.745 | 0.605 | 0.391 |
| ensemble | 0.848 | 0.794 | 0.689 | 0.459 |
| ensemble_weighted | 0.846 | 0.786 | 0.685 | 0.460 |
