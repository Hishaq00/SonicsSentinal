# Python model py-v4-ensemble-weighted

* Algorithm: **Soft-voting ensemble (XGBoost + Multi-layer Perceptron + Histogram Gradient Boosting) + critical-class weighting**
* Trained: 2026-09-25T09:21:18Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "ensemble", "base_params": {"members": ["xgboost", "mlp", "hist_gradient_boosting"], "weights": "equal", "combination": "mean of class probabilities"}, "class_weights": {"Gunshot": 2.5, "Panic Scream": 1.75, "Aggression": 1.25, "Person Asking for Help": 1.5}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.8444444444444444, "f1_macro": 0.8951229088942171}, "after": {"critical_recall_min": 0.8666666666666667, "f1_macro": 0.9006733785240195}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.834 | 0.852 | 0.846 | 0.844 | 0.800 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.690 | 0.444 | 0.541 | 90 |
| Glass Breaking | 1.000 | 0.978 | 0.989 | 45 |
| Alarm or Siren | 0.937 | 0.804 | 0.865 | 92 |
| Vehicle Horn | 0.907 | 0.880 | 0.893 | 100 |
| Animal Sound | 0.664 | 0.865 | 0.751 | 89 |
| Gunshot | 0.840 | 1.000 | 0.913 | 100 |
| Panic Scream | 0.957 | 1.000 | 0.978 | 45 |
| Aggression | 0.905 | 0.844 | 0.874 | 45 |
| Person Asking for Help | 0.818 | 0.800 | 0.809 | 45 |
| Background Noise | 0.800 | 0.844 | 0.822 | 90 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Support Vector Machine (RBF kernel) | 0.763 | 0.859 | 0.834 | 0.841 | 0.689 |  |
| Random Forest | 0.728 | 0.866 | 0.812 | 0.814 | 0.711 |  |
| Histogram Gradient Boosting | 0.738 | 0.883 | 0.825 | 0.833 | 0.756 |  |
| Multi-layer Perceptron (neural network) | 0.748 | 0.884 | 0.838 | 0.843 | 0.778 |  |
| XGBoost (gradient-boosted trees) | 0.738 | 0.886 | 0.842 | 0.853 | 0.822 |  |
| Soft-voting ensemble (XGBoost + Multi-layer Perceptron + Histogram Gradient Boosting) | 0.741 | 0.895 | 0.833 | 0.842 | 0.778 |  |
| Soft-voting ensemble (XGBoost + Multi-layer Perceptron + Histogram Gradient Boosting) + critical-class weighting | 0.741 | 0.895 | 0.834 | 0.844 | 0.800 | yes |
