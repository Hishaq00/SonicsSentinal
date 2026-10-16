# Python model py-v1-hist-gradient-boosting

* Algorithm: **Histogram Gradient Boosting**
* Trained: 2026-09-24T19:08:33Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"l2_regularization": 0.012561043700013555, "learning_rate": 0.08110848199986004, "max_leaf_nodes": 15, "min_samples_leaf": 40}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.918 | 0.922 | 0.918 | 0.919 | 0.822 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.932 | 0.911 | 0.921 | 45 |
| Glass Breaking | 0.978 | 0.978 | 0.978 | 45 |
| Alarm or Siren | 1.000 | 0.933 | 0.966 | 45 |
| Vehicle Horn | 0.977 | 0.933 | 0.955 | 45 |
| Animal Sound | 0.949 | 0.822 | 0.881 | 45 |
| Gunshot | 0.956 | 0.956 | 0.956 | 45 |
| Panic Scream | 0.977 | 0.956 | 0.966 | 45 |
| Aggression | 0.870 | 0.889 | 0.879 | 45 |
| Person Asking for Help | 0.740 | 0.822 | 0.779 | 45 |
| Background Noise | 0.846 | 0.978 | 0.907 | 45 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Support Vector Machine (RBF kernel) | 0.801 | 0.895 | 0.911 | 0.912 | 0.822 |  |
| Random Forest | 0.771 | 0.878 | 0.871 | 0.872 | 0.756 |  |
| Histogram Gradient Boosting | 0.798 | 0.914 | 0.918 | 0.919 | 0.822 | yes |
| Multi-layer Perceptron (neural network) | 0.802 | 0.906 | 0.924 | 0.924 | 0.822 |  |
| XGBoost (gradient-boosted trees) | 0.804 | 0.905 | 0.904 | 0.905 | 0.778 |  |

## Noise robustness (test accuracy with added background noise)

| Model | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| svm | 0.911 | 0.864 | 0.740 | 0.513 |
| random_forest | 0.880 | 0.807 | 0.638 | 0.329 |
| hist_gradient_boosting | 0.916 | 0.858 | 0.738 | 0.429 |
| mlp | 0.929 | 0.856 | 0.724 | 0.509 |
| xgboost | 0.913 | 0.860 | 0.740 | 0.404 |
