# Python model py-v5-hgb-weighted

* Algorithm: **Histogram Gradient Boosting + critical-class weighting**
* Trained: 2026-09-27T12:28:05Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "hist_gradient_boosting", "base_params": {"l2_regularization": 0.024400607090817518, "learning_rate": 0.08374405912358615, "max_leaf_nodes": 63, "min_samples_leaf": 40}, "class_weights": {"Glass Breaking": 2.0, "Gunshot": 1.75, "Panic Scream": 2.0, "Aggression": 3.0, "Person Asking for Help": 2.5}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.6888888888888889, "f1_macro": 0.8720854906856819}, "after": {"critical_recall_min": 0.7777777777777778, "f1_macro": 0.8846290070266369}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.858 | 0.881 | 0.875 | 0.877 | 0.800 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.889 | 0.889 | 0.889 | 45 |
| Glass Breaking | 1.000 | 0.978 | 0.989 | 45 |
| Alarm or Siren | 0.898 | 0.859 | 0.878 | 92 |
| Vehicle Horn | 0.887 | 0.860 | 0.873 | 100 |
| Animal Sound | 0.696 | 0.865 | 0.771 | 111 |
| Gunshot | 0.882 | 0.900 | 0.891 | 100 |
| Panic Scream | 0.913 | 0.933 | 0.923 | 45 |
| Aggression | 0.878 | 0.800 | 0.837 | 45 |
| Person Asking for Help | 0.818 | 0.800 | 0.809 | 45 |
| Drone Sound | 1.000 | 1.000 | 1.000 | 45 |
| Drilling or Grinder Sound | 0.873 | 0.856 | 0.864 | 104 |
| Fireworks | 0.977 | 0.933 | 0.955 | 45 |
| Vehicle Backfire | 0.844 | 0.844 | 0.844 | 45 |
| Explosion | 0.833 | 0.889 | 0.860 | 45 |
| Door Impact | 0.822 | 0.740 | 0.779 | 50 |
| Graffiti Spray | 0.978 | 1.000 | 0.989 | 45 |
| Background Noise | 0.790 | 0.736 | 0.762 | 174 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Histogram Gradient Boosting | 0.632 | 0.872 | 0.853 | 0.873 | 0.711 |  |
| Random Forest | 0.632 | 0.849 | 0.821 | 0.840 | 0.467 |  |
| XGBoost (gradient-boosted trees) | 0.658 | 0.846 | 0.835 | 0.859 | 0.667 |  |
| Soft-voting ensemble (Histogram Gradient Boosting + Random Forest + XGBoost) | 0.641 | 0.868 | 0.859 | 0.879 | 0.733 |  |
| Histogram Gradient Boosting + critical-class weighting | 0.632 | 0.881 | 0.858 | 0.877 | 0.800 | yes |
