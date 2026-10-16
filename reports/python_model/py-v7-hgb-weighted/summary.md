# Python model py-v7-hgb-weighted

* Algorithm: **Histogram Gradient Boosting + critical-class weighting**
* Trained: 2026-09-27T12:45:58Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "hist_gradient_boosting", "base_params": {"l2_regularization": 0.024400607090817518, "learning_rate": 0.08374405912358615, "max_leaf_nodes": 63, "min_samples_leaf": 40}, "class_weights": {"Glass Breaking": 1.75, "Gunshot": 1.25, "Panic Scream": 1.1, "Aggression": 1.25, "Person Asking for Help": 1.1}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.7111111111111111, "f1_macro": 0.8728748890832495}, "after": {"critical_recall_min": 0.7555555555555555, "f1_macro": 0.8787375349673356}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.861 | 0.884 | 0.875 | 0.879 | 0.733 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.867 | 0.867 | 0.867 | 45 |
| Glass Breaking | 0.957 | 0.978 | 0.967 | 45 |
| Alarm or Siren | 0.913 | 0.913 | 0.913 | 92 |
| Vehicle Horn | 0.887 | 0.860 | 0.873 | 100 |
| Animal Sound | 0.678 | 0.874 | 0.764 | 111 |
| Gunshot | 0.926 | 0.880 | 0.903 | 100 |
| Panic Scream | 0.952 | 0.889 | 0.920 | 45 |
| Aggression | 0.854 | 0.778 | 0.814 | 45 |
| Person Asking for Help | 0.805 | 0.733 | 0.767 | 45 |
| Drone Sound | 1.000 | 1.000 | 1.000 | 45 |
| Drilling or Grinder Sound | 0.910 | 0.875 | 0.892 | 104 |
| Fireworks | 0.977 | 0.956 | 0.966 | 45 |
| Vehicle Backfire | 0.867 | 0.867 | 0.867 | 45 |
| Explosion | 0.854 | 0.911 | 0.882 | 45 |
| Door Impact | 0.826 | 0.760 | 0.792 | 50 |
| Graffiti Spray | 0.978 | 1.000 | 0.989 | 45 |
| Background Noise | 0.782 | 0.741 | 0.761 | 174 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Random Forest | 0.638 | 0.844 | 0.821 | 0.840 | 0.467 |  |
| Histogram Gradient Boosting | 0.652 | 0.873 | 0.861 | 0.879 | 0.733 |  |
| Histogram Gradient Boosting + critical-class weighting | 0.652 | 0.878 | 0.861 | 0.879 | 0.733 | yes |
