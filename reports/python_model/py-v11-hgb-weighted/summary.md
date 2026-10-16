# Python model py-v11-hgb-weighted

* Algorithm: **Histogram Gradient Boosting + critical-class weighting**
* Trained: 2026-09-28T16:09:16Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "hist_gradient_boosting", "base_params": {"l2_regularization": 0.024400607090817518, "learning_rate": 0.08374405912358615, "max_leaf_nodes": 63, "min_samples_leaf": 40}, "class_weights": {"Glass Breaking": 3.0, "Gunshot": 1.1, "Panic Scream": 1.1, "Aggression": 1.5, "Person Asking for Help": 2.0, "Explosion": 1.25}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.6444444444444445, "f1_macro": 0.8692533140344396}, "after": {"critical_recall_min": 0.8222222222222222, "f1_macro": 0.883077235850568}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.834 | 0.892 | 0.858 | 0.870 | 0.778 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.891 | 0.911 | 0.901 | 45 |
| Glass Breaking | 1.000 | 0.980 | 0.990 | 51 |
| Alarm or Siren | 0.923 | 0.846 | 0.883 | 156 |
| Vehicle Horn | 0.948 | 0.835 | 0.888 | 109 |
| Animal Sound | 0.789 | 0.760 | 0.774 | 246 |
| Gunshot | 0.930 | 0.800 | 0.860 | 100 |
| Panic Scream | 0.933 | 0.933 | 0.933 | 45 |
| Aggression | 0.907 | 0.867 | 0.886 | 45 |
| Person Asking for Help | 0.745 | 0.778 | 0.761 | 45 |
| Drone Sound | 0.978 | 1.000 | 0.989 | 45 |
| Drilling or Grinder Sound | 0.826 | 0.460 | 0.591 | 124 |
| Fireworks | 0.925 | 0.961 | 0.942 | 51 |
| Vehicle Backfire | 0.950 | 0.844 | 0.894 | 45 |
| Explosion | 0.778 | 0.933 | 0.848 | 45 |
| Door Impact | 0.898 | 0.786 | 0.838 | 56 |
| Graffiti Spray | 1.000 | 1.000 | 1.000 | 45 |
| Background Noise | 0.745 | 0.885 | 0.809 | 550 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Multi-layer Perceptron (neural network) | 0.677 | 0.850 | 0.836 | 0.858 | 0.840 |  |
| Histogram Gradient Boosting | 0.672 | 0.869 | 0.836 | 0.873 | 0.733 |  |
| Histogram Gradient Boosting + critical-class weighting | 0.672 | 0.877 | 0.834 | 0.870 | 0.778 | yes |

## Live-microphone rendering (same recordings, as the live monitor delivers them)

| Split | Accuracy | Macro F1 | Min critical recall |
|---|---|---|---|
| validation | 0.842 | 0.854 | 0.733 |
| test | 0.830 | 0.854 | 0.730 |

## Confidence calibration

Temperature 1.9106 fitted on the validation segments. Expected calibration error on the test segments 0.1137 -> 0.0213, mean confidence 0.9178 -> 0.8148 against accuracy 0.8041. The predicted class does not change.
