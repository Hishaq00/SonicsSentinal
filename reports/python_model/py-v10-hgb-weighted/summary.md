# Python model py-v10-hgb-weighted

* Algorithm: **Histogram Gradient Boosting + critical-class weighting**
* Trained: 2026-09-28T12:31:53Z
* Features: 193 (ssf-1.0), segments of 2.0 s
* Hyper-parameters: `{"base_model": "hist_gradient_boosting", "base_params": {"l2_regularization": 0.024400607090817518, "learning_rate": 0.08374405912358615, "max_leaf_nodes": 63, "min_samples_leaf": 40}, "class_weights": {"Glass Breaking": 3.0, "Gunshot": 2.5, "Panic Scream": 2.0, "Aggression": 2.5, "Person Asking for Help": 3.0, "Explosion": 1.25}, "tuned_on": "validation split", "validation_effect": {"before": {"critical_recall_min": 0.6444444444444445, "f1_macro": 0.7083256288598374}, "after": {"critical_recall_min": 0.7111111111111111, "f1_macro": 0.719154616041807}}}`

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.793 | 0.705 | 0.753 | 0.720 | 0.667 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.894 | 0.933 | 0.913 | 45 |
| Glass Breaking | 0.925 | 0.961 | 0.942 | 51 |
| Alarm or Siren | 0.934 | 0.904 | 0.919 | 156 |
| Vehicle Horn | 0.950 | 0.872 | 0.909 | 109 |
| Dog Bark | 0.842 | 0.774 | 0.807 | 62 |
| Cat | 0.786 | 0.786 | 0.786 | 14 |
| Bird | 0.725 | 0.725 | 0.725 | 40 |
| Farm Animal | 0.706 | 0.632 | 0.667 | 19 |
| Wild Animal | 0.833 | 0.909 | 0.870 | 22 |
| Insect or Frog | 0.333 | 0.538 | 0.412 | 26 |
| Gunshot | 0.924 | 0.730 | 0.816 | 100 |
| Panic Scream | 0.911 | 0.911 | 0.911 | 45 |
| Aggression | 0.902 | 0.822 | 0.860 | 45 |
| Person Asking for Help | 0.714 | 0.667 | 0.690 | 45 |
| Drone Sound | 1.000 | 1.000 | 1.000 | 45 |
| Drilling or Grinder Sound | 0.835 | 0.655 | 0.734 | 116 |
| Fireworks | 0.852 | 0.902 | 0.876 | 51 |
| Vehicle Backfire | 0.900 | 0.800 | 0.847 | 45 |
| Explosion | 0.741 | 0.889 | 0.808 | 45 |
| Door Impact | 0.719 | 0.821 | 0.767 | 56 |
| Graffiti Spray | 1.000 | 0.978 | 0.989 | 45 |
| Human Speech | 0.840 | 0.913 | 0.875 | 69 |
| Crying Baby | 0.600 | 1.000 | 0.750 | 6 |
| Human Body Sound | 0.574 | 0.861 | 0.689 | 36 |
| Laughter or Applause | 0.636 | 0.583 | 0.609 | 12 |
| Footsteps | 0.333 | 0.500 | 0.400 | 6 |
| Keyboard or Mouse | 0.625 | 0.556 | 0.588 | 18 |
| Household Appliance | 0.636 | 0.840 | 0.724 | 25 |
| Rain or Water | 0.478 | 0.688 | 0.564 | 32 |
| Wind | 0.700 | 0.778 | 0.737 | 9 |
| Thunderstorm | 0.667 | 1.000 | 0.800 | 6 |
| Fire Crackling | 0.833 | 0.833 | 0.833 | 6 |
| Church Bell | 0.333 | 0.500 | 0.400 | 6 |
| Helicopter | 0.000 | 0.000 | 0.000 | 6 |
| Train or Aircraft | 0.273 | 0.500 | 0.353 | 12 |
| Engine Running | 0.375 | 0.500 | 0.429 | 6 |
| Background Noise | 0.739 | 0.590 | 0.656 | 144 |

## Model comparison

| Model | CV macro F1 | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Histogram Gradient Boosting | 0.672 | 0.708 | 0.793 | 0.721 | 0.622 |  |
| Histogram Gradient Boosting + critical-class weighting | 0.672 | 0.715 | 0.792 | 0.720 | 0.667 | yes |

## Live-microphone rendering (same recordings, as the live monitor delivers them)

| Split | Accuracy | Macro F1 | Min critical recall |
|---|---|---|---|
| validation | 0.760 | 0.690 | 0.689 |
| test | 0.783 | 0.716 | 0.640 |

## Confidence calibration

Temperature 1.329 fitted on the validation segments. Expected calibration error on the test segments 0.0753 -> 0.016, mean confidence 0.8089 -> 0.7313 against accuracy 0.7336. The predicted class does not change.
