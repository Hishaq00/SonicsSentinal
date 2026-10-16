# Performance (SRS 1.7-1)

Measured 2026-09-29 03:53 UTC over 5 runs on Windows-11-10.0.26200-SP0, Python 3.14.7, Python model `py-v12-ensemble-weighted`, GTM `SonicSentinel_GTM_v10@2026-09-28T17:35:28Z`.

| Measurement | Median | Worst | Target | Result |
|---|---|---|---|---|
| 30 s upload: server pipeline | 1.20 s | 1.33 s | <= 8 s | met |
| 30 s upload: GTM on the same segments (NumPy runner) | 0.28 s | 0.28 s | <= 8 s | met |
| 30 s upload: total (server + GTM) | 1.47 s | 1.59 s | <= 8 s | met |
| Live 2 s window: server prediction | 0.14 s | 0.16 s | <= 3 s | met |

Server pipeline = validation, pre-processing, quality analysis, waveform and spectrogram images, features, Python model, rule engine and database writes. In the web app the GTM model runs in the browser at the same time as the server works, so the user-visible time is at most the total above. Times exclude the one-time model load at start-up.
