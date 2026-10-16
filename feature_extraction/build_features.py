"""Extract features for every dataset recording, in both renderings the app produces.

    python -m feature_extraction.build_features [--force] [--no-live]

**upload** - how an uploaded file is analysed:
    decode -> pre-process (same settings as the web app) -> fixed-duration
    segmentation -> keep the segments that contain the event -> features.

**live** - how a microphone window is analysed (``feature_extraction/live_render.py``):
    decode -> microphone simulation (band limit, self-noise, browser gain control)
    -> 22.05 kHz -> windows -> pre-process each window on its own, no trimming
    -> features. Without these rows the model only ever sees trimmed, whole-file
    normalised audio and is weaker on the live monitor than on uploads.

Each row says which rendering it comes from (``rendering``). Training uses both;
the models are selected and reported on the upload rendering of the validation
and test splits, and the live rendering of those splits is reported separately.

All segments derived from one recording keep its Audio ID, class label and
split. Output:
    data/features/features.npz   (matrix used for training)
    data/features/features.csv   (same data, human-readable)
    data/features/feature_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from audio_preprocessing import PreprocessConfig, decode_audio, preprocess_audio, segment_audio, select_event_segments
from config import DATASET_DIR, FEATURES_DIR, METADATA_CSV, load_default_settings
from feature_extraction.features import FeatureConfig, extract_features, feature_names
from feature_extraction.live_render import live_windows, seed_from

MAX_SEGMENTS = 8
LIVE_MAX_WINDOWS = 6  # windows per recording in the live rendering


def segments_for_file(path, class_label: str, pre_cfg: PreprocessConfig, seg_s: float, hop_s: float):
    dec = decode_audio(path)
    pre = preprocess_audio(dec.samples, dec.sample_rate, pre_cfg)
    segs = segment_audio(pre.y, pre.sr, seg_s, hop_s)
    if class_label != "Background Noise":
        segs = select_event_segments(segs, keep_within_db=15.0)
    if len(segs) > MAX_SEGMENTS:
        segs = sorted(sorted(segs, key=lambda s: -s.rms_dbfs)[:MAX_SEGMENTS], key=lambda s: s.index)
    return segs, pre.sr


def _row_meta(row: dict, s, rendering: str, mic: str = "") -> dict:
    return {
        "audio_id": row["audio_id"],
        "root_audio_id": row["root_audio_id"] or row["audio_id"],
        "class_label": row["class_label"],
        "split": row["split"],
        "status": row["status"],
        "rendering": rendering,
        "microphone": mic,
        "segment_index": s.index,
        "start_s": round(s.start, 3),
        "end_s": round(s.end, 3),
        "segment_rms_dbfs": round(s.rms_dbfs, 1),
    }


def _process(row: dict, pre_cfg: PreprocessConfig, seg_s: float, hop_s: float, feat_cfg: FeatureConfig,
             live: bool = True):
    path = DATASET_DIR / row["relative_path"]
    feats, metas = [], []
    try:
        segs, _ = segments_for_file(path, row["class_label"], pre_cfg, seg_s, hop_s)
        for s in segs:
            feats.append(extract_features(s.audio, feat_cfg))
            metas.append(_row_meta(row, s, "upload"))
    except Exception as exc:  # a broken file must not stop the whole run
        return [], [], f"{row['audio_id']}: {exc}"
    if not live:
        return feats, metas, None
    try:
        dec = decode_audio(path)
        windows, mic = live_windows(dec.samples, dec.sample_rate, seed_from(row["audio_id"]), pre_cfg, seg_s, hop_s)
        if row["class_label"] != "Background Noise":  # only windows that contain the event carry its label
            keep = {s.index for s in select_event_segments([s for s, _ in windows], keep_within_db=15.0)}
            windows = [(s, y) for s, y in windows if s.index in keep]
        if len(windows) > LIVE_MAX_WINDOWS:
            loudest = sorted(windows, key=lambda w: -w[0].rms_dbfs)[:LIVE_MAX_WINDOWS]
            windows = sorted(loudest, key=lambda w: w[0].index)
        for s, y in windows:
            feats.append(extract_features(y, feat_cfg))
            metas.append(_row_meta(row, s, "live", mic))
    except Exception as exc:
        return feats, metas, f"{row['audio_id']} (live rendering): {exc}"
    return feats, metas, None


def config_fingerprint(df: pd.DataFrame, pre_cfg: PreprocessConfig, seg: tuple, feat_cfg: FeatureConfig,
                       live: bool) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([pre_cfg.to_dict(), seg, feat_cfg.to_dict(), live, LIVE_MAX_WINDOWS],
                        sort_keys=True).encode())
    # The class matters too: it decides which segments are kept (all of them for Background Noise).
    h.update("".join(sorted(df["audio_id"] + df["split"] + df["class_label"] + df["sha256"].fillna(""))).encode())
    return h.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--no-live", action="store_true",
                        help="only the upload rendering (faster, but weaker on the live monitor)")
    args = parser.parse_args(argv)
    live = not args.no_live

    settings = load_default_settings()
    pre_cfg = PreprocessConfig.from_settings(settings)
    seg_s = float(settings["segmentation"]["segment_seconds"])
    hop_s = float(settings["segmentation"]["segment_hop_seconds"])
    feat_cfg = FeatureConfig(sr=pre_cfg.target_sr, segment_seconds=seg_s)

    df = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    fp = config_fingerprint(df, pre_cfg, (seg_s, hop_s), feat_cfg, live)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = FEATURES_DIR / "feature_config.json"
    npz_path = FEATURES_DIR / "features.npz"
    if npz_path.exists() and cfg_path.exists() and not args.force:
        if json.loads(cfg_path.read_text()).get("fingerprint") == fp:
            print("Features are up to date (use --force to rebuild).")
            return 0

    rows = df.to_dict("records")
    print(f"Extracting features for {len(rows)} recordings "
          f"({seg_s:g} s segments, {hop_s:g} s hop, {len(feature_names(feat_cfg))} features, "
          f"{'upload + live microphone renderings' if live else 'upload rendering only'}) ...")
    t0 = time.time()
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=args.jobs, verbose=5, batch_size=8)(
        delayed(_process)(r, pre_cfg, seg_s, hop_s, feat_cfg, live) for r in rows
    )
    X, metas, errors = [], [], []
    for feats, meta, err in results:
        X.extend(feats)
        metas.extend(meta)
        if err:
            errors.append(err)
    X = np.vstack(X).astype(np.float32)
    meta_df = pd.DataFrame(metas)
    names = feature_names(feat_cfg)
    np.savez_compressed(
        npz_path,
        X=X,
        y=meta_df["class_label"].to_numpy(),
        audio_id=meta_df["audio_id"].to_numpy(),
        root_audio_id=meta_df["root_audio_id"].to_numpy(),
        split=meta_df["split"].to_numpy(),
        status=meta_df["status"].to_numpy(),
        rendering=meta_df["rendering"].to_numpy(),
        segment_index=meta_df["segment_index"].to_numpy(),
        feature_names=np.array(names),
    )
    out = pd.concat([meta_df, pd.DataFrame(X, columns=names).round(5)], axis=1)
    out.to_csv(FEATURES_DIR / "features.csv", index=False)
    cfg_path.write_text(json.dumps({
        "fingerprint": fp,
        "preprocessing": pre_cfg.to_dict(),
        "segment_seconds": seg_s,
        "segment_hop_seconds": hop_s,
        "feature_config": feat_cfg.to_dict(),
        "n_segments": int(len(X)),
        "n_features": len(names),
        "renderings": meta_df["rendering"].value_counts().to_dict(),
        "live_rendering": live,
        "live_max_windows": LIVE_MAX_WINDOWS,
        "errors": errors,
    }, indent=2), encoding="utf-8")
    print(f"{len(X)} segments x {X.shape[1]} features in {time.time() - t0:.0f} s. Errors: {len(errors)}")
    for e in errors[:10]:
        print("  ", e)
    print(meta_df.groupby(["split", "status", "rendering"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
