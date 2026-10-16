"""Lay out the competition submission: one self-contained "Source Code" folder.

The rule for what goes in is the submission brief's list, filtered the way a
public GitHub repository is filtered - i.e. .gitignore applies, so the 6.3 GB of
recorded audio and the 3.1 GB of feature caches stay out (the dataset is
published separately, see audio_dataset/README.md).

Two things are kept out of "Source Code" on purpose:

  * dist/ - build output (the deploy archives), not source;
  * the eleven superseded model binaries in python_models/saved/ - the brief
    asks for "Saved Python model", and shipping 130 MB of older experiments
    only makes the repository harder to clone. Their .json cards stay, so the
    experiment history is still readable.

Both land in a sibling folder instead of being deleted.

    python scripts/make_submission.py
    python scripts/make_submission.py --out D:/some/where
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The brief's list, plus the root files a repository needs to actually run,
# deploy and test (run.py / wsgi.py are the backend entry points, and the brief
# asks for "Python frontend and backend files"). scripts/ holds the
# model-training and dataset tooling the brief asks for as well.
EXTRA_ROOT_FILES = [
    "run.py", "wsgi.py", "Procfile", "Dockerfile", "render.yaml",
    ".env.example", ".gitignore", "pytest.ini", "requirements-gtm.txt", "HOW_TO_RUN.txt",
]


def tracked_files() -> list[str]:
    """Everything `git add -A` would stage: tracked + untracked, minus ignored."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def active_model() -> str | None:
    card = ROOT / "python_models" / "saved" / "active.json"
    if not card.is_file():
        return None
    return json.loads(card.read_text(encoding="utf-8")).get("file")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT.parent.parent / "submission"),
                    help="where to write the submission folders")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    src_dir = out / "Source Code"
    extra_dir = out / "Extra - not part of source submission"
    for d in (src_dir, extra_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    keep_model = active_model()
    if not keep_model:
        print("  warning: python_models/saved/active.json missing - no model kept")

    copied = skipped = 0
    bytes_in = bytes_out = 0
    for rel in tracked_files():
        srcp = ROOT / rel
        if not srcp.is_file():
            continue
        size = srcp.stat().st_size
        # superseded model binaries -> the sibling folder
        parts = Path(rel).parts
        is_old_model = (
            len(parts) == 3 and parts[0] == "python_models" and parts[1] == "saved"
            and rel.endswith(".joblib") and parts[2] != keep_model
        )
        target = (extra_dir if is_old_model else src_dir) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcp, target)
        if is_old_model:
            skipped += 1
            bytes_out += size
        else:
            copied += 1
            bytes_in += size

    # Directories the app writes into: keep them present but empty.
    for d in ("data/storage", "database"):
        p = src_dir / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    print(f"  Source Code : {copied} files, {bytes_in / 1e6:.1f} MB  -> {src_dir}")
    print(f"  Extra       : {skipped} files, {bytes_out / 1e6:.1f} MB  -> {extra_dir}")

    # Anything over GitHub's limits is worth knowing before the first push.
    big = sorted(
        ((p.stat().st_size, p.relative_to(src_dir).as_posix())
         for p in src_dir.rglob("*") if p.is_file() and p.stat().st_size > 50_000_000),
        reverse=True,
    )
    if big:
        print("\n  GitHub file-size check:")
        for size, rel in big:
            verdict = "REJECTED (>100 MB hard limit)" if size > 100_000_000 else "warning only (>50 MB)"
            print(f"    {size / 1e6:>7.1f} MB  {rel}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
