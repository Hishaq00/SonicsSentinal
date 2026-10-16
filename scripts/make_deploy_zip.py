"""Build a deployment archive of the web application.

What goes in is what the running app imports or serves: the Flask package, the
configuration, the templates and static assets, the browser (Teachable
Machine) model and - unless --slim is given - the one Python model that
python_models/saved/active.json points at.

What stays out is everything only the *training* side needs: the audio dataset,
the feature matrices, the stored uploads, the older model versions, the tests
and the notebooks. Those are hundreds of times larger than the app itself.

    python scripts/make_deploy_zip.py            # runs out of the box
    python scripts/make_deploy_zip.py --slim     # no Python model (small)

Both write into dist/.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# Whole directories the application needs at runtime.
TREES = [
    "src",                  # the Flask application
    "config",               # class list + default settings
    "alert_rules",          # the rule engine and its rules
    "audio_preprocessing",  # loading / resampling / quality checks
    "feature_extraction",   # the features the Python model expects
    "augmentation",         # imported by the pipeline
    "templates",
    "static",
    "gtm_model/model",      # the browser model: served to the client
    "documentation",
    "reports",              # the admin reports page serves these
]

# Individual files.
FILES = [
    "run.py", "wsgi.py", "requirements.txt", "requirements-gtm.txt",
    "Procfile", "render.yaml", "Dockerfile", ".env.example", "pytest.ini",
    "README.md", "LICENSE", "HOW_TO_RUN.txt", "AI_USAGE.md",
    "python_models/saved/active.json",
    # Shown on the admin dataset page; the models carry their own feature config.
    "data/dataset_metadata.csv",
    "data/dataset_statistics.json",
    "data/features/feature_config.json",
]

# Every .py directly inside these packages (their subdirectories are data).
FLAT_PY = ["python_models", "gtm_model"]

# Directories the application writes into: shipped empty so the first boot has
# somewhere to put the database and the uploads.
EMPTY_DIRS = ["database", "data/storage"]

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", ".idea", "node_modules"}
SKIP_SUFFIX = {".pyc", ".pyo", ".pyd"}
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".env"}

# Already-compressed payloads: deflating them again costs time and saves nothing.
STORE_SUFFIX = {".joblib", ".bin", ".woff2", ".png", ".jpg", ".jpeg", ".webp", ".zip", ".npz", ".gz"}


def wanted(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIX:
        return False
    return True


def collect(slim: bool) -> list[Path]:
    picked: list[Path] = []
    missing: list[str] = []

    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            missing.append(tree)
            continue
        picked += [p for p in base.rglob("*") if p.is_file() and wanted(p)]

    for name in FILES:
        p = ROOT / name
        if p.is_file():
            picked.append(p)
        else:
            missing.append(name)

    for pkg in FLAT_PY:
        base = ROOT / pkg
        if base.is_dir():
            picked += sorted(base.glob("*.py"))

    if not slim:
        active = ROOT / "python_models" / "saved" / "active.json"
        if active.is_file():
            card = json.loads(active.read_text(encoding="utf-8"))
            for name in (card.get("file"), f"{card.get('version')}.json"):
                p = ROOT / "python_models" / "saved" / str(name)
                if p.is_file():
                    picked.append(p)
                else:
                    missing.append(f"python_models/saved/{name}")
        else:
            missing.append("python_models/saved/active.json")

    if missing:
        print("  note: not found, skipped -> " + ", ".join(missing))
    # rglob and the explicit lists can overlap (e.g. active.json).
    return sorted(set(picked))


def build(slim: bool) -> Path:
    label = "slim" if slim else "full"
    out = DIST / ("sonicsentinel-deploy-slim.zip" if slim else "sonicsentinel-deploy.zip")
    DIST.mkdir(parents=True, exist_ok=True)
    files = collect(slim)

    raw = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            assert not rel.endswith("/.env") and rel != ".env", "refusing to package .env"
            raw += p.stat().st_size
            mode = zipfile.ZIP_STORED if p.suffix.lower() in STORE_SUFFIX else zipfile.ZIP_DEFLATED
            z.write(p, rel, compress_type=mode)
        for d in EMPTY_DIRS:
            z.writestr(f"{d}/.gitkeep", "")

    size = out.stat().st_size
    print(f"  {label}: {out.relative_to(ROOT).as_posix()}")
    print(f"    {len(files)} files, {raw / 1e6:.1f} MB uncompressed -> {size / 1e6:.1f} MB zipped")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slim", action="store_true",
                    help="leave the trained Python model out (train or upload it on the server)")
    ap.add_argument("--both", action="store_true", help="build the full and the slim archive")
    args = ap.parse_args()

    if args.both:
        build(slim=False)
        build(slim=True)
    else:
        build(slim=args.slim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
