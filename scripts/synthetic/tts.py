"""Text-to-speech phrase library for speech-based classes.

The SRS allows speech samples to be "generated synthetically where permitted".
On Windows we use the built-in SAPI voices (System.Speech) through PowerShell.
Each rendered utterance is one *source group*: every clip derived from it is
kept in the same dataset split so the test set never contains a phrase
recording that was also used for training.

Speaker variety: SAPI offers only a few voices, so each clip also gets a
"vocal-tract" transformation (resampling by 0.85-1.25, which shifts pitch and
formants together like a different-sized speaker).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HELP_PHRASES = [
    "Help me!", "Help me.", "Somebody help!", "Somebody help me!", "Please help!",
    "Please help me!", "Call for help!", "Emergency!", "Help! Help!",
    "Emergency! Call for help!", "Somebody please help!", "Help me, please!",
]
ANGRY_PHRASES = [
    "Get out of here!", "Back off!", "Stop pushing me!", "What is your problem?",
    "Don't touch me!", "Shut up!", "Leave right now!", "You are lying!",
    "Don't you dare!", "I said get out!", "Move away from me!", "What did you just say?",
]
NEUTRAL_PHRASES = [
    "The meeting starts at three.", "Can you pass me the file?", "I will be there in ten minutes.",
    "Did you see the game last night?", "Let's get some coffee.", "The printer is out of paper.",
    "We need to finish the report today.", "How was your weekend?",
    "Please close the door when you leave.", "The bus is running late again.",
    "I think it will rain later.", "Thanks, that was really helpful.",
    # More ordinary conversation, so the Human Speech category (28 Sep 2026) has enough different
    # source groups to split into train, validation and test without a phrase leaking across splits.
    "Good morning, how are you today?", "I left the keys on the table.",
    "Could you send me that email again?", "The shop closes at eight in the evening.",
    "My train gets in around six o'clock.", "She said the delivery arrives tomorrow.",
    "Let me check the schedule and call you back.", "There is a new cafe near the station.",
    "I finished reading that book last week.", "The weather has been strange this month.",
    "We should book the tickets before they sell out.", "He works in the office on the second floor.",
    "Do you know where the nearest pharmacy is?", "I need to charge my phone before we leave.",
    "The children are playing in the garden.", "That presentation went better than I expected.",
    "Remind me to water the plants tonight.", "It took almost an hour to get here.",
    "They moved to a bigger apartment last spring.", "I usually have breakfast around seven.",
    "The document is saved in the shared folder.", "Are you joining the call this afternoon?",
]
RATES = {"help": [-2, 0, 2, 4], "angry": [0, 2, 4], "neutral": [-1, 1]}
PHRASES = {"help": HELP_PHRASES, "angry": ANGRY_PHRASES, "neutral": NEUTRAL_PHRASES}

_PS_SCRIPT = r"""
param([string]$JobsFile)
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -Path $JobsFile | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($j in $jobs) {
  $synth.SelectVoice($j.voice)
  $synth.Rate = [int]$j.rate
  $synth.Volume = 100
  $synth.SetOutputToWaveFile($j.path, $fmt)
  $synth.Speak($j.text)
  $synth.SetOutputToNull()
}
$synth.Dispose()
"""

_LIST_VOICES = (
    "Add-Type -AssemblyName System.Speech; "
    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
    "ForEach-Object { $_.VoiceInfo.Name }"
)


def _powershell() -> str | None:
    if os.name != "nt":
        return None
    return "powershell"


def list_voices() -> list[str]:
    ps = _powershell()
    if not ps:
        return []
    try:
        out = subprocess.run([ps, "-NoProfile", "-Command", _LIST_VOICES], capture_output=True, text=True, timeout=60)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


@dataclass
class Utterance:
    group: str
    category: str
    text: str
    voice: str
    rate: int
    path: str


@dataclass
class TTSLibrary:
    cache_dir: Path
    utterances: dict[str, list[Utterance]] = field(default_factory=dict)
    _audio: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return any(self.utterances.values())

    @classmethod
    def build(cls, cache_dir: Path, log=print) -> "TTSLibrary":
        """The rendered phrase library, rendering only what the cache is missing.

        Adding a phrase to ``PHRASES`` (or installing another voice) therefore
        renders just the new utterances instead of re-rendering everything, and
        a phrase whose text was edited is rendered again.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = cache_dir / "manifest.json"
        lib = cls(cache_dir)
        cached: dict[str, Utterance] = {}
        if manifest.exists():
            for items in json.loads(manifest.read_text(encoding="utf-8")).values():
                for u in items:
                    if Path(u["path"]).exists():
                        cached[u["group"]] = Utterance(**u)
        voices = list_voices()
        if not voices:
            for u in cached.values():
                lib.utterances.setdefault(u.category, []).append(u)
            if lib.available:
                log(f"No text-to-speech voices found; using the {len(cached)} cached utterances.")
            else:
                log("No text-to-speech voices found: speech classes will use formant synthesis.")
            return lib
        jobs = []
        for cat, phrases in PHRASES.items():
            lib.utterances[cat] = []
            for pi, text in enumerate(phrases):
                for vi, voice in enumerate(voices):
                    for rate in RATES[cat]:
                        group = f"tts-{cat}-p{pi:02d}-v{vi}-r{rate:+d}"
                        path = str(cache_dir / cat / f"{group}.wav")
                        have = cached.get(group)
                        lib.utterances[cat].append(Utterance(group, cat, text, voice, rate, path))
                        if have is None or have.text != text:
                            Path(path).parent.mkdir(parents=True, exist_ok=True)
                            jobs.append({"voice": voice, "rate": rate, "text": text, "path": path})
        if not jobs:
            log(f"TTS library loaded from cache ({sum(len(v) for v in lib.utterances.values())} utterances).")
            return lib
        log(f"Rendering {len(jobs)} TTS utterances with voices: {', '.join(voices)}")
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tts.ps1"
            script.write_text(_PS_SCRIPT, encoding="utf-8")
            jobs_file = Path(tmp) / "jobs.json"
            jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-JobsFile", str(jobs_file)],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0:
                log("TTS rendering failed: " + proc.stderr[:300])
        for cat in list(lib.utterances):
            lib.utterances[cat] = [u for u in lib.utterances[cat] if Path(u.path).exists() and Path(u.path).stat().st_size > 1000]
        manifest.write_text(json.dumps({k: [u.__dict__ for u in v] for k, v in lib.utterances.items()}, indent=1), encoding="utf-8")
        log(f"TTS library ready: {sum(len(v) for v in lib.utterances.values())} utterances.")
        return lib

    def _load(self, utt: Utterance, sr: int) -> np.ndarray:
        key = f"{utt.path}@{sr}"
        if key not in self._audio:
            import soundfile as sf

            from audio_preprocessing.loader import resample
            from audio_preprocessing.preprocess import trim_silence

            y, file_sr = sf.read(utt.path, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if file_sr != sr:
                y = resample(y, file_sr, sr)
            y, _, _ = trim_silence(y, 45.0, 1024, 256)
            self._audio[key] = y / (np.max(np.abs(y)) + 1e-9)
        return self._audio[key]

    def sample(self, rng: np.random.Generator, category: str, sr: int, utterance: Utterance | None = None):
        """Return (audio, meta) for a random utterance with a random speaker transformation."""
        from audio_preprocessing.loader import resample

        pool = self.utterances.get(category) or []
        if not pool:
            raise RuntimeError(f"No TTS utterances for {category}")
        utt = utterance or pool[int(rng.integers(0, len(pool)))]
        y = self._load(utt, sr)
        factor = float(rng.uniform(0.85, 1.25))  # >1 = smaller vocal tract / higher pitch
        y = resample(y, sr, int(round(sr / factor)))
        return y.astype(np.float32), {
            "text": utt.text,
            "voice": f"{utt.voice} (x{factor:.2f})",
            "group": utt.group,
        }
