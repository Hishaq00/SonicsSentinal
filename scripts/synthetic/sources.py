"""Procedural sound generators, one per sound category.

Each generator returns ``(dry_signal, info)`` where ``dry_signal`` is the
event recorded close to the source (no room, no background, no device) and
``info`` describes the variant that was produced. The clip builder in
``scripts/generate_synthetic_dataset.py`` adds the environment, distance,
background interference and recording-device effects afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .dsp import (
    SR,
    adsr,
    bandpass,
    colored_noise,
    exp_env,
    fade,
    glottal_source,
    harmonic_tone,
    highpass,
    lowpass,
    modal_impact,
    n_samples,
    peak_normalize,
    peaking_eq,
    phase_from_freq,
    place,
    smooth_random,
    thud,
    vowel_sequence,
    white,
)


def _choice(rng: np.random.Generator, options, p=None):
    return options[int(rng.choice(len(options), p=p))]


# =========================================================================== #
# Machinery
# =========================================================================== #
def machine_base(rng: np.random.Generator, n: int, sr: int = SR) -> tuple[np.ndarray, float]:
    """Healthy machine: mains hum, rotor harmonics, blade-pass tone and mechanical noise."""
    t = np.arange(n) / sr
    rot = rng.uniform(18, 55)
    mains = float(rng.choice([50.0, 60.0]))
    hum = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * k * mains * t + rng.uniform(0, 6.28)) for k in range(1, 7))
    rotor = sum(rng.uniform(0.3, 1.0) / k ** 0.8 * np.sin(2 * np.pi * k * rot * t + rng.uniform(0, 6.28)) for k in range(1, 9))
    blades = int(rng.integers(3, 9))
    blade = np.sin(2 * np.pi * blades * rot * t) * (1 + 0.3 * np.sin(2 * np.pi * rot * t))
    mech = bandpass(colored_noise(rng, n, 1.0), 150, rng.uniform(2500, 6000), sr)
    base = 0.35 * hum / np.std(hum) + 0.45 * rotor / np.std(rotor) + 0.25 * blade + rng.uniform(0.3, 0.7) * mech / np.std(mech)
    return base, rot


def _bearing_fault(rng, n, sr, rot):
    rate = rot * rng.uniform(3.0, 5.5)
    res, decay = rng.uniform(2200, 6500), rng.uniform(0.0015, 0.005)
    imp = np.zeros(n)
    period = sr / rate
    pos = rng.uniform(0, period)
    while pos < n:
        imp[int(pos)] += rng.uniform(0.6, 1.4)
        pos += period * (1 + rng.normal(0, 0.02))
    k = np.arange(n_samples(0.03, sr)) / sr
    kernel = np.sin(2 * np.pi * res * k) * np.exp(-k / decay)
    t = np.arange(n) / sr
    return fftconvolve(imp, kernel)[:n] * (1 + 0.5 * np.sin(2 * np.pi * rot * t))


def _knock_fault(rng, n, sr, rot):
    out = np.zeros(n)
    period = sr / rot * float(rng.choice([1, 2, 3]))
    pos = rng.uniform(0, period)
    base_freqs = np.sort(rng.uniform(600, 3500, int(rng.integers(4, 7))))
    while pos < n:
        if rng.random() < rng.uniform(0.5, 0.95):
            freqs = base_freqs * rng.uniform(0.97, 1.03)
            hit = modal_impact(rng, freqs, rng.uniform(0.01, 0.06, len(freqs)), dur=0.12, sr=sr)
            place(out, hit, int(pos), rng.uniform(0.5, 1.2))
        pos += period * (1 + rng.normal(0, 0.05))
    return out


def _grinding_fault(rng, n, sr, rot):
    t = np.arange(n) / sr
    noise = bandpass(white(rng, n), rng.uniform(900, 1800), rng.uniform(3500, 7000), sr)
    am = (0.5 + 0.5 * np.abs(np.sin(np.pi * rot * t))) ** rng.uniform(1, 3)
    am *= 1 + 0.4 * np.sin(2 * np.pi * rng.uniform(80, 200) * t)
    return noise * am / (np.std(noise) + 1e-9)


def _squeal_fault(rng, n, sr, rot):
    f = rng.uniform(1800, 4200) * (1 + 0.03 * smooth_random(rng, n, rng.uniform(3, 8), sr))
    tone = harmonic_tone(f, sr, n_harm=3, tilt_db=-10, rng=rng)
    gate = (smooth_random(rng, n, 2.0, sr) > rng.uniform(-0.4, 0.2)).astype(float)
    gate = lowpass(gate, 20, sr, 2)
    return tone * np.clip(gate, 0, 1)


def _rattle_fault(rng, n, sr, rot):
    out = np.zeros(n)
    count = int(rng.uniform(15, 40) * n / sr)
    for _ in range(count):
        freqs = rng.uniform(2500, 8000, 3)
        hit = modal_impact(rng, freqs, rng.uniform(0.002, 0.01, 3), dur=0.03, sr=sr)
        place(out, hit, int(rng.integers(0, n)), rng.uniform(0.2, 0.8))
    return out


def _imbalance_fault(rng, n, sr, rot):
    t = np.arange(n) / sr
    wobble = np.sin(2 * np.pi * rot * t) + 0.7 * np.sin(2 * np.pi * 2 * rot * t + 1.0)
    wobble = np.tanh(2.5 * wobble)  # distortion from loose mounts
    thumps = np.zeros(n)
    period = sr / rot
    pos = 0.0
    while pos < n:
        place(thumps, thud(rng, rng.uniform(40, 90), 0.03, 0.12, sr), int(pos), rng.uniform(0.4, 1.0))
        pos += period * (1 + rng.normal(0, 0.03))
    return 0.6 * wobble + thumps


FAULTS = {
    "bearing-defect": _bearing_fault,
    "knocking": _knock_fault,
    "grinding": _grinding_fault,
    "belt-squeal": _squeal_fault,
    "loose-part-rattle": _rattle_fault,
    "imbalance": _imbalance_fault,
}


def machinery_fault(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    base, rot = machine_base(rng, n, sr)
    names = list(FAULTS)
    picked = [names[i] for i in rng.choice(len(names), size=int(rng.choice([1, 1, 2])), replace=False)]
    fault = np.zeros(n)
    for name in picked:
        f = FAULTS[name](rng, n, sr, rot)
        fault += f / (np.std(f) + 1e-9)
    y = base / (np.std(base) + 1e-9) + rng.uniform(0.7, 1.6) * fault
    return peak_normalize(fade(y, 0.05, 0.05, sr)), {"variant": "+".join(picked), "rotation_hz": round(rot, 1)}


def normal_machinery(rng: np.random.Generator, dur: float, sr: int = SR):
    base, rot = machine_base(rng, n_samples(dur, sr), sr)
    return peak_normalize(base), {"variant": "normal-machinery", "rotation_hz": round(rot, 1)}


# =========================================================================== #
# Glass
# =========================================================================== #
def glass_breaking(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    variant = _choice(rng, ["window-pane", "bottle", "glass-door", "drinking-glass"], [0.35, 0.25, 0.2, 0.2])
    length = min(dur, rng.uniform(0.9, 2.2))
    n = n_samples(length, sr)
    out = np.zeros(n)
    t0 = n_samples(rng.uniform(0.0, 0.04), sr)
    heavy = variant in ("window-pane", "glass-door")
    place(out, thud(rng, rng.uniform(70, 220), rng.uniform(0.02, 0.07), 0.3, sr), t0, rng.uniform(0.3, 0.9 if heavy else 0.4))
    crack_n = n_samples(0.09, sr)
    crack = highpass(white(rng, crack_n), rng.uniform(1200, 2500), sr) * exp_env(crack_n, rng.uniform(0.008, 0.03), sr)
    place(out, crack, t0, rng.uniform(0.6, 1.0))
    shards = int(rng.integers(25, 90)) if heavy else int(rng.integers(10, 40))
    spread = rng.uniform(0.12, 0.45)
    for _ in range(shards):
        at = t0 + int(rng.exponential(spread) * sr)
        if at >= n - n_samples(0.03, sr):
            continue
        k = int(rng.integers(2, 6))
        freqs = np.sort(rng.uniform(1800, 9500, k))
        tink = modal_impact(rng, freqs, rng.uniform(0.004, 0.04, k), rng.uniform(0.2, 1.0, k), dur=0.12, sr=sr)
        amp = rng.uniform(0.15, 0.6) * np.exp(-(at - t0) / (0.6 * sr))
        place(out, tink, at, amp)
    for _ in range(int(rng.integers(3, 12))):
        bn = n_samples(0.05, sr)
        burst = bandpass(white(rng, bn), 2000, 7500, sr) * exp_env(bn, rng.uniform(0.005, 0.02), sr)
        place(out, burst, t0 + int(rng.uniform(0, 0.5) * sr), rng.uniform(0.1, 0.4))
    if variant == "bottle":
        f = rng.uniform(850, 1500)
        place(out, modal_impact(rng, [f, f * 2.7, f * 5.1], [0.15, 0.08, 0.04], dur=0.4, sr=sr), t0, 0.5)
    return peak_normalize(out), {"variant": variant, "shards": shards}


# =========================================================================== #
# Alarms and sirens
# =========================================================================== #
def alarm_or_siren(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    variant = _choice(
        rng,
        ["wail-siren", "yelp-siren", "hi-lo-siren", "fire-alarm", "smoke-alarm", "alarm-bell", "car-alarm", "buzzer"],
        [0.18, 0.14, 0.14, 0.14, 0.1, 0.1, 0.12, 0.08],
    )
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    doppler = False
    if variant in ("wail-siren", "yelp-siren"):
        fc, dev = rng.uniform(700, 1150), rng.uniform(250, 500)
        period = rng.uniform(2.0, 5.0) if variant == "wail-siren" else rng.uniform(0.15, 0.35)
        ph = (t / period + rng.uniform(0, 1)) % 1.0
        shape = (1 - np.cos(2 * np.pi * ph)) / 2 if variant == "wail-siren" else ph
        f = fc - dev + 2 * dev * shape
        y = harmonic_tone(f, sr, n_harm=int(rng.integers(3, 8)), tilt_db=rng.uniform(-12, -5), rng=rng)
        doppler = rng.random() < 0.4
    elif variant == "hi-lo-siren":
        f1, ratio, step = rng.uniform(420, 750), rng.uniform(1.25, 1.42), rng.uniform(0.35, 0.7)
        f = np.where(((t + rng.uniform(0, step)) // step) % 2 == 0, f1, f1 * ratio)
        f = lowpass(f, 40, sr, 1)
        y = harmonic_tone(f, sr, n_harm=6, tilt_db=-6, rng=rng)
        doppler = rng.random() < 0.4
    elif variant in ("fire-alarm", "smoke-alarm"):
        f = np.full(n, rng.uniform(2700, 3500))
        tone = harmonic_tone(f, sr, n_harm=3, tilt_db=-9, odd_only=True, rng=rng)
        if variant == "fire-alarm":
            period, duty = rng.uniform(0.3, 1.0), rng.uniform(0.4, 0.6)
            gate = (((t + rng.uniform(0, period)) % period) < period * duty).astype(float)
        else:  # temporal-three pattern: three beeps then a pause
            cyc = (t + rng.uniform(0, 4.0)) % 4.0
            gate = ((cyc < 3.0) & ((cyc % 1.0) < 0.5)).astype(float)
        y = tone * lowpass(gate, 200, sr, 2)
    elif variant == "alarm-bell":
        y = np.zeros(n)
        f0 = rng.uniform(700, 1300)
        rate = rng.uniform(15, 25)
        pos = 0.0
        while pos < n:
            strike = modal_impact(rng, [f0, f0 * 2.32, f0 * 3.96, f0 * 5.4], [0.25, 0.12, 0.07, 0.04], dur=0.3, sr=sr)
            place(y, strike, int(pos), rng.uniform(0.8, 1.0))
            pos += sr / rate
    elif variant == "car-alarm":
        y = np.zeros(n)
        pos = 0
        while pos < n:
            pattern = _choice(rng, ["warble", "chirp", "beep"])
            seg = n_samples(rng.uniform(0.8, 1.8), sr)
            tt = np.arange(seg) / sr
            if pattern == "warble":
                f = rng.uniform(1500, 2300) + rng.uniform(250, 500) * np.sin(2 * np.pi * rng.uniform(10, 20) * tt)
            elif pattern == "chirp":
                cyc = rng.uniform(0.15, 0.3)
                f = 500 + 2000 * ((tt % cyc) / cyc)
            else:
                f = np.full(seg, rng.uniform(1000, 1500))
            s = harmonic_tone(f, sr, n_harm=4, tilt_db=-6, rng=rng)
            if pattern == "beep":
                s *= ((tt % 0.2) < 0.1)
            place(y, s, pos)
            pos += seg
    else:  # buzzer
        f = np.full(n, rng.uniform(380, 650))
        period = rng.uniform(0.5, 1.0)
        gate = ((t % period) < period * 0.6).astype(float)
        y = harmonic_tone(f, sr, n_harm=25, tilt_db=-4, rng=rng) * lowpass(gate, 100, sr, 2)
    if doppler:
        # Passing vehicle: amplitude swell (the pitch drift is part of the recording geometry).
        centre = rng.uniform(0.3, 0.7)
        y *= 0.35 + 0.65 * np.exp(-((t / max(t[-1], 1e-6) - centre) ** 2) / 0.05)
    return peak_normalize(fade(y, 0.02, 0.05, sr)), {"variant": variant, "doppler": doppler}


# =========================================================================== #
# Vehicle horns
# =========================================================================== #
def _honk(rng, freqs, length, sr, n_harm, tilt, air=False):
    n = n_samples(length, sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for f in freqs:
        curve = f * (1 + 0.004 * np.sin(2 * np.pi * 5 * t)) * (1 - 0.03 * np.exp(-t / 0.03))
        y += harmonic_tone(curve, sr, n_harm=n_harm, tilt_db=tilt, rng=rng)
    y *= adsr(n, rng.uniform(0.008, 0.03), 0.05, 0.85, rng.uniform(0.03, 0.08), sr)
    if air:
        an = min(n, n_samples(0.15, sr))
        y[:an] += 0.6 * bandpass(white(rng, an), 500, 3000, sr) * exp_env(an, 0.05, sr)
    return y


def vehicle_horn(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    variant = _choice(rng, ["car", "truck-air-horn", "motorbike", "bus", "scooter"], [0.4, 0.15, 0.15, 0.15, 0.15])
    if variant == "car":
        f1 = rng.uniform(360, 460)
        freqs = [f1, f1 * rng.uniform(1.18, 1.32)]
    elif variant == "truck-air-horn":
        f1 = rng.uniform(170, 320)
        freqs = [f1, f1 * 1.25, f1 * 1.5][: int(rng.integers(2, 4))]
    elif variant == "bus":
        f1 = rng.uniform(260, 380)
        freqs = [f1, f1 * 1.26, f1 * 1.5]
    elif variant == "motorbike":
        freqs = [rng.uniform(520, 760)]
    else:
        freqs = [rng.uniform(700, 950)]
    pattern = _choice(rng, ["single", "double", "long", "triple", "rapid"], [0.3, 0.25, 0.2, 0.1, 0.15])
    lengths = {
        "single": [rng.uniform(0.25, 0.6)],
        "double": [rng.uniform(0.15, 0.3)] * 2,
        "long": [rng.uniform(0.8, 2.0)],
        "triple": [rng.uniform(0.15, 0.3)] * 3,
        "rapid": [rng.uniform(0.08, 0.14)] * int(rng.integers(4, 7)),
    }[pattern]
    gap = rng.uniform(0.08, 0.25)
    total = sum(lengths) + gap * (len(lengths) - 1) + 0.1
    out = np.zeros(n_samples(min(max(total, 0.3), dur), sr))
    pos = 0
    n_harm, tilt = int(rng.integers(18, 32)), rng.uniform(-7, -3.5)
    for length in lengths:
        place(out, _honk(rng, freqs, length, sr, n_harm, tilt, air=(variant == "truck-air-horn")), pos)
        pos += n_samples(length + gap, sr)
    return peak_normalize(out), {"variant": f"{variant}/{pattern}"}


# =========================================================================== #
# Animals
# =========================================================================== #
def _voiced(rng, f0, sr, vowels, shift=1.0, tilt=-8.0, jitter=0.02, breath=0.1, bw=1.0):
    src = glottal_source(f0, rng, sr, tilt_db=tilt, jitter=jitter, breath=breath)
    return vowel_sequence(src, vowels, sr, shift=shift, bw_scale=bw)


def animal_sound(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    variant = _choice(rng, ["dog-bark", "cat-meow", "birds", "cow", "rooster", "crow", "sheep", "frog", "duck"],
                      [0.22, 0.12, 0.16, 0.08, 0.08, 0.1, 0.1, 0.07, 0.07])
    out = np.zeros(n_samples(dur, sr))
    if variant == "dog-bark":
        size = _choice(rng, ["small", "medium", "large"])
        fb = {"small": rng.uniform(700, 1000), "medium": rng.uniform(400, 650), "large": rng.uniform(220, 380)}[size]
        shift = {"small": 1.4, "medium": 1.1, "large": 0.8}[size]
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for _ in range(int(rng.integers(1, 5))):
            ln = n_samples(rng.uniform(0.08, 0.25), sr)
            u = np.linspace(0, 1, ln)
            f0 = fb * (1 + 0.35 * np.sin(np.pi * u))
            bark = _voiced(rng, f0, sr, [_choice(rng, ["a", "ae", "o"])], shift, -6, 0.03, 0.25, 1.5)
            bark *= adsr(ln, 0.01, 0.04, 0.6, 0.05, sr)
            place(out, bark, pos)
            pos += ln + n_samples(rng.uniform(0.15, 0.5), sr)
        info = {"variant": f"dog-bark/{size}"}
    elif variant == "cat-meow":
        ln = n_samples(min(dur, rng.uniform(0.5, 1.3)), sr)
        u = np.linspace(0, 1, ln)
        base = rng.uniform(420, 560)
        f0 = base * (1 + 0.5 * np.sin(np.pi * u) ** 1.5) * (1 + 0.1 * u)
        meow = _voiced(rng, f0, sr, ["i", "a", "o", "u"], rng.uniform(1.2, 1.45), -8, 0.02, 0.1)
        place(out, meow * adsr(ln, 0.05, 0.1, 0.8, 0.15, sr), n_samples(rng.uniform(0, 0.3), sr))
        info = {"variant": "cat-meow"}
    elif variant == "birds":
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for _ in range(int(rng.integers(3, 15))):
            ln = n_samples(rng.uniform(0.03, 0.12), sr)
            fs = rng.uniform(2000, 4500)
            f = np.linspace(fs, fs * rng.uniform(0.6, 1.6), ln)
            chirp = harmonic_tone(f, sr, n_harm=2, tilt_db=-14, rng=rng)
            if rng.random() < 0.3:
                chirp *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(20, 40) * np.arange(ln) / sr)
            place(out, chirp * adsr(ln, 0.005, 0.01, 0.8, 0.01, sr), pos, rng.uniform(0.4, 1.0))
            pos += ln + n_samples(rng.uniform(0.05, 0.3), sr)
        info = {"variant": "birds"}
    elif variant == "cow":
        ln = n_samples(min(dur, rng.uniform(1.0, 2.4)), sr)
        u = np.linspace(0, 1, ln)
        f0 = rng.uniform(85, 150) * (1 + 0.25 * np.sin(np.pi * u))
        moo = _voiced(rng, f0, sr, ["u", "o", "a", "o"], 0.8, -10, 0.02, 0.1)
        place(out, moo * adsr(ln, 0.15, 0.2, 0.8, 0.3, sr), n_samples(rng.uniform(0, 0.3), sr))
        info = {"variant": "cow"}
    elif variant == "rooster":
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for seconds, fa, fb2 in [(0.15, 400, 450), (0.25, 550, 600), (0.3, 600, 800), (0.6, 650, 480)]:
            ln = n_samples(seconds * rng.uniform(0.85, 1.15), sr)
            f0 = np.linspace(fa, fb2, ln) * rng.uniform(0.9, 1.15)
            part = _voiced(rng, f0, sr, ["a", "o"], 1.2, -5, 0.02, 0.15)
            place(out, part * adsr(ln, 0.02, 0.05, 0.85, 0.05, sr), pos)
            pos += ln
        info = {"variant": "rooster"}
    elif variant == "crow":
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for _ in range(int(rng.integers(2, 5))):
            ln = n_samples(rng.uniform(0.2, 0.35), sr)
            f0 = np.linspace(rng.uniform(550, 750), rng.uniform(420, 550), ln)
            caw = _voiced(rng, f0, sr, ["a"], 1.2, -6, 0.05, 0.6, 1.8)
            caw *= 1 + 0.5 * np.sin(2 * np.pi * 60 * np.arange(ln) / sr)
            place(out, caw * adsr(ln, 0.02, 0.05, 0.8, 0.05, sr), pos)
            pos += ln + n_samples(rng.uniform(0.15, 0.4), sr)
        info = {"variant": "crow"}
    elif variant == "sheep":
        ln = n_samples(min(dur, rng.uniform(0.6, 1.2)), sr)
        t = np.arange(ln) / sr
        f0 = rng.uniform(230, 380) * (1 + 0.03 * np.sin(2 * np.pi * 7 * t))
        bleat = _voiced(rng, f0, sr, ["e", "a"], 1.1, -7, 0.02, 0.15)
        bleat *= 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(6, 10) * t))
        place(out, bleat * adsr(ln, 0.03, 0.1, 0.8, 0.1, sr), n_samples(rng.uniform(0, 0.3), sr))
        info = {"variant": "sheep"}
    elif variant == "frog":
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for _ in range(int(rng.integers(2, 5))):
            ln = n_samples(rng.uniform(0.3, 0.8), sr)
            t = np.arange(ln) / sr
            tone = harmonic_tone(np.full(ln, rng.uniform(250, 600)), sr, n_harm=6, tilt_db=-6, rng=rng)
            pulses = (np.sin(2 * np.pi * rng.uniform(10, 20) * t) > 0.3).astype(float)
            place(out, tone * lowpass(pulses, 150, sr, 2) * adsr(ln, 0.02, 0.05, 0.9, 0.05, sr), pos)
            pos += ln + n_samples(rng.uniform(0.1, 0.4), sr)
        info = {"variant": "frog"}
    else:  # duck
        pos = n_samples(rng.uniform(0, 0.2), sr)
        for _ in range(int(rng.integers(2, 6))):
            ln = n_samples(rng.uniform(0.12, 0.2), sr)
            f0 = np.full(ln, rng.uniform(300, 450))
            quack = _voiced(rng, f0, sr, ["ae"], 1.0, -4, 0.04, 0.3, 1.3)
            quack = peaking_eq(quack, 1500, 8, 2.0, sr)
            place(out, quack * adsr(ln, 0.01, 0.03, 0.7, 0.03, sr), pos)
            pos += ln + n_samples(rng.uniform(0.08, 0.3), sr)
        info = {"variant": "duck"}
    return peak_normalize(out), info


# =========================================================================== #
# Gunshot
# =========================================================================== #
def _single_shot(rng, weapon, sr):
    n = n_samples(0.45, sr)
    t = np.arange(n) / sr
    lp = {"handgun": rng.uniform(4500, 8000), "rifle": rng.uniform(6000, 10000), "shotgun": rng.uniform(2500, 5000)}[weapon]
    tau = {"handgun": rng.uniform(0.006, 0.014), "rifle": rng.uniform(0.004, 0.009), "shotgun": rng.uniform(0.012, 0.025)}[weapon]
    noise = lowpass(white(rng, n), lp, sr, 2)
    env = np.minimum(1.0, t / 0.0003) * (np.exp(-t / tau) + db_gain(-15) * np.exp(-t / rng.uniform(0.04, 0.1)))
    blast = noise / (np.std(noise) + 1e-9) * env
    f = 120 * np.exp(-t / 0.05) + 45
    boom = np.sin(phase_from_freq(f, sr)) * np.exp(-t / rng.uniform(0.04, 0.1)) * rng.uniform(0.4, 1.0)
    shot = blast + boom
    if weapon == "rifle":  # supersonic crack (N-wave) just before the muzzle blast
        dn = n_samples(rng.uniform(0.0015, 0.003), sr)
        nwave = np.linspace(1, -1, dn) * rng.uniform(0.8, 1.2)
        lead = n_samples(rng.uniform(0.003, 0.03), sr)
        shot = np.concatenate([np.zeros(lead), shot])
        shot[:dn] += nwave
    if rng.random() < 0.3:  # mechanical action click
        click = modal_impact(rng, rng.uniform(2000, 4500, 3), [0.005, 0.004, 0.003], dur=0.02, sr=sr)
        place(shot, click, n_samples(rng.uniform(0.1, 0.3), sr), 0.1)
    return np.tanh(1.5 * shot / (np.max(np.abs(shot)) + 1e-9))


def db_gain(db):
    return 10 ** (db / 20)


def gunshot(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    weapon = _choice(rng, ["handgun", "rifle", "shotgun"], [0.5, 0.3, 0.2])
    shots = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.1, 0.05]))
    interval = rng.uniform(0.12, 0.8)
    out = np.zeros(n_samples(min(dur, 0.5 + shots * interval + 0.5), sr))
    pos = 0.0
    for _ in range(shots):
        place(out, _single_shot(rng, weapon, sr), int(pos * sr), rng.uniform(0.8, 1.0))
        pos += interval * (1 + rng.normal(0, 0.15))
    return peak_normalize(out), {"variant": f"{weapon}/{shots}-shot"}


# =========================================================================== #
# Human vocal events (synthetic voice)
# =========================================================================== #
def panic_scream(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    speaker = _choice(rng, ["female", "male", "child"], [0.5, 0.3, 0.2])
    base = {"female": rng.uniform(800, 1400), "male": rng.uniform(450, 850), "child": rng.uniform(1000, 1600)}[speaker]
    out = np.zeros(n_samples(dur, sr))
    pos = n_samples(rng.uniform(0.0, 0.2), sr)
    for i in range(int(rng.choice([1, 1, 2, 3]))):
        ln = n_samples(rng.uniform(0.6, 2.2) * (0.7 if i else 1.0), sr)
        if pos + ln > len(out):
            ln = len(out) - pos - n_samples(0.05, sr)
        if ln < n_samples(0.3, sr):
            break
        u = np.linspace(0, 1, ln)
        t = np.arange(ln) / sr
        contour = np.interp(u, [0, 0.15, 0.8, 1.0], [0.75, 1.0, 0.97, 0.8])
        f0 = base * contour * (1 + 0.05 * smooth_random(rng, ln, 3.0, sr))
        f0 *= 1 + rng.uniform(0.02, 0.04) * np.sin(2 * np.pi * rng.uniform(5, 8) * t)
        src = glottal_source(f0, rng, sr, tilt_db=rng.uniform(-6, -3), jitter=0.03, shimmer=0.15,
                             breath=rng.uniform(0.08, 0.25))
        src *= 1 + rng.uniform(0.2, 0.6) * np.sin(2 * np.pi * rng.uniform(40, 110) * t)  # roughness
        if rng.random() < 0.4:  # sub-harmonic / vocal fry segment
            src += 0.3 * harmonic_tone(f0 / 2, sr, n_harm=8, tilt_db=-6, rng=rng)
        shift = {"female": 1.25, "male": 1.1, "child": 1.35}[speaker] * rng.uniform(0.95, 1.1)
        vowels = [_choice(rng, ["a", "ae", "i", "e"])] if rng.random() < 0.6 else ["a", "ae"]
        scream = vowel_sequence(src, vowels, sr, shift=shift, bw_scale=2.0)
        env = adsr(ln, rng.uniform(0.03, 0.1), 0.1, rng.uniform(0.7, 0.95), rng.uniform(0.1, 0.3), sr)
        scream *= env * (1 + 0.15 * smooth_random(rng, ln, 10.0, sr))
        place(out, scream / (np.max(np.abs(scream)) + 1e-9), pos, 1.0 if i == 0 else rng.uniform(0.6, 0.9))
        pos += ln
        gasp = n_samples(rng.uniform(0.2, 0.4), sr)
        place(out, 0.15 * bandpass(white(rng, gasp), 1000, 5000, sr) * adsr(gasp, 0.05, 0.05, 0.8, 0.1, sr), pos)
        pos += gasp
    return peak_normalize(out), {"variant": f"scream/{speaker}", "speaker": speaker}


def shouting_voice(rng: np.random.Generator, length: float, sr: int, sex: str, effort: str = "shout"):
    """Speech-like babble: syllables with random vowels, consonant bursts and prosody."""
    n = n_samples(length, sr)
    out = np.zeros(n)
    if effort == "shout":
        f_base = rng.uniform(160, 280) if sex == "m" else rng.uniform(260, 420)
        tilt, rate = rng.uniform(-5, -2), rng.uniform(3.5, 6.0)
    else:
        f_base = rng.uniform(95, 140) if sex == "m" else rng.uniform(170, 240)
        tilt, rate = rng.uniform(-12, -9), rng.uniform(3.5, 5.0)
    shift = 1.0 if sex == "m" else 1.15
    pos = 0
    while pos < n - n_samples(0.1, sr):
        syl = n_samples(rng.uniform(0.7, 1.2) / rate, sr)
        u = np.linspace(0, 1, syl)
        emphasis = rng.uniform(0.0, 0.35)
        f0 = f_base * (1 + emphasis * np.sin(np.pi * u)) * rng.uniform(0.9, 1.1)
        src = glottal_source(f0, rng, sr, tilt_db=tilt, jitter=0.02, shimmer=0.1, breath=0.05)
        v = vowel_sequence(src, [_choice(rng, list("aeiou"))], sr, shift=shift)
        v *= adsr(syl, 0.02, 0.03, 0.9, 0.04, sr)
        cn = n_samples(rng.uniform(0.02, 0.06), sr)
        cons = bandpass(white(rng, cn), 2000, 6000, sr) * adsr(cn, 0.005, 0.01, 0.6, 0.01, sr)
        place(out, 0.3 * cons, pos)
        place(out, v / (np.max(np.abs(v)) + 1e-9), pos + cn // 2)
        pos += syl + cn
        if rng.random() < 0.15:  # pause between phrases
            pos += n_samples(rng.uniform(0.15, 0.4), sr)
    return out


def conflict_sounds(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    out = np.zeros(n)
    for _ in range(int(rng.integers(1, 5))):
        kind = _choice(rng, ["thud", "slap", "crash"])
        at = int(rng.integers(0, max(1, n - n_samples(0.3, sr))))
        if kind == "thud":
            place(out, thud(rng, rng.uniform(60, 120), rng.uniform(0.05, 0.12), 0.3, sr), at, rng.uniform(0.5, 1.0))
        elif kind == "slap":
            sn = n_samples(0.02, sr)
            place(out, highpass(white(rng, sn), 1000, sr) * exp_env(sn, 0.003, sr), at, rng.uniform(0.4, 0.9))
        else:
            for _ in range(int(rng.integers(2, 5))):
                freqs = rng.uniform(300, 3000, 4)
                place(out, modal_impact(rng, freqs, rng.uniform(0.02, 0.15, 4), dur=0.3, sr=sr),
                      at + n_samples(rng.uniform(0, 0.2), sr), rng.uniform(0.3, 0.8))
    return out


def shout_process(rng, y, sr):
    y = peaking_eq(y, rng.uniform(2000, 3000), rng.uniform(3, 6), 1.0, sr)
    drive = rng.uniform(1.5, 3.5)
    return np.tanh(drive * y / (np.max(np.abs(y)) + 1e-9)) / np.tanh(drive)


def aggression(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    n = n_samples(dur, sr)
    out = np.zeros(n)
    use_tts = tts is not None and tts.available and rng.random() < 0.45
    info: dict = {}
    if use_tts:
        pos = n_samples(rng.uniform(0, 0.2), sr)
        voices = []
        for i in range(int(rng.integers(2, 4))):
            clip, meta = tts.sample(rng, "angry", sr)
            clip = shout_process(rng, clip, sr)
            place(out, clip, pos, rng.uniform(0.7, 1.0))
            voices.append(meta["voice"])
            pos += int(len(clip) * rng.uniform(0.5, 0.9))  # next speaker interrupts
            info.setdefault("phrases", []).append(meta["text"])
            info["source_group"] = meta["group"]
        info["variant"] = "tts-argument"
        info["speaker"] = "+".join(voices)
    else:
        speakers = int(rng.integers(2, 5))
        for _ in range(speakers):
            sex = _choice(rng, ["m", "f"])
            ln = rng.uniform(0.6, 1.0) * dur
            v = shouting_voice(rng, ln, sr, sex)
            v = shout_process(rng, v, sr)
            place(out, v, n_samples(rng.uniform(0, max(0.01, dur - ln)), sr), rng.uniform(0.6, 1.0))
        info["variant"] = f"shouting/{speakers}-voices"
        info["speaker"] = f"{speakers} synthetic voices"
    if rng.random() < 0.45:
        out += 0.8 * conflict_sounds(rng, n, sr) * (np.max(np.abs(out)) + 1e-9)
        info["variant"] += "+physical-conflict"
    return peak_normalize(out), info


def _formant_help(rng, sr):
    """Fallback 'help' utterance by formant synthesis when no TTS voice exists."""
    f0 = rng.uniform(110, 240)
    parts = []
    h = n_samples(0.06, sr)
    parts.append(0.3 * bandpass(white(rng, h), 800, 4000, sr) * adsr(h, 0.01, 0.02, 0.7, 0.02, sr))
    e = n_samples(0.18, sr)
    parts.append(vowel_sequence(glottal_source(np.full(e, f0 * 1.1), rng, sr), ["e"], sr) * adsr(e, 0.02, 0.05, 0.9, 0.03, sr))
    l_ = n_samples(0.07, sr)
    parts.append(lowpass(glottal_source(np.full(l_, f0), rng, sr), 1200, sr) * 0.5)
    parts.append(np.zeros(n_samples(0.04, sr)))
    p = n_samples(0.02, sr)
    parts.append(0.4 * highpass(white(rng, p), 500, sr) * exp_env(p, 0.005, sr))
    y = np.concatenate(parts)
    return y / (np.max(np.abs(y)) + 1e-9)


def person_asking_for_help(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    out = np.zeros(n_samples(dur, sr))
    info: dict = {}
    pos = n_samples(rng.uniform(0, 0.25), sr)
    if tts is not None and tts.available:
        for i in range(int(rng.choice([1, 1, 2]))):
            clip, meta = tts.sample(rng, "help", sr)
            if rng.random() < 0.6:
                clip = shout_process(rng, clip, sr)  # shouted / distressed delivery
            if pos + len(clip) > len(out):
                break
            place(out, clip, pos, 1.0 if i == 0 else rng.uniform(0.7, 1.0))
            pos += len(clip) + n_samples(rng.uniform(0.15, 0.5), sr)
            info.setdefault("phrases", []).append(meta["text"])
            info["speaker"] = meta["voice"]
            info["source_group"] = meta["group"]
        info["variant"] = "tts-help-phrase"
    if not info.get("phrases"):
        for _ in range(int(rng.integers(1, 3))):
            clip = _formant_help(rng, sr)
            place(out, clip, pos)
            pos += len(clip) + n_samples(rng.uniform(0.2, 0.4), sr)
        info = {"variant": "formant-help", "phrases": ["Help"], "speaker": "formant-synth"}
    return peak_normalize(out), info


# =========================================================================== #
# Background / ambient beds
# =========================================================================== #
def _passing_car(rng, n, sr):
    t = np.arange(n) / sr
    centre, width = rng.uniform(0, t[-1] if n > 1 else 1), rng.uniform(0.8, 2.0)
    env = np.exp(-((t - centre) ** 2) / (2 * (width / 2.5) ** 2))
    swoosh = bandpass(colored_noise(rng, n, 1.0), 250, 3000, sr) * env
    engine = harmonic_tone(np.full(n, rng.uniform(60, 110)) * (1 + 0.05 * np.tanh((centre - t) * 2)), sr, 8, -8, rng=rng)
    return swoosh + 0.3 * engine * env


def background_bed(rng: np.random.Generator, n: int, sr: int = SR, kind: str | None = None, tts=None) -> tuple[np.ndarray, str]:
    kinds = ["traffic", "rain", "wind", "hvac-office", "crowd", "machinery-normal", "room-tone", "conversation", "street-mix"]
    kind = kind or _choice(rng, kinds)
    t = np.arange(n) / sr
    if kind == "traffic":
        y = lowpass(colored_noise(rng, n, 2.0), 600, sr) + 0.3 * bandpass(colored_noise(rng, n, 1.0), 200, 2500, sr)
        for _ in range(int(rng.integers(0, 4))):
            y += rng.uniform(0.5, 1.5) * _passing_car(rng, n, sr)
    elif kind == "rain":
        y = 0.4 * highpass(colored_noise(rng, n, 1.0), 800, sr)
        drops = np.zeros(n)
        idx = rng.integers(0, n, int(rng.uniform(300, 1500) * n / sr))
        drops[idx] = rng.uniform(0.2, 1.0, idx.size) * rng.choice([-1, 1], idx.size)
        y += highpass(drops, 2000, sr)
    elif kind == "wind":
        y = lowpass(colored_noise(rng, n, 2.0), rng.uniform(300, 900), sr) * (0.6 + 0.4 * smooth_random(rng, n, 0.3, sr))
        if rng.random() < 0.3:
            y += 0.2 * bandpass(white(rng, n), 600, 1200, sr) * (0.5 + 0.5 * smooth_random(rng, n, 0.5, sr))
    elif kind == "hvac-office":
        mains = float(rng.choice([50.0, 60.0]))
        y = 0.2 * sum(np.sin(2 * np.pi * k * mains * t) / k for k in range(1, 5))
        y += lowpass(colored_noise(rng, n, 1.0), 1500, sr)
        for _ in range(int(rng.integers(0, 12))):  # keyboard clicks
            place(y, modal_impact(rng, rng.uniform(2000, 6000, 2), [0.003, 0.002], dur=0.02, sr=sr),
                  int(rng.integers(0, n)), rng.uniform(0.1, 0.3))
    elif kind == "crowd":
        y = np.zeros(n)
        for _ in range(int(rng.integers(6, 13))):
            y += rng.uniform(0.3, 1.0) * shouting_voice(rng, n / sr, sr, _choice(rng, ["m", "f"]), effort="normal")
        y = lowpass(y, 4000, sr)
    elif kind == "machinery-normal":
        y, _ = machine_base(rng, n, sr)
    elif kind == "room-tone":
        y = 0.3 * colored_noise(rng, n, 1.0) + 0.1 * np.sin(2 * np.pi * float(rng.choice([50.0, 60.0])) * t)
    elif kind == "conversation":
        y = np.zeros(n)
        if tts is not None and tts.available and rng.random() < 0.7:
            pos = n_samples(rng.uniform(0, 0.3), sr)
            while pos < n:
                clip, _ = tts.sample(rng, "neutral", sr)
                place(y, clip, pos, rng.uniform(0.5, 1.0))
                pos += len(clip) + n_samples(rng.uniform(0.2, 0.8), sr)
        else:
            for _ in range(int(rng.integers(1, 3))):
                y += shouting_voice(rng, n / sr, sr, _choice(rng, ["m", "f"]), effort="normal")
        y += 0.05 * colored_noise(rng, n, 1.0)
    else:  # street-mix
        y = background_bed(rng, n, sr, "traffic")[0] + 0.5 * background_bed(rng, n, sr, "crowd")[0]
    return y / (np.std(y) + 1e-9), kind


def benign_transients(rng: np.random.Generator, n: int, sr: int = SR) -> tuple[np.ndarray, str]:
    """Everyday impacts that are *not* events of interest (hard negatives)."""
    kind = _choice(rng, ["door-close", "footsteps", "cup-clink", "knock", "clapping"])
    out = np.zeros(n)
    if kind == "door-close":
        at = int(rng.integers(0, max(1, n - n_samples(0.4, sr))))
        place(out, thud(rng, rng.uniform(60, 110), 0.08, 0.35, sr), at)
        place(out, modal_impact(rng, rng.uniform(1500, 4000, 3), [0.02, 0.015, 0.01], dur=0.05, sr=sr), at + n_samples(0.03, sr), 0.4)
    elif kind == "footsteps":
        pos = n_samples(rng.uniform(0, 0.3), sr)
        step = rng.uniform(0.45, 0.65)
        while pos < n:
            place(out, thud(rng, rng.uniform(80, 150), 0.03, 0.12, sr), pos, rng.uniform(0.4, 0.7))
            pos += n_samples(step * rng.uniform(0.95, 1.05), sr)
    elif kind == "cup-clink":
        for _ in range(int(rng.integers(1, 3))):
            f = rng.uniform(2500, 5000)
            place(out, modal_impact(rng, [f, f * 2.1], [0.05, 0.03], dur=0.2, sr=sr), int(rng.integers(0, n)), 0.5)
    elif kind == "knock":
        pos = int(rng.integers(0, max(1, n - n_samples(0.8, sr))))
        for _ in range(int(rng.integers(2, 5))):
            place(out, modal_impact(rng, rng.uniform(200, 900, 3), [0.03, 0.02, 0.01], dur=0.1, sr=sr), pos)
            pos += n_samples(rng.uniform(0.12, 0.2), sr)
    else:
        pos = n_samples(rng.uniform(0, 0.3), sr)
        while pos < n:
            cn = n_samples(0.02, sr)
            place(out, bandpass(white(rng, cn), 800, 5000, sr) * exp_env(cn, 0.004, sr), pos, rng.uniform(0.5, 1.0))
            pos += n_samples(rng.uniform(0.2, 0.35), sr)
    return out, kind


def background_noise(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    n = n_samples(dur, sr)
    y, kind = background_bed(rng, n, sr, tts=tts)
    variant = kind
    if rng.random() < 0.3:
        extra, k2 = background_bed(rng, n, sr, tts=tts)
        y = y + rng.uniform(0.2, 0.6) * extra
        variant += f"+{k2}"
    if rng.random() < 0.3:
        tr, tk = benign_transients(rng, n, sr)
        y = y + rng.uniform(1.0, 3.0) * tr
        variant += f"+{tk}"
    return peak_normalize(y), {"variant": variant}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.std(x) + 1e-12)


def empty_room(rng: np.random.Generator, n: int, sr: int = SR) -> tuple[np.ndarray, str]:
    """A room with nothing happening, as a microphone records it: low rumble with room modes, mains hum,
    steady appliance tones, electrical hiss and sometimes a faint high whine. The level hardly changes."""
    parts = []
    y = _unit(lowpass(colored_noise(rng, n, rng.uniform(1.3, 2.3)), rng.uniform(250, 1200), sr))
    for _ in range(int(rng.integers(1, 4))):  # room modes
        f0 = rng.uniform(28, 160)
        y += 10 ** (rng.uniform(-12, -3) / 20) * _unit(bandpass(white(rng, n), f0 * 0.94, f0 * 1.06, sr, 2))
    t = np.arange(n) / sr
    if rng.random() < 0.75:  # mains hum and its harmonics
        mains = float(rng.choice([50.0, 60.0]))
        hum = sum(rng.uniform(0.2, 1.0) / k ** rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * k * mains * t + rng.uniform(0, 6.28))
                  for k in range(1, 7))
        y += 10 ** (rng.uniform(-15, 6) / 20) * _unit(hum)
        parts.append(f"hum{mains:.0f}")
    if rng.random() < 0.4:  # fridge / fan / computer
        for _ in range(int(rng.integers(1, 3))):
            f = rng.uniform(90, 420)
            tone = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * (1 + 0.1 * smooth_random(rng, n, 0.5, sr))
            y += 10 ** (rng.uniform(-20, -6) / 20) * _unit(tone)
        parts.append("appliance")
    y += 10 ** (rng.uniform(-40, -18) / 20) * _unit(highpass(white(rng, n), rng.uniform(1500, 4000), sr))  # hiss
    if rng.random() < 0.25:
        y += 10 ** (rng.uniform(-40, -25) / 20) * np.sqrt(2) * np.sin(2 * np.pi * rng.uniform(1000, 5000) * t)
        parts.append("whine")
    y *= 1 + 0.03 * smooth_random(rng, n, 0.2, sr)  # slow drift, well under 1 dB
    if rng.random() < 0.35:  # band-limited recording (phone, compressed file)
        y = lowpass(y, rng.uniform(5000, 9000), sr)
        parts.append("bandlimited")
    return y, "+".join(["empty-room"] + parts)


def background_empty_room(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    y, variant = empty_room(rng, n_samples(dur, sr), sr)
    return peak_normalize(y), {"variant": variant}


def _click(rng, sr, lo=1500, hi=9000, decay=(0.003, 0.015), thump=0.0):
    """One small mechanical click: a very short broadband tick, a few ringing partials, maybe a soft thump."""
    m = n_samples(0.06, sr)
    t = np.arange(m) / sr
    k = int(rng.integers(3, 6))
    ring = modal_impact(rng, rng.uniform(lo, hi, k), rng.uniform(*decay, k), dur=0.06, sr=sr)
    tick = np.zeros(m)
    w = n_samples(rng.uniform(0.0005, 0.002), sr)
    tick[:w] = rng.standard_normal(w) * np.linspace(1, 0, w)
    out = ring / (np.max(np.abs(ring)) + 1e-9) + rng.uniform(0.5, 1.5) * tick
    if thump:
        out += thump * np.sin(2 * np.pi * rng.uniform(150, 400) * t) * np.exp(-t / 0.01)
    return out


def everyday_clicks(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    """Harmless clicks recorded close to the microphone - camera shutters, switches, keyboards, pens, staplers,
    mouse clicks - over quiet room tone. They are short and sharp like a gunshot but have no blast, boom or
    echo tail; the models must learn they are background."""
    n = n_samples(dur, sr)
    y = np.zeros(n)
    kind = _choice(rng, ["camera-shutter", "film-camera", "light-switch", "keyboard", "pen-click", "stapler",
                         "mouse-click"], [0.25, 0.15, 0.12, 0.16, 0.1, 0.1, 0.12])
    pos = n_samples(rng.uniform(0.1, 0.6), sr)
    if kind == "camera-shutter":  # mirror up + shutter / mirror down, single shots or a burst
        for _ in range(int(rng.choice([1, 1, 2, 3, 5]))):
            if pos >= n:
                break
            place(y, _click(rng, sr, 1500, 9000, thump=rng.uniform(0.1, 0.4)), pos)
            place(y, _click(rng, sr, 1500, 9000, thump=rng.uniform(0.1, 0.4)), pos + n_samples(rng.uniform(0.04, 0.15), sr),
                  rng.uniform(0.6, 1.0))
            pos += n_samples(rng.uniform(0.12, 0.9), sr)
    elif kind == "film-camera":  # shutter click, then the advance lever's ratchet
        place(y, _click(rng, sr, 1200, 7000, thump=0.3), pos)
        pos += n_samples(rng.uniform(0.3, 0.8), sr)
        for _ in range(int(rng.integers(6, 14))):
            place(y, _click(rng, sr, 2500, 9000, (0.002, 0.006)), pos, rng.uniform(0.2, 0.5))
            pos += n_samples(rng.uniform(0.02, 0.045), sr)
    else:
        spec = {"light-switch": (800, 4000, (0.2, 1.2), 2), "keyboard": (1000, 5000, (0.08, 0.3), 25),
                "pen-click": (2500, 8000, (0.15, 0.6), 4), "stapler": (400, 3000, (0.3, 1.0), 3),
                "mouse-click": (2000, 7000, (0.08, 0.5), 6)}[kind]
        lo, hi, gap, count = spec
        for _ in range(int(rng.integers(1, count + 1))):
            if pos >= n:
                break
            thump = rng.uniform(0.3, 0.8) if kind == "stapler" else 0.0
            place(y, _click(rng, sr, lo, hi, thump=thump), pos, rng.uniform(0.6, 1.0))
            pos += n_samples(rng.uniform(*gap), sr)
    bed, _ = empty_room(rng, n, sr)
    y = y / (np.max(np.abs(y)) + 1e-9) + 10 ** (rng.uniform(-40, -22) / 20) * bed / (np.max(np.abs(bed)) + 1e-9)
    return peak_normalize(y), {"variant": f"everyday-clicks/{kind}"}


def neutral_speech(rng: np.random.Generator, dur: float, sr: int = SR, tts=None, **_):
    """Ordinary calm speech (not a call for help, not shouting) in a quiet room."""
    n = n_samples(dur, sr)
    y = np.zeros(n)
    info: dict = {"variant": "neutral-speech"}
    pos = n_samples(rng.uniform(0.0, 0.3), sr)
    if tts is not None and tts.available:
        for i in range(int(rng.integers(1, 4))):
            clip, meta = tts.sample(rng, "neutral", sr)
            if pos + len(clip) > n and i > 0:
                break
            place(y, clip, pos, rng.uniform(0.7, 1.0))
            pos += len(clip) + n_samples(rng.uniform(0.2, 0.8), sr)
            info.setdefault("phrases", []).append(meta["text"])
            info.setdefault("speaker", meta["voice"])
            info.setdefault("source_group", meta["group"])
    else:  # no voices installed: calm formant-synthesised syllables
        while pos < n:
            m = n_samples(rng.uniform(0.12, 0.3), sr)
            f0 = np.full(m, rng.uniform(100, 220)) * np.linspace(1.05, 0.95, m)
            place(y, _voiced(rng, f0, sr, [_choice(rng, list("aeiou"))]) * adsr(m, 0.02, 0.05, 0.8, 0.05, sr), pos)
            pos += m + n_samples(rng.uniform(0.05, 0.4), sr)
        info["variant"] = "neutral-speech/formant"
    bed, _ = empty_room(rng, n, sr)
    y = y / (np.max(np.abs(y)) + 1e-9) + 10 ** (rng.uniform(-40, -18) / 20) * bed / (np.max(np.abs(bed)) + 1e-9)
    return peak_normalize(y), info


def large_animals(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    """Horse neigh, lion / tiger roar, wolf howl and elephant trumpet."""
    n = n_samples(dur, sr)
    out = np.zeros(n)
    kind = _choice(rng, ["horse-neigh", "big-cat-roar", "wolf-howl", "elephant-trumpet"], [0.35, 0.3, 0.2, 0.15])
    pos = n_samples(rng.uniform(0.05, 0.4), sr)
    if kind == "horse-neigh":  # high whinny with fast vibrato falling in pitch, then a snort
        m = n_samples(rng.uniform(0.9, 1.6), sr)
        t = np.arange(m) / sr
        f0 = np.linspace(rng.uniform(600, 950), rng.uniform(200, 320), m) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(8, 13) * t))
        call = _voiced(rng, f0, sr, ["i", "e", "a"], shift=rng.uniform(0.75, 0.9), tilt=-6, jitter=0.04, breath=0.3)
        place(out, call * adsr(m, 0.03, 0.1, 0.9, 0.25, sr), pos)
        s = n_samples(0.25, sr)
        snort = bandpass(white(rng, s), 300, 3000, sr) * exp_env(s, 0.06, sr)
        place(out, 0.6 * snort / (np.max(np.abs(snort)) + 1e-9), pos + m + n_samples(rng.uniform(0.05, 0.3), sr))
    elif kind == "big-cat-roar":  # low, rough roar, then a series of grunts
        m = n_samples(rng.uniform(1.0, 2.2), sr)
        f0 = rng.uniform(70, 160) * (1 + 0.5 * np.sin(np.linspace(0.2, np.pi - 0.2, m)))
        roar = _voiced(rng, f0, sr, ["a", "o", "uh"], shift=rng.uniform(0.55, 0.75), tilt=-5, jitter=0.12, breath=0.4)
        place(out, roar * adsr(m, 0.08, 0.1, 0.9, 0.35, sr), pos)
        pos += m + n_samples(0.15, sr)
        for _ in range(int(rng.integers(0, 6))):
            g = n_samples(rng.uniform(0.18, 0.3), sr)
            grunt = _voiced(rng, np.full(g, rng.uniform(60, 110)), sr, ["uh"], shift=0.6, tilt=-6, jitter=0.1, breath=0.3)
            place(out, 0.7 * grunt * adsr(g, 0.02, 0.05, 0.8, 0.08, sr), pos)
            pos += g + n_samples(rng.uniform(0.15, 0.35), sr)
    elif kind == "wolf-howl":
        m = n_samples(rng.uniform(1.5, 3.0), sr)
        base = rng.uniform(300, 520)
        f0 = base * (1 + 0.35 * np.sin(np.linspace(0, np.pi, m)) ** 2) * (1 + 0.01 * smooth_random(rng, m, 5, sr))
        howl = _voiced(rng, f0, sr, ["u", "o"], shift=1.0, tilt=-12, jitter=0.01, breath=0.1)
        place(out, howl * adsr(m, 0.2, 0.2, 0.9, 0.5, sr), pos)
    else:  # elephant trumpet: bright, brassy, rough onset
        m = n_samples(rng.uniform(0.8, 1.8), sr)
        f0 = rng.uniform(380, 700) * np.linspace(1.0, rng.uniform(0.8, 1.15), m)
        trumpet = harmonic_tone(f0, sr, n_harm=18, tilt_db=-3, rng=rng) * (1 + 0.3 * white(rng, m) * exp_env(m, 0.1, sr))
        place(out, trumpet * adsr(m, 0.05, 0.1, 0.9, 0.3, sr), pos)
    return peak_normalize(out), {"variant": f"large-animal/{kind}"}


GENERATORS = {
    "Machinery Fault": machinery_fault,
    "Glass Breaking": glass_breaking,
    "Alarm or Siren": alarm_or_siren,
    "Vehicle Horn": vehicle_horn,
    "Animal Sound": animal_sound,
    "Gunshot": gunshot,
    "Panic Scream": panic_scream,
    "Aggression": aggression,
    "Person Asking for Help": person_asking_for_help,
    "Background Noise": background_noise,
}

# Generators that are only used when asked for (``generate_synthetic_dataset --variant``), so the default
# generation above stays reproducible for the existing seeds.
VARIANT_GENERATORS = {
    "empty-room": ("Background Noise", background_empty_room),
    "everyday-clicks": ("Background Noise", everyday_clicks),
    # Calm speech variant mapped to Background Noise
    "neutral-speech": ("Background Noise", neutral_speech),
    # Large wild animals mapped to Animal Sound class
    "large-animals": ("Animal Sound", large_animals),
}
