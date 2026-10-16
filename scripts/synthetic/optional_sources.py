"""Procedural generators for the SRS optional categories.

Drone Sound, Drilling or Grinder Sound, Fireworks, Vehicle Backfire, Explosion, Door Impact and Graffiti
Spray. Same contract as ``sources.py``: each generator returns the dry event (recorded close to the source,
peak-normalised) and ``{"variant": ...}``; ``scripts/generate_synthetic_dataset.py`` adds room, distance,
background and recording device.

Fireworks, Vehicle Backfire, Explosion and Door Impact are impulsive like Gunshot. What tells them apart is
modelled explicitly: fireworks have launch whistles and crackling tails, backfires are duller pops from an
exhaust pipe with engine sound around them, explosions have a slow heavy boom and a long rumble, door
impacts are wooden or metal panel thuds with latch rattle.
"""
from __future__ import annotations

import numpy as np

from .dsp import (
    SR,
    bandpass,
    colored_noise,
    harmonic_tone,
    highpass,
    lowpass,
    modal_impact,
    n_samples,
    peak_normalize,
    phase_from_freq,
    place,
    resonator,
    smooth_random,
    thud,
    white,
)


def _choice(rng: np.random.Generator, options, p=None):
    return options[int(rng.choice(len(options), p=p))]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.std(x) + 1e-12)


def _db(db: float) -> float:
    return 10 ** (db / 20)


def _env_ar(n: int, attack: float, release: float, sr: int) -> np.ndarray:
    """Attack / release envelope over the whole length."""
    env = np.ones(n)
    a, r = min(n, n_samples(attack, sr)), min(n, n_samples(release, sr))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


# =========================================================================== #
# Drone Sound (multirotor UAV)
# =========================================================================== #
def drone_sound(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    size = _choice(rng, ["quad", "hexa", "octo", "mini"], [0.55, 0.2, 0.1, 0.15])
    rotors = {"quad": 4, "hexa": 6, "octo": 8, "mini": 4}[size]
    base = rng.uniform(300, 520) if size == "mini" else rng.uniform(110, 300)  # blade-pass frequency
    flight = _choice(rng, ["hover", "pass-by", "approach", "recede", "manoeuvre"], [0.3, 0.25, 0.15, 0.1, 0.2])
    # Throttle: every rotor follows the common manoeuvre plus its own small corrections.
    common = 1.0 + (0.12 if flight == "manoeuvre" else 0.03) * smooth_random(rng, n, rng.uniform(0.3, 1.2), sr)
    doppler = np.ones(n)
    gain = np.ones(n)
    if flight == "pass-by":
        t0, v, d = rng.uniform(0.3, 0.7) * dur, rng.uniform(5, 18), rng.uniform(3, 15)
        x = v * (t - t0)
        doppler = 1.0 / (1.0 + (v * x / np.sqrt(x ** 2 + d ** 2)) / 343.0)
        gain = d / np.sqrt(x ** 2 + d ** 2)
    elif flight in ("approach", "recede"):
        ramp = np.linspace(0.25, 1.0, n)
        gain = ramp if flight == "approach" else ramp[::-1]
    y = np.zeros(n)
    tilt = rng.uniform(-8, -4)
    for _ in range(rotors):
        f = base * rng.uniform(0.96, 1.04) * common * (1 + 0.01 * smooth_random(rng, n, rng.uniform(2, 6), sr)) * doppler
        y += harmonic_tone(f, sr, n_harm=24, tilt_db=tilt, rng=rng) * rng.uniform(0.7, 1.0)
    # Aerodynamic noise modulated at the blade-pass rate, and the electric motors' whine.
    aero = bandpass(white(rng, n), 400, 6000, sr, 2) * (1 + 0.6 * np.sin(phase_from_freq(base * common * doppler, sr)))
    y = _unit(y) + _db(rng.uniform(-14, -6)) * _unit(aero)
    y += _db(rng.uniform(-26, -14)) * np.sin(phase_from_freq(base * rng.uniform(6, 14) * common * doppler, sr))
    y *= gain * _env_ar(n, rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.6), sr)
    return peak_normalize(y), {"variant": f"{size}/{flight}"}


# =========================================================================== #
# Drilling or Grinder Sound
# =========================================================================== #
def drilling_or_grinder(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    tool = _choice(rng, ["drill", "hammer-drill", "angle-grinder", "bench-grinder", "impact-driver"],
                   [0.25, 0.2, 0.25, 0.15, 0.15])
    y = np.zeros(n)
    pos = 0
    bursts = []
    while pos < n:  # trigger pulls: on for 0.6-3 s, off for 0.1-0.8 s
        on = n_samples(rng.uniform(0.6, 3.0), sr)
        bursts.append((pos, min(n, pos + on)))
        pos += on + n_samples(rng.uniform(0.1, 0.8), sr)
    for a, b in bursts:
        m = b - a
        tt = np.arange(m) / sr
        spin = 1 - np.exp(-tt / rng.uniform(0.05, 0.25))  # motor spin-up
        load = np.clip(smooth_random(rng, m, rng.uniform(0.5, 2.0), sr), 0, 1)  # bit / disc biting the material
        if tool in ("angle-grinder", "bench-grinder"):
            f_m = rng.uniform(140, 200) if tool == "angle-grinder" else rng.uniform(45, 60)
            f = f_m * (0.2 + 0.8 * spin) * (1 - 0.12 * load)
            motor = harmonic_tone(f, sr, n_harm=40, tilt_db=rng.uniform(-5, -2), rng=rng)
            disc = bandpass(white(rng, m), 2000, 9500, sr, 2) * (0.2 + load)
            sparks = np.zeros(m)
            idx = rng.integers(0, m, int(m / sr * rng.uniform(200, 900)))
            sparks[idx] = rng.uniform(0.3, 1.0, idx.size)
            sparks = highpass(sparks, 3000, sr) * load
            seg = _unit(motor) + _db(rng.uniform(-6, 2)) * _unit(disc) + _db(rng.uniform(-12, -4)) * _unit(sparks)
        else:
            f_m = rng.uniform(150, 450)
            f = f_m * (0.3 + 0.7 * spin) * (1 - 0.18 * load)
            motor = harmonic_tone(f, sr, n_harm=30, tilt_db=rng.uniform(-7, -4), rng=rng)
            chatter = bandpass(white(rng, m), 800, 6000, sr, 2) * (0.1 + load)
            seg = _unit(motor) + _db(rng.uniform(-10, -2)) * _unit(chatter)
            if tool in ("hammer-drill", "impact-driver"):
                rate = rng.uniform(25, 55) if tool == "hammer-drill" else rng.uniform(40, 70)
                hits = np.zeros(m)
                step = sr / rate
                for k in np.arange(0, m, step):
                    if load[int(k)] > 0.2 or tool == "hammer-drill":
                        hits[int(k)] = rng.uniform(0.6, 1.0)
                impact = np.convolve(hits, modal_impact(rng, rng.uniform(600, 3000, 3), [0.004, 0.003, 0.002], dur=0.015, sr=sr))[:m]
                seg = seg + _db(rng.uniform(-6, 2)) * _unit(impact + 1e-9 * white(rng, m))
        y[a:b] += seg * _env_ar(m, 0.02, 0.08, sr)
    return peak_normalize(y), {"variant": tool}


# =========================================================================== #
# Fireworks
# =========================================================================== #
def _crackle(rng, length, sr, density, decay=True):
    m = n_samples(length, sr)
    out = np.zeros(m)
    idx = rng.integers(0, m, int(length * density))
    out[idx] = rng.uniform(0.2, 1.0, idx.size) * rng.choice([-1, 1], idx.size)
    out = bandpass(out, 1500, 9000, sr, 2)
    return out * np.exp(-np.arange(m) / sr / rng.uniform(0.4, 1.5)) if decay else out


def _burst(rng, sr, size):
    m = n_samples(rng.uniform(0.6, 1.4), sr)
    t = np.arange(m) / sr
    boom_f = rng.uniform(45, 110) * (1 + 0.8 * np.exp(-t / 0.02))
    boom = np.sin(phase_from_freq(boom_f, sr)) * np.exp(-t / rng.uniform(0.12, 0.35))
    crack = lowpass(white(rng, m), rng.uniform(3000, 7000), sr, 2) * np.exp(-t / rng.uniform(0.008, 0.02))
    return size * (_unit(boom) * 0.8 + _unit(crack) * rng.uniform(0.6, 1.2))


def fireworks(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    kind = _choice(rng, ["shells", "rocket", "firecracker-string", "fountain"], [0.4, 0.2, 0.25, 0.15])
    y = np.zeros(n)
    if kind == "firecracker-string":  # rapid irregular pops
        pos = n_samples(rng.uniform(0.0, 0.3), sr)
        while pos < n:
            pop = _burst(rng, sr, rng.uniform(0.3, 1.0))[: n_samples(0.12, sr)]
            place(y, pop * np.exp(-np.arange(len(pop)) / sr / 0.02), pos)
            pos += n_samples(rng.uniform(0.02, 0.12), sr)
    elif kind == "fountain":  # continuous hiss and crackle
        hiss = highpass(white(rng, n), 2000, sr) * (0.6 + 0.4 * smooth_random(rng, n, 3.0, sr))
        y = (_unit(hiss) + 0.8 * _unit(_crackle(rng, dur, sr, 400, decay=False))[:n]) * _env_ar(n, 0.2, 0.4, sr)
    else:
        pos = n_samples(rng.uniform(0.0, 0.4), sr)
        for _ in range(int(rng.integers(1, 4))):
            if pos >= n:
                break
            rise = rng.uniform(0.4, 1.2)
            m = n_samples(rise, sr)
            t = np.arange(m) / sr
            if kind == "rocket" or rng.random() < 0.5:  # launch whistle or whoosh
                if rng.random() < 0.5:
                    launch = np.sin(phase_from_freq(np.linspace(rng.uniform(600, 1200), rng.uniform(2000, 3500), m), sr))
                else:
                    launch = bandpass(white(rng, m), 800, 5000, sr, 2)
                place(y, _unit(launch) * np.linspace(0.2, 0.6, m) * 0.3, pos)
                pos += m
            b = _burst(rng, sr, rng.uniform(0.7, 1.0))
            place(y, b, pos)
            if rng.random() < 0.7:
                place(y, 0.5 * _unit(_crackle(rng, rng.uniform(0.6, 2.0), sr, rng.uniform(150, 600))), pos + n_samples(0.05, sr))
            pos += len(b) + n_samples(rng.uniform(0.1, 0.9), sr)
    return peak_normalize(y[:n]), {"variant": kind}


# =========================================================================== #
# Vehicle Backfire
# =========================================================================== #
def _engine(rng, n, sr, rev=False):
    f = rng.uniform(25, 60) * np.ones(n)
    if rev:
        f *= 1 + 1.2 * np.clip(np.sin(np.linspace(0, np.pi, n)), 0, 1)
    f *= 1 + 0.02 * smooth_random(rng, n, 4.0, sr)
    tone = harmonic_tone(f, sr, n_harm=16, tilt_db=-5, rng=rng)
    return _unit(tone) + 0.3 * _unit(lowpass(colored_noise(rng, n, 1.0), 700, sr))


def _pop(rng, sr, loud=1.0):
    m = n_samples(0.35, sr)
    t = np.arange(m) / sr
    body = lowpass(white(rng, m), rng.uniform(900, 2500), sr, 2) * np.exp(-t / rng.uniform(0.006, 0.02))
    pipe = resonator(white(rng, m) * np.exp(-t / 0.004), rng.uniform(80, 300), rng.uniform(20, 60), sr)
    return loud * (_unit(body) + rng.uniform(0.6, 1.4) * _unit(pipe) * np.exp(-t / rng.uniform(0.03, 0.08)))


def vehicle_backfire(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    kind = _choice(rng, ["single-pop", "double-pop", "decel-crackle", "rev-bang"], [0.35, 0.25, 0.2, 0.2])
    y = _db(rng.uniform(-16, -6)) * _engine(rng, n, sr, rev=kind == "rev-bang") if rng.random() < 0.85 else np.zeros(n)
    start = n_samples(rng.uniform(0.2, 0.5) * dur, sr)
    if kind == "single-pop":
        place(y, _pop(rng, sr), start)
    elif kind == "double-pop":
        place(y, _pop(rng, sr), start)
        place(y, _pop(rng, sr, rng.uniform(0.5, 1.0)), start + n_samples(rng.uniform(0.08, 0.5), sr))
    elif kind == "decel-crackle":  # burbles on overrun
        pos = start
        for _ in range(int(rng.integers(4, 12))):
            place(y, _pop(rng, sr, rng.uniform(0.3, 0.9)), pos)
            pos += n_samples(rng.uniform(0.04, 0.2), sr)
    else:
        place(y, _pop(rng, sr), n_samples(0.6 * dur, sr))
    return peak_normalize(y[:n]), {"variant": kind}


# =========================================================================== #
# Explosion
# =========================================================================== #
def explosion(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    kind = _choice(rng, ["blast-near", "blast-distant", "blast-debris", "gas-whoomp"], [0.3, 0.3, 0.25, 0.15])
    start = n_samples(rng.uniform(0.05, 0.5), sr)
    m = n - start
    t = np.arange(m) / sr
    attack = rng.uniform(0.02, 0.06) if kind == "gas-whoomp" else rng.uniform(0.0005, 0.004)
    rise = np.minimum(1.0, t / attack)
    f = rng.uniform(22, 38) + rng.uniform(40, 90) * np.exp(-t / rng.uniform(0.08, 0.25))
    boom = np.sin(phase_from_freq(f, sr)) * np.exp(-t / rng.uniform(0.4, 1.2))
    rumble = lowpass(colored_noise(rng, m, 2.0), rng.uniform(150, 500), sr) * np.exp(-t / rng.uniform(0.8, 2.5))
    blast = lowpass(white(rng, m), rng.uniform(1500, 6000), sr, 2) * np.exp(-t / rng.uniform(0.03, 0.1))
    y = rise * (_unit(boom) * 1.0 + _unit(rumble) * rng.uniform(0.6, 1.2) + _unit(blast) * rng.uniform(0.4, 1.0))
    if kind == "blast-distant":
        y = lowpass(y, rng.uniform(300, 900), sr)
    if kind == "blast-debris":
        debris = np.zeros(m)
        for _ in range(int(rng.integers(20, 80))):
            p = n_samples(rng.uniform(0.2, max(0.3, m / sr - 0.2)), sr)
            place(debris, modal_impact(rng, rng.uniform(300, 3000, 2), [0.01, 0.006], dur=0.04, sr=sr), p,
                  rng.uniform(0.05, 0.3))
        y += _unit(debris) * 0.3 * np.exp(-t / 2.0)
    out = np.zeros(n)
    out[start:] = y
    return peak_normalize(out), {"variant": kind}


# =========================================================================== #
# Door Impact
# =========================================================================== #
def _door_hit(rng, sr, material, force=1.0):
    if material == "metal":
        ring = modal_impact(rng, rng.uniform(180, 1800, 5), rng.uniform(0.08, 0.5, 5), dur=0.8, sr=sr)
        return force * (_unit(thud(rng, rng.uniform(60, 120), 0.05, 0.8, sr)) + 0.8 * _unit(ring))
    body = thud(rng, rng.uniform(55, 140), rng.uniform(0.04, 0.12), 0.5, sr)
    panel = modal_impact(rng, rng.uniform(140, 650, 4), rng.uniform(0.03, 0.15, 4), dur=0.5, sr=sr)
    latch = modal_impact(rng, rng.uniform(1500, 4200, 3), [0.01, 0.008, 0.006], dur=0.5, sr=sr)
    return force * (_unit(body) + 0.7 * _unit(panel) + rng.uniform(0.1, 0.4) * _unit(latch))


def door_impact(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    kind = _choice(rng, ["slam", "metal-slam", "banging", "kicking", "knocking"], [0.3, 0.15, 0.2, 0.15, 0.2])
    y = np.zeros(n)
    pos = n_samples(rng.uniform(0.05, 0.4), sr)
    if kind in ("slam", "metal-slam"):
        place(y, _door_hit(rng, sr, "metal" if kind == "metal-slam" else "wood"), pos)
        if rng.random() < 0.5:  # frame rattle right after
            for k in range(int(rng.integers(2, 6))):
                place(y, modal_impact(rng, rng.uniform(800, 3000, 2), [0.01, 0.008], dur=0.03, sr=sr),
                      pos + n_samples(0.05 + 0.03 * k, sr), rng.uniform(0.05, 0.2))
    elif kind in ("banging", "kicking"):
        for _ in range(int(rng.integers(2, 7))):
            if pos >= n:
                break
            place(y, _door_hit(rng, sr, "wood", rng.uniform(0.6, 1.0) if kind == "banging" else 1.0), pos)
            pos += n_samples(rng.uniform(0.25, 0.9) if kind == "banging" else rng.uniform(0.6, 1.5), sr)
    else:
        for _ in range(int(rng.integers(1, 3))):
            for _ in range(int(rng.integers(2, 6))):
                if pos >= n:
                    break
                knock = modal_impact(rng, rng.uniform(250, 900, 3), [0.02, 0.015, 0.01], dur=0.08, sr=sr)
                place(y, knock, pos, rng.uniform(0.6, 1.0))
                pos += n_samples(rng.uniform(0.1, 0.25), sr)
            pos += n_samples(rng.uniform(0.5, 1.2), sr)
    return peak_normalize(y[:n]), {"variant": kind}


# =========================================================================== #
# Graffiti Spray
# =========================================================================== #
def graffiti_spray(rng: np.random.Generator, dur: float, sr: int = SR, **_):
    n = n_samples(dur, sr)
    kind = _choice(rng, ["shake+spray", "spray-bursts", "long-spray", "sputter"], [0.35, 0.3, 0.2, 0.15])
    y = np.zeros(n)
    pos = n_samples(rng.uniform(0.0, 0.3), sr)
    if kind == "shake+spray":  # mixing ball rattling in the can
        shake_end = pos + n_samples(rng.uniform(0.6, 1.6), sr)
        rate = rng.uniform(5, 10)
        body = rng.uniform(1800, 4500)
        while pos < min(n, shake_end):
            for k in range(2):  # ball hits both ends of the can
                click = modal_impact(rng, [body * rng.uniform(0.95, 1.05), body * 2.3, body * 3.1], [0.012, 0.008, 0.005],
                                     dur=0.04, sr=sr)
                place(y, click, pos + n_samples(k / (2 * rate), sr), rng.uniform(0.5, 1.0))
            pos += n_samples(1 / rate, sr)
        pos += n_samples(rng.uniform(0.1, 0.4), sr)
    nozzle = rng.uniform(3000, 7000)
    while pos < n:
        length = rng.uniform(1.2, 3.0) if kind == "long-spray" else rng.uniform(0.3, 1.2)
        m = min(n - pos, n_samples(length, sr))
        if m <= 0:
            break
        hiss = highpass(white(rng, m), rng.uniform(1200, 2500), sr)
        hiss = hiss + 0.6 * bandpass(white(rng, m), nozzle * 0.85, nozzle * 1.15, sr, 2)
        env = _env_ar(m, 0.01, 0.03, sr) * (1 + 0.1 * smooth_random(rng, m, 8, sr))
        if kind == "sputter":
            env *= (smooth_random(rng, m, rng.uniform(8, 20), sr) > rng.uniform(-0.3, 0.3)).astype(float) * 0.8 + 0.2
        place(y, _unit(hiss) * env, pos)
        pos += m + n_samples(rng.uniform(0.1, 0.6), sr)
    return peak_normalize(y[:n]), {"variant": kind}


OPTIONAL_GENERATORS = {
    "Drone Sound": drone_sound,
    "Drilling or Grinder Sound": drilling_or_grinder,
    "Fireworks": fireworks,
    "Vehicle Backfire": vehicle_backfire,
    "Explosion": explosion,
    "Door Impact": door_impact,
    "Graffiti Spray": graffiti_spray,
}
