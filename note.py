from datetime import datetime
from typing import Dict

import math
import random
import numpy as np

from schemas import Note, Curve

c0_frequency = 16.3516


def apply_attack(amp, i, attack_duration, attack_curve, sample_rate):
    if attack_duration > i / sample_rate:
        if attack_curve == Curve.linear:
            amp *= (i / (attack_duration * sample_rate))
        elif attack_curve == Curve.logarithmic:
            amp *= (1 - (1 - (i / (attack_duration * sample_rate))) ** 2) ** (1 / 2)
        elif attack_curve == Curve.exponential:
            amp *= 1 - (1 - (i / (attack_duration * sample_rate)) ** 2) ** (1 / 2)
        else:
            raise ValueError
    return amp


def apply_decay(amp, i, decay_duration, decay_curve, sample_rate):
    if decay_curve == Curve.linear:
        amp *= 1 - (i / (decay_duration * sample_rate))
    elif decay_curve == Curve.logarithmic:
        amp *= (1 - (i / (decay_duration * sample_rate)) ** 2) ** (1 / 2)
    elif decay_curve == Curve.exponential:
        amp *= 1 - (1 - (1 - (i / (decay_duration * sample_rate))) ** 2) ** (1 / 2)
    elif decay_curve == Curve.constant:
        pass
    else:
        raise ValueError
    return amp


def apply_release(amp, i, duration, release_duration, release_curve, sample_rate):
    if duration < i / sample_rate:
        if release_curve == Curve.linear:
            amp *= 1 - ((i - duration * sample_rate) / (release_duration * sample_rate))
        elif release_curve == Curve.logarithmic:
            amp *= (1 - ((i - duration * sample_rate) / (release_duration * sample_rate)) ** 2) ** (1 / 2)
        elif release_curve == Curve.exponential:
            amp *= 1 - (1 - (1 - ((i - duration * sample_rate) / (release_duration * sample_rate))) ** 2) ** (1 / 2)
        else:
            raise ValueError
    return amp


def generate_note(
        note: Note = Note.C,
        octave: int = 4,
        duration: float = 1.0,
        attack_duration: float = 0.05,
        attack_curve: Curve = Curve.linear,
        decay_duration: float = 10.0,
        decay_curve: Curve = Curve.constant,
        release_duration: float = 0.5,
        release_curve: Curve = Curve.linear,
        amplitude: float = 1,
        formants: Dict[int, float] = None,
        noise: float = 0.0001,
        noise_sample_rate: int = 40,
        detune: float = 0.05,
        gain: float = 0,
        sample_rate: int = 44100
):
    start_time = datetime.now()
    print(note.name, octave, duration)
    signal = None
    random_detune = random.uniform(-detune, detune)
    for overtone in formants:
        wave = []
        for i in range(int(sample_rate * (duration + release_duration))):
            amp = formants[overtone] * amplitude
            amp = apply_attack(amp, i, attack_duration, attack_curve, sample_rate)
            amp = apply_decay(amp, i, decay_duration, decay_curve, sample_rate)
            amp = apply_release(amp, i, duration, release_duration, release_curve, sample_rate)

            random_noise = 0
            if i % noise_sample_rate == random.randrange(0, noise_sample_rate):
                random_noise = random.uniform(-noise, noise)

            frequency = c0_frequency * 2 ** (octave + (note.value + random_detune) / 12)
            wave.append(math.sin(math.pi * 2 * overtone * frequency * (i / sample_rate) + random_noise * amp) * amp)

        if signal is not None:
            signal += np.array(wave)
        else:
            signal = np.array(wave)

    cut_limit = 1/2**gain
    cut_signal = np.clip(signal, -cut_limit, cut_limit) / cut_limit

    print(datetime.now()-start_time)

    return cut_signal
