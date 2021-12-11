from typing import Dict

from scipy.io.wavfile import write
import math
import random
import numpy as np
import sys

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
        formants: Dict[int, float] = None,
        noise: float = 0.0001,
        sample_rate: int = 44100
):
    signal = []
    for i in range(int(sample_rate * (duration + release_duration))):
        value = 0
        for overtone in formants:
            amp = formants[overtone]
            amp = apply_attack(amp, i, attack_duration, attack_curve, sample_rate)
            amp = apply_decay(amp, i, decay_duration, decay_curve, sample_rate)
            amp = apply_release(amp, i, duration, release_duration, release_curve, sample_rate)

            frequency = c0_frequency * 2 ** (octave + note.value / 12)
            value += math.sin(math.pi * 2 * overtone * frequency * (i / sample_rate)) * amp
        signal.append(value)
    return signal


result = generate_note(
    note=Note.E,
    duration=1,
    attack_duration=0.001,
    decay_duration=100,
    decay_curve=Curve.exponential,
    release_duration=0.1,
    formants={1: 1/2, 2: 1/4, 3: 1/8, 4: 1/16}
)

print(max(result), min(result))

write('note_test.wav', 44100, np.array(result))
