import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.io.wavfile import write
import random

from note import generate_note
from schemas import Note, Curve


def generate_formants(
        base_divisor: float = 2,
        overtone_decay: Curve = Curve.exponential,
        overtone_random: float = 0.25,
        n_overtones: int = 16,
) -> Dict:
    formants = {}
    for i in range(n_overtones):
        if overtone_decay == Curve.constant:
            formants[i] = 1 / base_divisor
        elif overtone_decay == Curve.logarithmic:
            formants[i] = 1 / (base_divisor + math.log(i + 1))
        elif overtone_decay == Curve.linear:
            formants[i] = 1 / (base_divisor + i)
        elif overtone_decay == Curve.quadratic:
            formants[i] = 1 / (base_divisor * (i + 1))
        elif overtone_decay == Curve.exponential:
            formants[i] = 1 / (base_divisor ** (i + 1))
        else:
            raise ValueError
        formants[i] += random.uniform(-formants[i] * overtone_random / 2, formants[i] * overtone_random / 2)
    print(formants)
    return formants


def generate_harmony(chords: List[np.ndarray], start_times: List[float], sample_rate: int):
    length = 0
    for i in range(len(start_times)):
        start_time = start_times[i]
        chord = chords[i]
        end_point = int(start_time * sample_rate) + chord.shape[0]
        if end_point > length:
            length = end_point

    harmony = np.zeros(length)
    for i in range(len(start_times)):
        start_time = start_times[i]
        chord = chords[i]
        step = int(start_time * sample_rate)
        harmony[step:step + chord.shape[0]] += chord
    return harmony


def generate_notes(note_strings: List[str], **kwargs):
    cache = {}
    notes = []
    for note_string in note_strings:
        if note_string not in cache:
            note_list = note_string.split('_')
            note_value = note_list[0][:-1]
            octave = int(note_list[0][-1])
            duration = int(note_list[1])

            cache[note_string] = generate_note(
                note=eval(f'Note.{note_value}'),
                octave=octave,
                duration=duration,
                **kwargs
            )
        notes.append(cache[note_string])
    return notes


def main():
    sample_rate = 4410

    chord_formants = generate_formants(
        base_divisor=4,
        overtone_decay=Curve.exponential,
        n_overtones=4,
        overtone_random=0.5
    )

    melody_formants = generate_formants(
        base_divisor=3,
        overtone_decay=Curve.exponential,
        n_overtones=9,
        overtone_random=0.5
    )

    chord_attributes = {
        'attack_duration': 0.035,
        'attack_curve': Curve.exponential,
        'decay_duration': 50,
        'decay_curve': Curve.exponential,
        'release_duration': 0.1,
        'formants': chord_formants,
        'noise': 0.1,
        'noise_sample_rate': sample_rate // 100,
        'gain': 0,
        'sample_rate': sample_rate
    }

    melody_attributes = {
        'attack_duration': 0.03,
        'attack_curve': Curve.linear,
        'decay_duration': 50,
        'decay_curve': Curve.exponential,
        'release_duration': 0.25,
        'release_curve': Curve.exponential,
        'formants': melody_formants,
        'noise': 0.05,
        'noise_sample_rate': sample_rate // 40,
        'gain': 0,
        'sample_rate': sample_rate
    }

    harmony = generate_harmony(
        chords=generate_notes(
            ['C3_16', 'G3_8', 'E4_8', 'A3_4', 'F4_4', 'G3_4', 'E4_4',
             'F2_8', 'C3_8', 'A3_8', 'F4_8', 'G2_8', 'D3_8', 'B3_8', 'G4_8'],
            **chord_attributes
        ),
        start_times=[0, 0, 0, 8, 8, 12, 12, 16, 16, 16, 16, 24, 24, 24],
        sample_rate=sample_rate
    )

    harmony = generate_harmony(
        chords=[harmony] + generate_notes(['G4_8', 'A4_4', 'G4_6', 'F4_2', 'E4_2', 'D4_10'], **melody_attributes),
        start_times=[0, 0, 8, 12, 18, 20, 22],
        sample_rate=sample_rate
    )

    harmony = generate_harmony(
        chords=[harmony] + generate_notes(
            ['E4_2', 'D4_2', 'C4_1', 'D4_1', 'E4_2', 'C4_1', 'D4_1', 'E4_1', 'F4_1', 'E4_2',
             'D4_1', 'E4_1', 'D4_1', 'E4_1', 'D4_1', 'C4_2', 'D4_1', 'C4_1', 'B3_9'],
            **melody_attributes
        ),
        start_times=[0, 0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23],
        sample_rate=sample_rate
    )

    print(max(harmony), min(harmony), harmony.shape)

    write('data/note_test.wav', sample_rate, harmony)


if __name__ == "__main__":
    main()
