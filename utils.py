from typing import List

import numpy as np

from note import generate_note
from schemas import Note


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

            print(note_string, note_list, note_value, octave)

            duration = float(note_list[1])

            cache[note_string] = generate_note(
                note=getattr(Note, note_value),
                octave=octave,
                duration=duration,
                **kwargs
            )
        notes.append(cache[note_string])
    return notes
