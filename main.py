from scipy.io.wavfile import write

from formant import generate_formants
from schemas import Curve
from utils import generate_harmony, generate_notes


def shen_khar_venakhi():
    sample_rate = 44100

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

    last_chord_attributes = {
        'attack_duration': 0.035,
        'attack_curve': Curve.exponential,
        'decay_duration': 8.1,
        'decay_curve': Curve.linear,
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
             'F2_8', 'C3_8', 'A3_8', 'F4_8',
             'C2_16', 'C4_16', 'G2_8', 'E3_8', 'A2_4', 'F3_4', 'G2_4', 'E3_4',
             'F2_8', 'C3_8', 'A3_8',
             'C2_8', 'C4_8', 'G2_8', 'E3_8', 'G2_4', 'D3_4', 'B3_4', 'F2_4', 'C3_4', 'A3_4',
             'C2_8', 'C4_8', 'G2_4', 'E3_4', 'B2_4', 'G3_4',
             'C3_16', 'G3_8', 'E4_8', 'A3_4', 'F4_4', 'G3_4', 'E4_4',
             'F2_8', 'C3_8', 'A3_8', 'F4_8'],
            **chord_attributes
        ),
        start_times=[0, 0, 0, 8, 8, 12, 12,
                     16, 16, 16, 16,
                     32, 32, 32, 32, 40, 40, 44, 44,
                     48, 48, 48,
                     56, 56, 56, 56, 64, 64, 64, 68, 68, 68,
                     72, 72, 72, 72, 76, 76,
                     80, 80, 80, 88, 88, 92, 92,
                     96, 96, 96, 96],
        sample_rate=sample_rate
    )

    harmony = generate_harmony(
        chords=[harmony] + generate_notes(
            ['G2_8', 'D3_8', 'B3_8', 'G4_8', 'G2_8', 'D3_8', 'B3_8', 'G4_8'],
            **last_chord_attributes
        ),
        start_times=[0, 24, 24, 24, 24, 104, 104, 104, 104],
        sample_rate=sample_rate
    )

    harmony = generate_harmony(
        chords=[harmony] + generate_notes(
            ['G4_8', 'A4_4', 'G4_6', 'F4_2', 'E4_2', 'D4_6',
             'G4_8', 'A4_4', 'G4_6', 'F4_2', 'E4_2', 'D4_6'],
            **melody_attributes
        ),
        start_times=[0, 0, 8, 12, 18, 20, 22, 80, 88, 92, 98, 100, 102],
        sample_rate=sample_rate
    )

    harmony = generate_harmony(
        chords=[harmony] + generate_notes(
            ['E4_2', 'D4_2', 'C4_1', 'D4_1', 'E4_2', 'C4_1', 'D4_1', 'E4_1', 'F4_1', 'E4_2',
             'D4_1', 'E4_1', 'D4_1', 'E4_1', 'D4_1', 'C4_2', 'D4_1', 'C4_1', 'B3_5'],
            **melody_attributes
        ),
        start_times=[0, 80, 82, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103],
        sample_rate=sample_rate
    )

    print(max(harmony), min(harmony), harmony.shape)

    write('data/shen_khar_venakhi.wav', sample_rate, harmony)


def djent():
    sample_rate = 44100

    formants = generate_formants(
        base_divisor=16,
        overtone_decay=Curve.quadratic,
        n_overtones=16,
        overtone_random=4
    )

    attributes = {
        'attack_duration': 0.005,
        'attack_curve': Curve.linear,
        'decay_duration': 2,
        'decay_curve': Curve.exponential,
        'release_duration': 0.25,
        'release_curve': Curve.linear,
        'formants': formants,
        'noise': 5,
        'noise_sample_rate': sample_rate // 4000,
        'gain': 3,
        'sample_rate': sample_rate
    }

    harmony = generate_harmony(
        chords=generate_notes(
            ['B1_0.0625', 'B1_0.0625', 'B1_0.0625',
             'E2_0.0625', 'E2_0.0625', 'E2_0.0625',
             'B2_0.0625', 'B2_0.0625', 'B2_0.0625',
             'E3_0.0625', 'E3_0.0625', 'E3_0.0625'],
            **attributes
        ),
        start_times=[0, 0.25, 0.5, 0, 0.25, 0.5, 0, 0.25, 0.5, 0, 0.25, 0.5],
        sample_rate=sample_rate
    )

    print(max(harmony), min(harmony), harmony.shape)

    write('data/djent.wav', sample_rate, harmony)


def main():
    shen_khar_venakhi()


if __name__ == "__main__":
    main()
