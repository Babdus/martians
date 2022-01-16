import numpy as np
import math
from scipy.stats import norm

constants = ['e', 'pi']

np_functions = [
    'power',
    'abs',
    'sqrt',
    'log10',
    'log2',
    'log',
    'arcsinh',
    'arccosh',
    'arctanh',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'sin',
    'cos',
    'tan',
    'rint'
]


def function_parser(f):
    f = f.replace(' ', '')
    f = f.replace('^', '**')
    if 'x' not in f:
        f = f'(x/x)*({f})'
    for constant in constants:
        f = f.replace(constant, f'math.{constant}')
    f = f.replace('round', 'rint')
    for i, func in enumerate(np_functions):
        f = f.replace(func, chr(i + 65))
    for i in range(len(np_functions)):
        f = f.replace(chr(i + 65), f'np.{np_functions[i]}')
    return f


def get_samples(length, sample_rate, start=0):
    return np.linspace(start, length+start, int(sample_rate * length), endpoint=False)


def parse_frequency_function(frequency_function_string, duration, sample_rate):
    f = function_parser(frequency_function_string)
    x = get_samples(duration, sample_rate)
    frequency_function = eval(f)
    frequency_function = np.nan_to_num(frequency_function, nan=0.0)
    return frequency_function


def get_sine_wave(frequency, duration, sample_rate, amplitude=1):
    samples = get_samples(duration, sample_rate)
    return np.sin(2 * np.pi * frequency * samples) * amplitude


def initialize_signal(duration, sample_rate):
    return np.zeros(get_samples(duration, sample_rate).shape)


def initialize_formant_function(n_overtones, value=0.0):
    return np.full(n_overtones, value)


def get_formant(n_overtones, mu, sigma, amplitude):
    samples = get_samples(n_overtones, 1, start=1)
    bell_curve = norm.pdf(samples, mu, sigma)
    return bell_curve / np.max(bell_curve) * amplitude


def normalize(array):
    return array / np.max(array)


def add_overtones_to_signal(signal, frequency_function, duration, sample_rate, formants, n_overtones):
    formant_functions = []
    for formant in formants:
        formant_functions.append(get_formant(n_overtones, formant['mu'], formant['sigma'], formant['amplitude']))
    overtones = initialize_formant_function(n_overtones, value=(1.0 if len(formant_functions) == 0 else 0.0))
    for formant_function in formant_functions:
        overtones += formant_function
    overtones = normalize(overtones)
    last_signal_max = np.max(signal)
    signal = initialize_signal(duration, sample_rate)
    for overtone, amplitude in enumerate(overtones, start=1):
        signal += get_sine_wave(frequency_function * overtone, duration, sample_rate, amplitude)

    return (signal * last_signal_max) / np.max(signal), overtones


def reverse_signal(signal):
    return np.flip(signal)


def add_noise(signal, duration, sample_rate, noise_frequency, noise_amount):
    samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise_wave = np.sin(2 * np.pi * noise_frequency * samples_1)
    noise_wave *= np.random.rand(samples_1.shape[0]) * noise_amount
    signal += noise_wave
    return normalize(signal)


def shift_signal(signal, shift, sample_rate):
    shift_samples = int(sample_rate * shift)
    shifted_signal = np.pad(signal, (shift_samples,), 'constant', constant_values=(0, 0))[:signal.shape[0]]
    signal += shifted_signal
    return normalize(signal)


def add_gain(signal, gain):
    signal *= gain
    signal = np.where(signal > 1, 1, signal)
    signal = np.where(signal < -1, -1, signal)
    return signal


def parse_amplitude_function(f, duration, sample_rate):
    f = function_parser(f)
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = eval(f)
    return y


def modify_amplitude_with_function(signal, y):
    signal *= y
    return signal


def get_attack_curve(attack_duration, attack_degree, duration, sample_rate):
    samples = get_samples(duration, sample_rate)
    attack_curve = np.power(samples, 2 ** attack_degree) / np.power(attack_duration, 2 ** attack_degree)
    attack_curve = np.where(attack_curve > 1, 1, attack_curve)
    return attack_curve


def get_decay_degree(decay_start, decay_duration, decay_degree, duration, sample_rate):
    samples = get_samples(decay_duration, sample_rate)
    decay_degree = -decay_degree
    decay_curve = 1 - np.power(samples, 2 ** decay_degree) / np.power(decay_duration, 2 ** decay_degree)
    decay_curve = np.where(decay_curve > 1, 1, decay_curve)
    decay_curve = np.where(decay_curve < 0, 0, decay_curve)
    n_samples_before_start_decay = int(decay_start * sample_rate)
    n_samples_after_end_decay = int(sample_rate * duration) - n_samples_before_start_decay - int(
        sample_rate * decay_duration)
    n_samples_after_end_decay = n_samples_after_end_decay if n_samples_after_end_decay > 0 else 0
    decay_curve = np.pad(decay_curve, (n_samples_before_start_decay, n_samples_after_end_decay), 'edge')
    decay_curve = decay_curve[:int(sample_rate * duration)]
    return decay_curve


def apply_attack_and_decay(signal, attack_curve, decay_curve):
    signal *= attack_curve * decay_curve
    return signal


def signal_pipeline(properties, sample_rate):
    duration = properties['duration']
    frequency_function_string = properties['frequency_function_string']
    frequency_function = parse_frequency_function(frequency_function_string, duration, sample_rate)
    signal = get_sine_wave(frequency_function, duration, sample_rate)

    n_overtones = properties['timbre']['n_overtones']
    formants = properties['timbre']['formants']
    signal, overtones = add_overtones_to_signal(signal, frequency_function, duration, sample_rate, formants,
                                                n_overtones)
    for modifier_properties in properties['modifier_properties']:
        if 'Reverse' in modifier_properties:
            signal = reverse_signal(signal)
        elif 'Overdrive' in modifier_properties:
            signal = add_gain(signal, modifier_properties['Overdrive']['gain'])
        elif 'Shifted copy' in modifier_properties:
            signal = shift_signal(signal, modifier_properties['Shifted copy']['shift'], sample_rate)
        elif 'Noise' in modifier_properties:
            signal = add_noise(signal, duration, sample_rate, modifier_properties['Noise']['noise_frequency'],
                               modifier_properties['Noise']['noise_amount'])
        elif 'Amplitude custom function' in modifier_properties:
            y = parse_amplitude_function(modifier_properties['Amplitude custom function']['f'], duration, sample_rate)
            signal = modify_amplitude_with_function(signal, y)
        elif 'Amplitude envelope' in modifier_properties:
            attack_curve = get_attack_curve(modifier_properties['Amplitude envelope']['attack_duration'],
                                            modifier_properties['Amplitude envelope']['attack_degree'], duration,
                                            sample_rate)
            decay_curve = get_decay_degree(modifier_properties['Amplitude envelope']['decay_start'],
                                           modifier_properties['Amplitude envelope']['decay_duration'],
                                           modifier_properties['Amplitude envelope']['decay_degree'], duration,
                                           sample_rate)
            signal = apply_attack_and_decay(signal, attack_curve, decay_curve)
    return signal
