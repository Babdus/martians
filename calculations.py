import numpy as np
from matplotlib import pyplot as plt
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
