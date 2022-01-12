import io
import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.io.wavfile import write as write_wav
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


def create_audio_player(audio_data, sample_rate):
    virtual_file = io.BytesIO()
    write_wav(virtual_file, rate=sample_rate, data=audio_data)

    return virtual_file


def plot_signal(signal, duration, sample_rate, figsize):
    samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    fig_1, ax_1 = plt.subplots(figsize=figsize)
    ax_1.plot(samples_1, signal)
    st.pyplot(fig_1)


def show_signal(signal, duration, sample_rate, figsize=(20, 3)):
    st.text('Signal')
    plot_signal(signal, duration, sample_rate, figsize=figsize)
    st.audio(create_audio_player(signal, sample_rate))


def timbre(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Timbre'):
        col1, col2 = st.columns([1, 1])
        with col1:
            n_overtones = st.number_input('Number of overtones', min_value=2, max_value=100, value=2, step=1, key=f'overtones{modifier_index}')
        with col2:
            n_formants = st.number_input('Number of formants', min_value=0, max_value=10, value=0, step=1, key=f'formants{modifier_index}')
        n_overtones = int(n_overtones)
        n_formants = int(n_formants)
        int_frequency = int(frequency)
        samples_2 = np.linspace(int_frequency, int_frequency*n_overtones, n_overtones, endpoint=True)
        fig_2, ax_2 = plt.subplots(figsize=(20, 3))
        ax_2.bar(samples_2, np.ones(samples_2.shape))
        st.pyplot(fig_2, key=f'plot1{modifier_index}')

        samples_3 = np.arange(int_frequency, int_frequency*n_overtones, 1)
        bell_curve_sum = np.zeros(samples_3.shape)

        for formant in range(n_formants):
            st.text(f'Formant {formant+1}')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                mu = st.number_input('Mu (Hz)', min_value=int_frequency, max_value=sample_rate, value=int_frequency * (formant+1), step=50, key=f'mu{formant}{modifier_index}')
            with col2:
                sigma = st.number_input('Sigma (Hz)', min_value=1.0, max_value=float(sample_rate), value=100.0 * (formant+1), step=50.0, key=f'sigma{formant}{modifier_index}')
            with col3:
                amplitude = st.number_input('Amplitude (Pa)', min_value=0.0, max_value=1.0, value=1 - 0.1 * formant, step=0.05, key=f'amplitude{formant}{modifier_index}')

            bell_curve_raw = norm.pdf(samples_3, mu, sigma)
            bell_curve = bell_curve_raw / np.max(bell_curve_raw) * amplitude
            bell_curve_sum += bell_curve

        if n_formants == 0:
            bell_curve_sum = np.ones(samples_3.shape)

        bell_curve_sum = bell_curve_sum / np.max(bell_curve_sum)
        fig_4, ax_4 = plt.subplots(figsize=(20, 3))
        ax_4.plot(samples_3, bell_curve_sum)

        padded_bell_curve_sum = np.pad(bell_curve_sum, (math.floor(int_frequency/2), math.ceil(int_frequency/2)), 'edge')
        overtones = padded_bell_curve_sum.reshape(-1, int_frequency).mean(axis=1)
        ax_4.bar(samples_2, overtones)
        st.pyplot(fig_4, key=f'plot2{modifier_index}')

        overtones = overtones / np.max(overtones)

        samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        last_signal_max = np.max(signal)
        signal = np.zeros(samples_1.shape)
        for overtone, amplitude in enumerate(overtones, start=1):
            signal += np.sin(2 * np.pi * frequency*overtone * samples_1) * amplitude

        signal = (signal * last_signal_max) / np.max(signal)

        show_signal(signal, duration, sample_rate)

    return signal


def amplitude_envelope(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Amplitude Envelope'):
        samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.text('Attack')
        with col2:
            st.text('Decay')

        col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 2])

        with col1:
            attack_duration = st.slider('Duration (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01, key=f'ampenv{modifier_index}attdur')
        with col2:
            attack_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f'ampenv{modifier_index}attdeg')

        with col3:
            decay_start = st.slider('Starting time (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01, key=f'ampenv{modifier_index}decst')
        with col4:
            decay_duration = st.slider('Duration (s)', min_value=0.0, max_value=duration * 5, value=0.05, step=0.01, key=f'ampenv{modifier_index}decdur')
        with col5:
            decay_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f'ampenv{modifier_index}decdeg')

        col1, col2 = st.columns([1, 1])

        with col1:
            attack_curve = np.power(samples_1, 2 ** attack_degree) / np.power(attack_duration, 2 ** attack_degree)
            attack_curve = np.where(attack_curve > 1, 1, attack_curve)
            plot_signal(attack_curve, duration, sample_rate, figsize=(10, 2))
        with col2:

            samples_2 = np.linspace(0, decay_duration, int(sample_rate * decay_duration), endpoint=False)

            decay_degree = -decay_degree
            decay_curve = 1 - np.power(samples_2, 2 ** decay_degree) / np.power(decay_duration, 2 ** decay_degree)
            decay_curve = np.where(decay_curve > 1, 1, decay_curve)
            decay_curve = np.where(decay_curve < 0, 0, decay_curve)
            n_samples_before_start_decay = int(decay_start * sample_rate)
            n_samples_after_end_decay = int(sample_rate * duration) - n_samples_before_start_decay - int(sample_rate * decay_duration)
            n_samples_after_end_decay = n_samples_after_end_decay if n_samples_after_end_decay > 0 else 0
            decay_curve = np.pad(decay_curve, (n_samples_before_start_decay, n_samples_after_end_decay), 'edge')
            decay_curve = decay_curve[:int(sample_rate * duration)]
            plot_signal(decay_curve, duration, sample_rate, figsize=(10, 2))

        signal *= attack_curve * decay_curve
        show_signal(signal, duration, sample_rate)

    return signal


def amplitude_custom_function(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Amplitude custom function'):
        st.caption('Amplitude (Pa) as a function of time (s)')
        f = st.text_input('y =', key=f'ampfunc{modifier_index}', value='x')
        st.caption('Permitted symbols are "x", numbers, constants "e" and "pi", operators +-*/^, the parentheses (), and functions abs, round, sqrt, log, log2, log10, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh')
        f = function_parser(f)
        x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        y = eval(f)
        plot_signal(y, duration, sample_rate, figsize=(20, 3))
        signal *= y
        show_signal(signal, duration, sample_rate)
    return signal


def overdrive(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Overdrive'):
        gain = st.slider('Gain (Pa)', min_value=1.0, max_value=20.0, value=1.0, step=0.1, key=f'gain{modifier_index}')

        signal *= gain
        signal = np.where(signal > 1, 1, signal)
        signal = np.where(signal < -1, -1, signal)

        show_signal(signal, duration, sample_rate)
    return signal


def shifted_copy(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Shifted copy'):
        shift = st.slider('Shift (s)', min_value=0.0, max_value=0.1, value=0.005, step=0.001, key=f'shit{modifier_index}')
        shift_samples = int(sample_rate * shift)
        shifted_signal = np.pad(signal, (shift_samples,), 'constant', constant_values=(0, 0))[:signal.shape[0]]
        signal += shifted_signal
        signal = signal / max(signal)

        show_signal(signal, duration, sample_rate)
    return signal


def noise(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Noise'):
        noise_amount = st.slider('Amount (Pa)', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f'noiseamount{modifier_index}')
        noise_frequency = st.number_input('Frequency (Hz)', min_value=1, max_value=22050, value=4410, step=1, key=f'noisefreq{modifier_index}')
        samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        noise_wave = np.sin(2 * np.pi * noise_frequency * samples_1)
        noise_wave *= np.random.rand(samples_1.shape[0]) * noise_amount
        signal += noise_wave
        signal = signal / max(signal)

        show_signal(signal, duration, sample_rate)
    return signal


def reverse(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Reverse'):
        signal = np.flip(signal)
        show_signal(signal, duration, sample_rate)
    return signal


def none(signal, frequency, sample_rate, duration, modifier_index):
    return signal


def generate_signal(i_signal, sample_rate):
    st.header(f'Signal {i_signal}')
    st.subheader('Initial wave')
    col1, col2 = st.columns([1, 1])

    with col1:
        frequency = st.number_input('Frequency (Hz)', min_value=1.0, max_value=22050.0, value=110.0, step=1.0, key=f'freq{i_signal}')
    with col2:
        duration = st.slider('Duration (s)', min_value=0.0, max_value=12.0, value=1.0, step=0.125, key=f'duration{i_signal}')

    samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples_1)

    show_signal(signal, duration, sample_rate)

    function_mapper = {
        'None': none,
        'Reverse': reverse,
        'Amplitude envelope': amplitude_envelope,
        'Amplitude custom function': amplitude_custom_function,
        'Overdrive': overdrive,
        'Shifted copy': shifted_copy,
        'Noise': noise
    }
    st.subheader('Timbre')
    signal = timbre(signal, frequency, sample_rate, duration, f'{i_signal}-1')

    st.subheader('Modifiers')
    col1, col2 = st.columns([1, 2])
    with col1:
        n_modifiers = st.number_input('Number of signal modifiers', min_value=0, max_value=20, value=0, step=1, key=f'nmodifiers{i_signal}')
        n_modifiers = int(n_modifiers)

    for index in range(n_modifiers):
        col1, col2 = st.columns([1, 2])
        with col1:
            modifier = st.selectbox('Select modifier', list(function_mapper.keys()), key=f'{i_signal}{index}')
        signal = function_mapper[modifier](signal, frequency, sample_rate, duration, f'{i_signal}{index}')

    st.subheader('Final signal')
    show_signal(signal, duration, sample_rate)

    file_name = st.text_input('File name', key=f'filename{i_signal}')
    save_button = st.button('Save to file', on_click=write_wav, args=(f'data/{file_name}.wav', sample_rate, signal), key=f'savebutton{i_signal}')
    if save_button:
        st.write(f'Saved at data/{file_name}.wav')

    return signal


def mixer(signals, sample_rate):
    bit_rate = st.slider('Bit rate', min_value=15, max_value=960, value=120, key='bitrate')
    bit_duration = 60 / bit_rate
    col1, col2 = st.columns([1, 1])
    with col1:
        bits_per_bar = st.number_input('Bits per bar', min_value=1, max_value=16, value=8, key='bitsperbar')
        bits_per_bar = int(bits_per_bar)
    with col2:
        notes_per_bit = st.number_input('Notes per bit', min_value=1, max_value=16, value=4, key='notesperbit')
        notes_per_bit = int(notes_per_bit)

    bar_duration = bit_duration * bits_per_bar
    sample_per_bit = int(bit_duration * sample_rate)
    sample_per_bar = sample_per_bit * bits_per_bar
    final_signal = np.zeros(sample_per_bar)

    st.text('Select signals')
    columns = st.columns(bits_per_bar)
    for i, col in enumerate(columns):
        with col:
            for j in range(notes_per_bit):
                i_signal = st.selectbox('', [None] + list(range(len(signals))), key=f'signalselect{j}{i}')

                if i_signal is None:
                    continue
                signal = signals[i_signal]
                if len(signal) + sample_per_bit * i > sample_per_bar:
                    final_signal[sample_per_bit * i:] += signal[:sample_per_bar-sample_per_bit * i]
                else:
                    final_signal[sample_per_bit * i:sample_per_bit * i + len(signal)] += signal
    show_signal(final_signal, bar_duration, sample_rate)

    file_name = st.text_input('File name', key=f'filenamefinal')
    save_button = st.button('Save to file', on_click=write_wav, args=(f'data/{file_name}.wav', sample_rate, final_signal),
                            key=f'savebuttonfinal')
    if save_button:
        st.write(f'Saved at data/{file_name}.wav')


def main():
    st.sidebar.text('Sidebar')

    # audio_file = open('data/შენ ხარ ვენახი.wav', 'rb')
    # st.audio(audio_file.read())

    col1, col2 = st.columns([1, 1])

    with col1:
        sample_rate = st.number_input('Sample rate (Hz)', min_value=1000, max_value=192000, value=44100, step=1000,
                                      key=f'samplerate')
        sample_rate = int(sample_rate)
    with col2:
        n_signals = st.number_input('Number of signals', min_value=1, max_value=64, value=1, step=1)
    n_signals = int(n_signals)
    signals = []
    for i_signal in range(n_signals):
        signal = generate_signal(i_signal, sample_rate)
        signals.append(signal)

    mixer(signals, sample_rate)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Audio signal laboratory")
    main()
