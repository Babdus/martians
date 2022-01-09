import io
import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.io.wavfile import write as write_wav
from scipy.stats import norm


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
            n_overtones = st.number_input('Number of overtones', min_value=1, max_value=100, value=20, step=1, key=f'overtones{modifier_index}')
        with col2:
            n_formants = st.number_input('Number of formants', min_value=0, max_value=10, value=3, step=1, key=f'formants{modifier_index}')
        n_overtones = int(n_overtones)
        n_formants = int(n_formants)
        samples_2 = np.linspace(frequency, frequency*n_overtones, n_overtones, endpoint=True)
        fig_2, ax_2 = plt.subplots(figsize=(20, 3))
        ax_2.bar(samples_2, np.ones(samples_2.shape))
        st.pyplot(fig_2, key=f'plot1{modifier_index}')

        samples_3 = np.arange(frequency, frequency*n_overtones, 1)
        bell_curve_sum = np.zeros(samples_3.shape)

        for formant in range(n_formants):
            st.text(f'Formant {formant+1}')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                mu = st.number_input('Mu (Hz)', min_value=frequency, max_value=sample_rate, value=frequency * (formant+1), step=50, key=f'mu{formant}{modifier_index}')
            with col2:
                sigma = st.number_input('Sigma (Hz)', min_value=1.0, max_value=float(sample_rate), value=100.0 * (formant+1), step=50.0, key=f'sigma{formant}{modifier_index}')
            with col3:
                amplitude = st.number_input('Amplitude', min_value=0.0, max_value=1.0, value=1 - 0.1 * formant, step=0.05, key=f'amplitude{formant}{modifier_index}')

            bell_curve_raw = norm.pdf(samples_3, mu, sigma)
            bell_curve = bell_curve_raw / np.max(bell_curve_raw) * amplitude
            bell_curve_sum += bell_curve

        if n_formants == 0:
            bell_curve_sum = np.ones(samples_3.shape)

        bell_curve_sum = bell_curve_sum / np.max(bell_curve_sum)
        fig_4, ax_4 = plt.subplots(figsize=(20, 3))
        ax_4.plot(samples_3, bell_curve_sum)

        padded_bell_curve_sum = np.pad(bell_curve_sum, (math.floor(frequency/2), math.ceil(frequency/2)), 'edge')
        overtones = padded_bell_curve_sum.reshape(-1, frequency).mean(axis=1)
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
            attack_duration = st.slider('Duration', min_value=0.0, max_value=duration, value=0.05, step=0.01, key=f'ampenv{modifier_index}attdur')
        with col2:
            attack_degree = st.slider('Curve', min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f'ampenv{modifier_index}attdeg')

        with col3:
            decay_start = st.slider('Starting time', min_value=0.0, max_value=duration, value=0.05, step=0.01, key=f'ampenv{modifier_index}decst')
        with col4:
            decay_duration = st.slider('Duration', min_value=0.0, max_value=duration * 5, value=0.05, step=0.01, key=f'ampenv{modifier_index}decdur')
        with col5:
            decay_degree = st.slider('Curve', min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key=f'ampenv{modifier_index}decdeg')

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


def reverse(signal, frequency, sample_rate, duration, modifier_index):
    with st.expander('Reverse'):
        signal = np.flip(signal)
        show_signal(signal, duration, sample_rate)
    return signal


def none(signal, frequency, sample_rate, duration, modifier_index):
    return signal


def main():
    st.sidebar.text('Sidebar')

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        sample_rate = st.number_input('Sample rate', min_value=1000, max_value=192000, value=44100, step=1000)
    with col2:
        frequency = st.number_input('Frequency (Hz)', min_value=1, max_value=22100, value=110, step=1)
    with col3:
        duration = st.slider('Duration (s)', min_value=0.0, max_value=12.0, value=1.0, step=0.125)

    sample_rate = int(sample_rate)
    frequency = int(frequency)

    samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * samples_1)

    st.write(sample_rate)

    show_signal(signal, duration, sample_rate)

    function_mapper = {
        'None': none,
        'Reverse': reverse,
        'Amplitude envelope': amplitude_envelope
    }

    signal = timbre(signal, frequency, sample_rate, duration, -1)

    st.subheader('Modifiers')
    col1, col2 = st.columns([1, 5])
    with col1:
        n_modifiers = st.number_input('Number of signal modifiers', min_value=0, max_value=20, value=0, step=1)
        n_modifiers = int(n_modifiers)

    for index in range(n_modifiers):
        modifier = st.selectbox('Select modifier', list(function_mapper.keys()), key=index)
        signal = function_mapper[modifier](signal, frequency, sample_rate, duration, index)

    st.subheader('Final signal')
    show_signal(signal, duration, sample_rate)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Audio signal laboratory")
    main()
