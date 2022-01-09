import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm
from scipy.io.wavfile import write as write_wav

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    sample_rate = st.number_input('Sample rate', min_value=1000, max_value=192000, value=44100, step=1000)
with col2:
    frequency = st.number_input('Frequency (Hz)', min_value=1, max_value=sample_rate, value=110, step=1)
with col3:
    duration = st.number_input('Duration (s)', min_value=0.0, max_value=30.0, value=1.0, step=0.125)

samples_1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * frequency * samples_1)
fig_1, ax_1 = plt.subplots(figsize=(10, 5))
ax_1.plot(samples_1, signal)
st.pyplot(fig_1)

write_wav('data/signal.wav', sample_rate, signal)
audio_file = open('data/signal.wav', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/wav')

col1, col2 = st.columns([1, 1])

with col1:
    n_overtones = st.number_input('Number of overtones', min_value=1, max_value=100, value=20, step=1)
with col2:
    n_formants = st.number_input('Number of formants', min_value=0, max_value=10, value=3, step=1)

samples_2 = np.linspace(frequency, frequency*n_overtones, n_overtones, endpoint=True)
fig_2, ax_2 = plt.subplots(figsize=(10, 3))
ax_2.bar(samples_2, np.ones(samples_2.shape))
st.pyplot(fig_2)

samples_3 = np.arange(frequency, frequency*n_overtones, 1)
bell_curve_sum = np.zeros(samples_3.shape)

for formant in range(n_formants):
    st.text(f'Formant {formant+1}')
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        mu = st.number_input('Mu (Hz)', min_value=frequency, max_value=sample_rate, value=frequency * (formant+1), step=50, key=formant*3)
    with col2:
        sigma = st.number_input('Sigma (Hz)', min_value=1.0, max_value=float(sample_rate), value=100.0 * (formant+1), step=50.0, key=formant*3+1)
    with col3:
        amplitude = st.number_input('Amplitude', min_value=0.0, max_value=1.0, value=1 - 0.1 * formant, step=0.05, key=formant*3+2)

    samples_3 = np.arange(frequency, frequency*n_overtones, 1)
    bell_curve_raw = norm.pdf(samples_3, mu, sigma)
    bell_curve = bell_curve_raw / np.max(bell_curve_raw) * amplitude
    # fig_3, ax_3 = plt.subplots(figsize=(10, 3))
    # ax_3.plot(samples_3, bell_curve)
    # st.pyplot(fig_3)
    bell_curve_sum += bell_curve

bell_curve_sum = bell_curve_sum / np.max(bell_curve_sum)
fig_4, ax_4 = plt.subplots(figsize=(10, 3))
ax_4.plot(samples_3, bell_curve_sum)
ax_4.bar(samples_2, np.ones(samples_2.shape))
st.pyplot(fig_4)
