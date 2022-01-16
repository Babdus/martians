import io

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
from scipy.io.wavfile import write as write_wav

from calculations import get_samples, get_sine_wave, function_parser, parse_frequency_function, get_formant, \
    add_overtones_to_signal, reverse_signal, add_noise, shift_signal, add_gain, parse_amplitude_function, \
    modify_amplitude_with_function, get_attack_curve, get_decay_degree, apply_attack_and_decay, signal_pipeline


def create_audio_player(audio_data, sample_rate):
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_data, sample_rate, subtype='PCM_24', format='wav')
    return virtual_file


def get_overtones_figure(overtones):
    samples = get_samples(len(overtones), 1, start=1)
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.bar(samples, overtones)
    return fig


def plot_signal(signal, duration, sample_rate, figsize):
    samples_1 = get_samples(duration, sample_rate)
    fig_1, ax_1 = plt.subplots(figsize=figsize)
    ax_1.plot(samples_1, signal)
    st.pyplot(fig_1)


def show_signal(signal, duration, sample_rate, figsize=(20, 3)):
    st.text('Signal')
    plot_signal(signal, duration, sample_rate, figsize=figsize)
    st.audio(create_audio_player(signal, sample_rate))


def timbre(signal, frequency_function, sample_rate, duration, modifier_index):
    with st.expander('Timbre'):
        col1, col2 = st.columns([1, 1])
        with col1:
            n_overtones = st.number_input('Number of overtones', min_value=1, max_value=100, value=20, step=1,
                                          key=f'overtones{modifier_index}')
            n_overtones = int(n_overtones)
        with col2:
            n_formants = st.number_input('Number of formants', min_value=0, max_value=10, value=3, step=1,
                                         key=f'formants{modifier_index}')
            n_formants = int(n_formants)

        formants = []
        for formant in range(n_formants):
            st.text(f'Formant {formant+1}')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                mu = st.number_input('Mu (Hz)', min_value=0.0, max_value=n_overtones*2.0, value=formant*4+1.0, step=1.0,
                                     key=f'mu{formant}{modifier_index}')
            with col2:
                sigma = st.number_input('Sigma (Hz)', min_value=0.001, max_value=float(sample_rate), value=1.0,
                                        step=1.0, key=f'sigma{formant}{modifier_index}')
            with col3:
                amplitude = st.number_input('Amplitude (Pa)', min_value=0.0, max_value=1.0, value=1 - 0.2 * formant,
                                            step=0.05, key=f'amplitude{formant}{modifier_index}')
            formants.append({'mu': mu, 'sigma': sigma, 'amplitude': amplitude})

        signal, overtones = add_overtones_to_signal(signal, frequency_function, duration, sample_rate, formants,
                                                    n_overtones)
        figure = get_overtones_figure(overtones)
        st.pyplot(figure)
        show_signal(signal, duration, sample_rate)

    return signal, {'n_overtones': n_overtones, 'formants': formants}


def amplitude_envelope(signal, sample_rate, duration, modifier_index):
    with st.expander('Amplitude Envelope'):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.text('Attack')
        with col2:
            st.text('Decay')

        col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 2])

        with col1:
            attack_duration = st.slider('Duration (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01,
                                        key=f'ampenv{modifier_index}attdur')
        with col2:
            attack_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                                      key=f'ampenv{modifier_index}attdeg')

        with col3:
            decay_start = st.slider('Starting time (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01,
                                    key=f'ampenv{modifier_index}decst')
        with col4:
            decay_duration = st.slider('Duration (s)', min_value=0.0, max_value=duration * 5, value=0.05, step=0.01,
                                       key=f'ampenv{modifier_index}decdur')
        with col5:
            decay_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                                     key=f'ampenv{modifier_index}decdeg')

        col1, col2 = st.columns([1, 1])

        attack_curve = get_attack_curve(attack_duration, attack_degree, duration, sample_rate)
        decay_curve = get_decay_degree(decay_start, decay_duration, decay_degree, duration, sample_rate)

        with col1:
            plot_signal(attack_curve, duration, sample_rate, figsize=(10, 2))
        with col2:
            plot_signal(decay_curve, duration, sample_rate, figsize=(10, 2))

        signal = apply_attack_and_decay(signal, attack_curve, decay_curve)
        show_signal(signal, duration, sample_rate)

    return signal, {'attack_duration': attack_duration, 'attack_degree': attack_degree, 'decay_start': decay_start,
                    'decay_duration': decay_duration, 'decay_degree': decay_degree}


def amplitude_custom_function(signal, sample_rate, duration, modifier_index):
    with st.expander('Amplitude custom function'):
        st.caption('Amplitude (Pa) as a function of time (s)')
        f = st.text_input('y =', key=f'ampfunc{modifier_index}', value='x')
        st.caption('Permitted symbols are "x", numbers, constants "e" and "pi", operators +-*/^, the parentheses (), '
                   'and functions abs, round, sqrt, log, log2, log10, sin, cos, tan, arcsin, arccos, arctan, sinh, '
                   'cosh, tanh, arcsinh, arccosh, arctanh')
        y = parse_amplitude_function(f, duration, sample_rate)
        plot_signal(y, duration, sample_rate, figsize=(20, 3))
        signal = modify_amplitude_with_function(signal, y)
        show_signal(signal, duration, sample_rate)
    return signal, {'f': f}


def overdrive(signal, sample_rate, duration, modifier_index):
    with st.expander('Overdrive'):
        gain = st.slider('Gain (Pa)', min_value=1.0, max_value=20.0, value=1.0, step=0.1, key=f'gain{modifier_index}')
        signal = add_gain(signal, gain)
        show_signal(signal, duration, sample_rate)
    return signal, {'gain': gain}


def shifted_copy(signal, sample_rate, duration, modifier_index):
    with st.expander('Shifted copy'):
        shift = st.slider('Shift (s)', min_value=0.0, max_value=0.1, value=0.005,
                          step=0.001, key=f'shit{modifier_index}')
        signal = shift_signal(signal, shift, sample_rate)
        show_signal(signal, duration, sample_rate)
    return signal, {'shift': shift}


def noise(signal, sample_rate, duration, modifier_index):
    with st.expander('Noise'):
        noise_amount = st.slider('Amount (Pa)', min_value=0.0, max_value=1.0, value=0.1,
                                 step=0.01, key=f'noiseamount{modifier_index}')
        noise_frequency = st.number_input('Frequency (Hz)', min_value=1, max_value=22050, value=4410,
                                          step=1, key=f'noisefreq{modifier_index}')
        signal = add_noise(signal, duration, sample_rate, noise_frequency, noise_amount)
        show_signal(signal, duration, sample_rate)
    return signal, {'noise_amount': noise_amount, 'noise_frequency': noise_frequency}


def reverse(signal, sample_rate, duration, modifier_index):
    with st.expander('Reverse'):
        signal = reverse_signal(signal)
        show_signal(signal, duration, sample_rate)
    return signal, {'reverse': True}


def none(signal, sample_rate, duration, modifier_index):
    return signal, {}


def generate_signal(i_signal, sample_rate):
    st.header(f'Signal {i_signal}')
    st.subheader('Initial wave')
    col1, col2 = st.columns([1, 1])

    with col2:
        duration = st.slider(
            'Duration (s)',
            min_value=0.0,
            max_value=12.0,
            value=1.0,
            step=0.125,
            key=f'duration{i_signal}'
        )
    with col1:
        st.caption('Frequency (Hz) as a function of time (s)')
        frequency_function_string = st.text_input('y =', key=f'freqfunc{i_signal}', value='x')
        st.caption(
            'Permitted symbols are "x", numbers, constants "e" and "pi", operators +-*/^, the parentheses (), '
            'and functions abs, round, sqrt, log, log2, log10, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh,'
            ' arcsinh, arccosh, arctanh'
        )
        frequency_function = parse_frequency_function(frequency_function_string, duration, sample_rate)
        plot_signal(frequency_function, duration, sample_rate, figsize=(20, 3))

    signal = get_sine_wave(frequency_function, duration, sample_rate)
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
    signal, timbre_properties = timbre(
        signal=signal,
        frequency_function=frequency_function,
        sample_rate=sample_rate,
        duration=duration,
        modifier_index=f'{i_signal}-1'
    )

    st.subheader('Modifiers')
    col1, col2 = st.columns([1, 2])
    with col1:
        n_modifiers = st.number_input(
            'Number of signal modifiers',
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key=f'nmodifiers{i_signal}'
        )
        n_modifiers = int(n_modifiers)

    modifier_properties = []
    for index in range(n_modifiers):
        col1, col2 = st.columns([1, 2])
        with col1:
            modifier = st.selectbox('Select modifier', list(function_mapper.keys()), key=f'{i_signal}{index}')
        signal, properties = function_mapper[modifier](signal, sample_rate, duration, f'{i_signal}{index}')
        modifier_properties.append({modifier: properties})

    st.subheader('Final signal')
    show_signal(signal, duration, sample_rate)

    file_name = st.text_input('File name', key=f'filename{i_signal}')
    save_button = st.button('Save to file', on_click=write_wav, args=(f'data/{file_name}.wav', sample_rate, signal),
                            key=f'savebutton{i_signal}')
    if save_button:
        st.write(f'Saved at data/{file_name}.wav')

    return signal, {
        'duration': duration,
        'frequency_function_string': frequency_function_string,
        'timbre': timbre_properties,
        'modifier_properties': modifier_properties
    }


def mixer(signals, sample_rate):
    st.header('Mixer')
    bit_rate = st.slider('Bit rate', min_value=15, max_value=960, value=120, key='bitrate')
    bit_duration = 60 / bit_rate
    col1, col2 = st.columns([1, 1])
    with col1:
        bits_per_bar = st.number_input('Bits per bar', min_value=1, max_value=16, value=8, key='bitsperbar')
        bits_per_bar = int(bits_per_bar)
    with col2:
        notes_per_bit = st.number_input('Notes per bit', min_value=1, max_value=16, value=4, key='notesperbit')
        notes_per_bit = int(notes_per_bit)

    sample_per_bit = int(bit_duration * sample_rate)
    sample_per_bar = sample_per_bit * bits_per_bar
    bar_duration = (sample_per_bit * bits_per_bar) / sample_rate
    final_signal = np.zeros(sample_per_bar)

    st.text('Select signals')
    columns = st.columns(bits_per_bar)
    for i, col in enumerate(columns):
        with col:
            for j in range(notes_per_bit):
                i_signal = st.selectbox('', [None] + list(range(len(signals))), key=f'signalselect{j}{i}')

                if i_signal is None:
                    continue
                signal = signals[i_signal]['signal']
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
        sample_rate = st.number_input(
            'Sample rate (Hz)',
            min_value=1000,
            max_value=192000,
            value=44100,
            step=1000,
            key=f'samplerate'
        )
        sample_rate = int(sample_rate)
    with col2:
        n_signals = st.number_input('Number of signals', min_value=1, max_value=64, value=1, step=1)
    n_signals = int(n_signals)
    signals = []
    for i_signal in range(n_signals):
        signal, properties = generate_signal(i_signal, sample_rate)
        signals.append({'signal': signal, 'properties': properties})

    mixer(signals, sample_rate)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Audio signal laboratory")
    main()
