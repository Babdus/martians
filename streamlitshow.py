import io

import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
from scipy.io.wavfile import write as write_wav

from calculations import get_samples, get_sine_wave, parse_frequency_function, \
    add_overtones_to_signal, reverse_signal, add_noise, shift_signal, add_gain, parse_amplitude_function, \
    modify_amplitude_with_function, get_attack_curve, get_decay_degree, apply_attack_and_decay, mix, change_sign


def create_audio_player(audio_data, sample_rate):
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_data, sample_rate, subtype='PCM_24', format='wav')
    return virtual_file


def get_overtones_figure(overtones):
    samples = get_samples(len(overtones), 1, start=1)
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.bar(samples, overtones, color='#ee8899', )
    ax.set_ylabel('amplitude')
    ax.set_xlabel('overtone')
    return fig


def plot_signal(signal, duration, sample_rate, figsize=(10, 1.5), ylabel='amplitude', xlabel='time', color='#9988ee'):
    samples = get_samples(duration, sample_rate)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(samples, signal, color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    st.pyplot(fig)


def show_signal(signal, duration, sample_rate, figsize=(10, 1.5), color='#9988ee'):
    st.text('Signal')
    plot_signal(signal, duration, sample_rate, figsize=figsize, color=color)
    st.audio(create_audio_player(signal, sample_rate))


def timbre(signal, frequency_function, sample_rate, duration, modifier_index):
    with st.expander('Timbre'):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            n_overtones = st.number_input('Number of overtones', min_value=1, max_value=100, value=20, step=1,
                                          key=f'overtones{modifier_index}')
            n_overtones = int(n_overtones)
        with col2:
            n_formants = st.number_input('Number of formants', min_value=0, max_value=10, value=3, step=1,
                                         key=f'formants{modifier_index}')
            n_formants = int(n_formants)

        formants = []
        formant_columns = st.columns([1, 1, 1, 3])
        for formant in range(n_formants):
            with formant_columns[0]:
                st.text(f'Formant {formant + 1}')
                mu = st.number_input('Mu (Hz)', min_value=0.0, max_value=n_overtones*2.0, value=formant*4+1.0, step=1.0,
                                     key=f'mu{formant}{modifier_index}')
            with formant_columns[1]:
                st.text('_')
                sigma = st.number_input('Sigma (Hz)', min_value=0.001, max_value=float(sample_rate), value=1.0,
                                        step=1.0, key=f'sigma{formant}{modifier_index}')
            with formant_columns[2]:
                st.text('_')
                amplitude = st.number_input('Amplitude (Pa)', min_value=0.0, max_value=1.0, value=1 - 0.2 * formant,
                                            step=0.05, key=f'amplitude{formant}{modifier_index}')
            formants.append({'mu': mu, 'sigma': sigma, 'amplitude': amplitude})

        signal, overtones = add_overtones_to_signal(signal, frequency_function, duration, sample_rate, formants,
                                                    n_overtones)
        with col3:
            st.text('Overtones')
            figure = get_overtones_figure(overtones)
            st.pyplot(figure)
        with formant_columns[3]:
            show_signal(signal, duration, sample_rate)

    return signal, {'n_overtones': n_overtones, 'formants': formants}


def amplitude_envelope(signal, sample_rate, duration, modifier_index):
    with st.expander('Amplitude Envelope'):

        attack_columns = st.columns([1, 1, 2])

        with attack_columns[0]:
            st.text('Attack')
            attack_duration = st.number_input('Duration (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01,
                                        key=f'ampenv{modifier_index}attdur')
        with attack_columns[1]:
            st.text('_')
            attack_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                                      key=f'ampenv{modifier_index}attdeg')
        attack_curve = get_attack_curve(attack_duration, attack_degree, duration, sample_rate)
        with attack_columns[2]:
            plot_signal(attack_curve, duration, sample_rate)

        decay_columns = st.columns([1, 1, 1, 3])

        with decay_columns[0]:
            st.text('Decay')
            decay_start = st.number_input('Starting time (s)', min_value=0.0, max_value=duration, value=0.05, step=0.01,
                                    key=f'ampenv{modifier_index}decst')
        with decay_columns[1]:
            st.text('_')
            decay_duration = st.number_input('Duration (s)', min_value=0.0, max_value=duration * 5, value=0.05, step=0.01,
                                       key=f'ampenv{modifier_index}decdur')
        with decay_columns[2]:
            st.text('_')
            decay_degree = st.slider('Curve (Pa/exp(s))', min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                                     key=f'ampenv{modifier_index}decdeg')
        decay_curve = get_decay_degree(decay_start, decay_duration, decay_degree, duration, sample_rate)
        with decay_columns[3]:
            plot_signal(decay_curve, duration, sample_rate)

        signal = apply_attack_and_decay(signal, attack_curve, decay_curve)
        with decay_columns[3]:
            show_signal(signal, duration, sample_rate)

    return signal, {'attack_duration': attack_duration, 'attack_degree': attack_degree, 'decay_start': decay_start,
                    'decay_duration': decay_duration, 'decay_degree': decay_degree}


def amplitude_custom_function(signal, sample_rate, duration, modifier_index):
    with st.expander('Amplitude custom function'):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption('Amplitude (Pa) as a function of time (s)')
            f = st.text_input('y =', key=f'ampfunc{modifier_index}', value='x')
            st.caption('Permitted symbols are "x", numbers, constants "e" and "pi", operators +-*/^, the parentheses ()'
                       ', and functions abs, round, sqrt, log, log2, log10, sin, cos, tan, arcsin, arccos, arctan, sinh'
                       ', cosh, tanh, arcsinh, arccosh, arctanh')
        y = parse_amplitude_function(f, duration, sample_rate)
        with col2:
            plot_signal(y, duration, sample_rate)
        signal = modify_amplitude_with_function(signal, y)
        with col2:
            show_signal(signal, duration, sample_rate)
    return signal, {'f': f}


def overdrive(signal, sample_rate, duration, modifier_index):
    with st.expander('Overdrive'):
        col1, col2 = st.columns([1, 1])
        with col1:
            gain = st.slider('Gain (Pa)', min_value=1.0, max_value=20.0, value=1.0, step=0.1, key=f'gain{modifier_index}')
        signal = add_gain(signal, gain)
        with col2:
            show_signal(signal, duration, sample_rate)
    return signal, {'gain': gain}


def shifted_copy(signal, sample_rate, duration, modifier_index):
    with st.expander('Shifted copy'):
        col1, col2 = st.columns([1, 1])
        with col1:
            shift = st.slider('Shift (ms)', min_value=0.0, max_value=100.0, value=5.0,
                              step=0.1, key=f'shit{modifier_index}')
        signal = shift_signal(signal, shift/1000, sample_rate)
        with col2:
            show_signal(signal, duration, sample_rate)
    return signal, {'shift': shift}


def noise(signal, sample_rate, duration, modifier_index):
    with st.expander('Noise'):
        col1, col2 = st.columns([1, 1])
        with col1:
            noise_amount = st.slider('Amount (Pa)', min_value=0.0, max_value=1.0, value=0.1,
                                     step=0.01, key=f'noiseamount{modifier_index}')
            noise_frequency = st.number_input('Frequency (Hz)', min_value=1, max_value=22050, value=4410,
                                              step=1, key=f'noisefreq{modifier_index}')
        signal = add_noise(signal, duration, sample_rate, noise_frequency, noise_amount)
        with col2:
            show_signal(signal, duration, sample_rate)
    return signal, {'noise_amount': noise_amount, 'noise_frequency': noise_frequency}


def reverse(signal, sample_rate, duration, modifier_index):
    with st.expander('Reverse'):

        col1, col2 = st.columns([1, 1])
        with col1:
            st.text('Reverse direction')
            horizontal = st.checkbox('Horizontally', key=f'Reversecheckh{modifier_index}')
            vertical = st.checkbox('Vertically', key=f'Reversecheckv{modifier_index}')

        if horizontal:
            signal = reverse_signal(signal)
        if vertical:
            signal = change_sign(signal)

        with col2:
            show_signal(signal, duration, sample_rate)
    return signal, {'reverse': True}


def none(signal, sample_rate, duration, modifier_index):
    return signal, {}


def generate_signal(i_signal, sample_rate):
    st.header(f'Signal {i_signal}')
    st.subheader('Initial wave')
    col1, col2 = st.columns([1, 1])

    with col1:
        duration = st.slider(
            'Duration (s)',
            min_value=0.0,
            max_value=12.0,
            value=1.0,
            step=0.125,
            key=f'duration{i_signal}'
        )
        st.caption('Frequency (Hz) as a function of time (s)')
        frequency_function_string = st.text_input('y =', key=f'freqfunc{i_signal}', value='x')
        st.caption(
            'Permitted symbols are "x", numbers, constants "e" and "pi", operators +-*/^, the parentheses (), '
            'and functions abs, round, sqrt, log, log2, log10, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh,'
            ' arcsinh, arccosh, arctanh'
        )
        frequency_function = parse_frequency_function(frequency_function_string, duration, sample_rate)
    with col2:
        plot_signal(frequency_function, duration, sample_rate, ylabel='frequency', color='#c488c4')

    signal = get_sine_wave(frequency_function, duration, sample_rate)
    with col2:
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
    show_signal(signal, duration, sample_rate, figsize=(20, 3))

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
    bpm = st.slider('BPM', min_value=15, max_value=960, value=120, key='bpm')

    col1, col2 = st.columns([1, 1])
    with col1:
        beats_per_bar = st.number_input('Beats per bar', min_value=1, max_value=16, value=8, key='beatsperbar')
        beats_per_bar = int(beats_per_bar)
    with col2:
        notes_per_beat = st.number_input('Notes per beat', min_value=1, max_value=16, value=4, key='notesperbeat')
        notes_per_beat = int(notes_per_beat)

    signal_index_matrix = []

    st.text('Select signals')
    columns = st.columns(beats_per_bar)
    for i, col in enumerate(columns):
        signal_index_column = []
        with col:
            for j in range(notes_per_beat):
                i_signal = st.selectbox('', [None] + list(range(len(signals))), key=f'signalselect{j}{i}')
                signal_index_column.append(i_signal)
        signal_index_matrix.append(signal_index_column)

    final_signal, bar_duration = mix(signals, sample_rate, bpm, beats_per_bar, signal_index_matrix)

    show_signal(final_signal, bar_duration, sample_rate, figsize=(20, 3), color='#88c4c4')

    file_name = st.text_input('File name', key=f'filenamefinal')
    save_button = st.button('Save to file', on_click=write_wav, args=(f'data/{file_name}.wav', sample_rate, final_signal),
                            key=f'savebuttonfinal')
    if save_button:
        st.write(f'Saved at data/{file_name}.wav')


def main():
    st.sidebar.text('Sidebar')

    # audio_file = open('data/შენ ხარ ვენახი.wav', 'rb')
    # st.audio(audio_file.read())

    col1, col2, col3 = st.columns([1, 1, 2])

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
