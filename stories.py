from scipy.io.wavfile import write
import math
import random
import numpy as np
import sys

signals = []

for s in range(int(sys.argv[1])):
    tubes = {}

    for i in range(3):

        base_frequency = random.randint(2,5) * 20
        overtone = 1
        formants = {}
        while overtone < 15:
            step_multiplier = overtone//10 + 1
            formants[overtone] = random.uniform(1,3) / step_multiplier
            step = random.randint(step_multiplier,6*step_multiplier)
            overtone += step

        print(formants)

        tubes[base_frequency] = formants

    rate = 48000
    signal = []
    decay_rate = random.uniform(1,4)
    for i in range(rate * 4):
        amplitude = 0
        for base_frequency in tubes:
            formants = tubes[base_frequency]
            value = 0
            for overtone in formants:
                amp = formants[overtone]/32 * math.exp(-i*decay_rate/rate)
                noise = random.uniform(1,1+math.exp(-i*decay_rate*2/rate)/10000)
                value += math.sin(math.pi * 2 * overtone * base_frequency * (i / rate) * noise) * amp
        signal.append(value)

    write(f'story{s}.wav', rate, np.array(signal))
    signals.append(signal)
    print('signal len', len(signal))

final_signal = []

for n in range(16):
    if len(final_signal) == 0:
        final_signal = signals[n%4].copy()
        print(len(signals[n%4]))
    else:
        print(n, len(final_signal))
        for i in range(int(rate*3.5)):
            final_signal[-int(rate*3.5)+i] += signals[n%4][i]
        print(n, len(final_signal))
        final_signal += signals[n%4][int(rate*3.5):]
        print(len(signals[n%4]))
        print(n, len(final_signal))
print(n, len(final_signal))

write('final_story.wav', rate, np.array(final_signal))
