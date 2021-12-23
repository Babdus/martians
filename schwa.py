from scipy.io.wavfile import write
import math
import random
import numpy as np
import pandas as pd
import sys

def formant(herz, sigma, power, x):
    return math.exp(power * (1/np.cosh((x-herz)/sigma)))-1

df = pd.read_csv('vowel.csv', index_col="vowel")

row = df.loc['e']

v_f = { 0: lambda x: formant(row['f0_hz'], row['f0_σ'], row['f0_p'], x),
        1: lambda x: formant(row['f1_hz'], row['f1_σ'], row['f1_p'], x),
        2: lambda x: formant(row['f2_hz'], row['f2_σ'], row['f2_p'], x),
        3: lambda x: formant(row['f3_hz'], row['f3_σ'], row['f3_p'], x),
        5: lambda x: formant(row['f4_hz'], row['f4_σ'], row['f4_p'], x) }

base_freq = 150

v_o = {}
for i in range(1, 41):
    v_o[i*base_freq] = sum(v_f[f](i*base_freq) for f in v_f)

# for i in range(1,41):
#     power = 1/i * formants[i-1]
#
#     overtones[i] = round(power, 2)

print(v_o)

rate = 48000
signal = []
decay_rate = random.uniform(1,4)
for i in range(rate//2):
    value = 0
    if i/rate < 1:
        v = v_o
    elif i/rate < 2:
        v = v_o
    elif i/rate < 3:
        v = v_o
    elif i/rate < 4:
        v = v_o
    elif i/rate < 5:
        v = v_o
    for overtone in v:
        amp = 1
        noise = 1
        value += math.sin(math.pi * 2 * overtone * (i / rate) * noise) * v[overtone] / 4
    signal.append(value)

write('data/e.wav', rate, np.array(signal))