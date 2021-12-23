import math
import random
from typing import Dict

from schemas import Curve


def generate_formants(
        base_divisor: float = 2,
        overtone_decay: Curve = Curve.exponential,
        overtone_random: float = 0.25,
        n_overtones: int = 16,
) -> Dict:
    formants = {}
    for i in range(n_overtones):
        if overtone_decay == Curve.constant:
            formants[i] = 1 / base_divisor
        elif overtone_decay == Curve.logarithmic:
            formants[i] = 1 / (base_divisor + math.log(i + 1))
        elif overtone_decay == Curve.linear:
            formants[i] = 1 / (base_divisor + i)
        elif overtone_decay == Curve.quadratic:
            formants[i] = 1 / (base_divisor * (i + 1))
        elif overtone_decay == Curve.exponential:
            formants[i] = 1 / (base_divisor ** (i + 1))
        else:
            raise ValueError
        formants[i] += random.uniform(-formants[i] * overtone_random / 2, formants[i] * overtone_random / 2)
    print(formants)
    return formants
