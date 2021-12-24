from enum import Enum


class Curve(Enum):
    constant = 0
    logarithmic = 1
    linear = 2
    quadratic = 3
    exponential = 4
    polynomial = 5
