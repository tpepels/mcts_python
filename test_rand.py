# cython: language_level=3
import time
from cython.cimports.includes import c_uniform_random, c_random, c_random_seed



for i in range(10):
    print(f"{c_uniform_random(0, 1)}, {c_random(0, 10)}")