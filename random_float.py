import time
from fastrand import pcg32randint as randint, pcg32_seed

pcg32_seed(int(time.time()))
print(f"Set seed to current time.: {int(time.time())}")


def uniform(a, b) -> float:
    "Get a random number in the range [a, b)."
    # Normalize rand() to [0, 1)
    random_float: float = randint(1, 10000) / 10000
    # Scale and translate to range [a, b)
    return a + (b - a) * random_float


print(uniform(0, 1))
print(uniform(0, 1))
print(uniform(0, 1))
print(uniform(0, 1))
