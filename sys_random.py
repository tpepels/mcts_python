import random
import time

# Create an instance of SystemRandom
sys_rng = random.SystemRandom()

# Get a random number from SystemRandom and combine it with the current time
combined_seed = int(sys_rng.random() * 1e8) + int(time.time() * 1e6)
print(int(sys_rng.random() * 1e8))
# Set the seed for the default random number generator
random.seed(combined_seed)

for i in range(5):
    print(random.random())

for i in range(5):
    print(sys_rng.random())