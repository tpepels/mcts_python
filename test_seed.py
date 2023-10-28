import random
import os

# Use os.urandom() to generate a cryptographically secure random seed
seed_bytes = os.urandom(8)  # Generate 8 random bytes
seed = int.from_bytes(seed_bytes, "big")  # Convert bytes to an integer

# Set the random seed
random.seed(seed)
print(f"Random seed set to: {seed}")
