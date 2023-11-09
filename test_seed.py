import random
import os

# Use os.urandom() to generate a cryptographically secure random seed
seed_bytes = os.urandom(8)  # Generate 8 random bytes
seed = int.from_bytes(seed_bytes, "little")  # Convert bytes to an integer

# Set the random seed
random.seed(seed)
print(f"Random seed set to: {seed}")

combinations = []

for i in range(10):
    for j in range(3):
        # Join the two digits to make a two-digit number
        number = int(str(i) + str(j))

        # Exclude 0 from the list
        if number != 0:
            combinations.append(str(number))

# Print the list of combinations
print(",".join(combinations))
print(len(combinations))
