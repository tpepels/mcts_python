import cython
from libc.time cimport time

Z = 1.96  # 95% confidence interval

@cython.cdivision(False) # * Important * - without this the results differ and are incorrect
cpdef list generate_spiral(int size):
    x = y = 0
    dx = 0
    dy = -1
    spiral_coords = []
    adjusted_size = size + 1 if size % 2 == 0 else size
    for i in range(adjusted_size**2):
        if (-adjusted_size/2 < x <= adjusted_size/2) and (-adjusted_size/2 < y <= adjusted_size/2):
            adjusted_x = (x + adjusted_size//2) % size
            adjusted_y = (y + adjusted_size//2) % size
            spiral_coords.append((adjusted_x, adjusted_y))
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return spiral_coords[:size**2]

cdef unsigned int hash_tuple(tuple input_tuple, short max_value) except -1:
    cdef unsigned int hash_value = 0
    cdef unsigned int number
    cdef short i
    for i in range(len(input_tuple)):
        if input_tuple[i] < 0:
            number = -input_tuple[i] + 999
        else:
            number = input_tuple[i]
        hash_value = (hash_value * 31 + (number + (number << 10)))
        hash_value ^= (hash_value >> 6)
    
    # Finalize the hash value to enhance the avalanche effect
    hash_value += (hash_value << 3)
    hash_value ^= (hash_value >> 11)
    hash_value += (hash_value << 15)
    # Ensure the hash value is non-negative and fits within the specified range
    return hash_value % max_value