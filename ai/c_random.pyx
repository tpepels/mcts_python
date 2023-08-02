from libc.stdlib cimport srand, rand, RAND_MAX

cdef void c_random_seed(unsigned int seed):
    srand(seed)

cdef double c_uniform_random(double low, double high):
    return low + (rand() / RAND_MAX) * (high - low)

cdef int c_random(int n):
    return rand() % n