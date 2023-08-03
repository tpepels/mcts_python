#cython: language_level=3
cdef void c_random_seed(unsigned int seed)
cdef double c_uniform_random(double low, double high)
cdef int c_random(int _min, int _max)
cdef void c_shuffle(list arr)
cdef void c_shuffle_array(long[:] arr)