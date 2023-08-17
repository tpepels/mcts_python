#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: initializedcheck=False
#cython: overflowcheck=False

from libc.stdlib cimport srand, rand, RAND_MAX
from libc.math cimport tanh

cdef inline void c_random_seed(unsigned int seed):
    srand(seed)

cdef inline double c_uniform_random(double low, double high):
    return low + (rand() / RAND_MAX) * (high - low)

cdef inline int c_random(int _min, int _max):
    return _min + rand() % (_max - _min + 1)

cdef inline void c_shuffle(list arr):
    cdef int n = len(arr)
    cdef int swap_idx
    cdef object tmp

    for i in range(n-1, 0, -1):
        swap_idx = c_random(0, i)
        tmp = arr[i]
        arr[i] = arr[swap_idx]
        arr[swap_idx] = tmp

cdef inline void c_shuffle_array(long[:] arr):
    cdef int n = arr.shape[0]
    cdef int swap_idx
    cdef long tmp

    for i in range(n-1, 0, -1):
        swap_idx = c_random(0, i)
        tmp = arr[i]
        arr[i] = arr[swap_idx]
        arr[swap_idx] = tmp

cdef inline double normalize(double value, double a):
    """
    Normalize value with range [-a,a] to [-1, 1] using tanh

    Args:
        value (float): the value to normalize
        a (float): absolute max
    """
    return tanh(value / a)

cdef inline list where_is_k(int[:] board , int k):
    cdef int i
    cdef list indices = []

    for i in range(board.shape[0]):
        if board[i] == k:
            indices.append(i)

    return indices