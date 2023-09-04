#cython: language_level=3

from libc.stdlib cimport srand, rand, RAND_MAX
from libc.math cimport tanh

cdef double Z

cdef inline void c_random_seed(unsigned int seed):
    srand(seed)

cdef inline double c_uniform_random(double low, double high) except -99999:
    return low + (rand() / RAND_MAX) * (high - low)

cdef inline int c_random(int _min, int _max) except -99999:
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

cdef inline double normalize(double value, double a) except -99999:
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

cdef inline list where_is_k2d(int[:,:] board , int k):
    cdef int i
    cdef int j
    cdef list indices = []

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == k:
                indices.append((i, j))

    return indices


cdef inline int f_index(int[:] arr, int value, int n) except -2:
    cdef int i
    for i in range(n):  # Assuming the second dimension always has size 4
        if arr[i] == value:
            return i
    return -1  # Return -1 if the value is not found

cpdef list generate_spiral(int size)


cdef inline int find_2d_index(int[:, :] arr, int x, int y)  except -2:
    cdef int i

    # Iterate through the positions for the given size
    for i in range(arr.shape[0] * arr.shape[0]):
        if arr[i][0] == x and arr[i][1] == y:
            return i  # Return the index if the coordinates match

    return -1  # Return -1 if the coordinates were not found

