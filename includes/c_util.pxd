#cython: language_level=3

from libc.stdlib cimport srand, rand, RAND_MAX
from libc.math cimport tanh

cdef double Z

cdef inline float normalize(float value, float a) except -99999:
    """
    Normalize value with range [-a,a] to [-1, 1] using tanh

    Args:
        value (float): the value to normalize
        a (float): absolute max
    """
    return tanh(value / a)

cdef inline list[char] where_is_k(char[:] board , char k):
    cdef short i
    cdef list indices = []

    for i in range(board.shape[0]):
        if board[i] == k:
            indices.append(i)

    return indices


cdef inline list[char] where_is_k2d(char[:,:] board , char k):
    cdef short i
    cdef short j
    cdef list indices = []

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] == k:
                indices.append((i, j))

    return indices

cdef inline short f_index(char[:] arr, char value, int n) except -2:
    cdef short i
    for i in range(n):  # Assuming the second dimension always has size 4
        if arr[i] == value:
            return i
    return -1  # Return -1 if the value is not found

cpdef list generate_spiral(int size)


cdef inline short find_2d_index(char[:, :] arr, short x, short y)  except -2:
    cdef short i

    # Iterate through the positions for the given size
    for i in range(arr.shape[0] * arr.shape[0]):
        if arr[i][0] == x and arr[i][1] == y:
            return i  # Return the index if the coordinates match

    return -1  # Return -1 if the coordinates were not found

