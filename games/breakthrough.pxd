#cython: language_level=3

cdef int[3] dirs = [-1, 0, 1]

# List of values representing the importance of each square on the board. In view of player 2.
cdef int[64] lorentz_values = [
    5,  15, 15, 5,  5,  15, 15, 5,
    2,  3,  3,  3,  3,  3,  3,  2,
    4,  6,  6,  6,  6,  6,  6,  4,
    7,  10, 10, 10, 10, 10, 10, 7,
    11, 15, 15, 15, 15, 15, 15, 11,
    16, 21, 21, 21, 21, 21, 21, 16,
    20, 28, 28, 28, 28, 28, 28, 20,
    36, 36, 36, 36, 36, 36, 36, 36
]

cdef int[3][2] BL_DIR = [[1, 0], [1, -1], [1, 1]]
cdef int[3][2] WH_DIR = [[-1, 0], [-1, -1], [-1, 1]]