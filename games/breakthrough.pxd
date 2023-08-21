#cython: language_level=3

cdef int[3] dirs

# List of values representing the importance of each square on the board. In view of player 2.
cdef int[64] lorentz_values

cdef int[3][2] BL_DIR
cdef int[3][2] WH_DIR