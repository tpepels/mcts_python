#cython: language_level=3

cdef char[3] dirs

# List of values representing the importance of each square on the board. In view of player 2.
cdef char[64] lorentz_values

cdef char[3][2] BL_DIR
cdef char[3][2] WH_DIR