from libcpp.set cimport set as cset
from libcpp.pair cimport pair

cdef cset[pair[int, int]] BOARD_CORNERS


cdef int BOARD_SIZE
cdef tuple PASS_MOVE
cdef int[21] PIECE_SIZES
cdef int[21] UNIQUE_ROTATIONS
cdef list[list[int[:,:]]] ROTATED_PIECES 
cdef list[list[int[:,:]]] PIECE_INDICES