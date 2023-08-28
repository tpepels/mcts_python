from libcpp.set cimport set as cset
from libcpp.pair cimport pair

cdef cset[pair[int, int]] BOARD_CORNERS


cdef int BOARD_SIZE
cdef tuple PASS_MOVE
cdef int[21] PIECE_SIZES
cdef int[21] UNIQUE_ROTATIONS
cdef list[list[int[:,:]]] ROTATED_PIECES 
cdef list[list[int[:,:]]] PIECE_INDICES

# TODO Dit moet je nog controleren
cdef inline int hash_action(int x, int y, int piece_index, int rotation):
    cdef int hash = 17  # A random prime number to start things off
    hash = (hash << 5) ^ x
    hash = (hash << 5) ^ y
    hash = (hash << 5) ^ piece_index
    hash = (hash << 5) ^ rotation
    return hash