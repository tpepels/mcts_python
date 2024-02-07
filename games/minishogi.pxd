cdef list MOVES
cdef dict PIECES
cdef list PIECE_CHARS
cdef int P2_OFFS
cdef int[10] MATERIAL

# Define the callback type signature
ctypedef bint (*MoveCallback)(int from_row, int from_col, int to_row, int to_col, int piece, bint is_defensive)
