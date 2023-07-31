from collections import defaultdict, OrderedDict

cdef class MoveHistory:
    cdef object table
    cpdef int get(self, tuple move)
    cpdef void update(self, tuple move, int increment)

cdef class TranspositionTable:
    cdef public unsigned long size
    cdef object table
    cdef unsigned c_cache_hits
    cdef unsigned c_cache_misses
    cdef unsigned c_collisions
    cdef unsigned c_cleanups
    cdef unsigned cache_hits
    cdef unsigned cache_misses
    cdef unsigned collisions
    cdef unsigned cleanups
    cpdef public tuple get(self, long long key, unsigned depth, int player, str board=*)
    cpdef public void put(self, long long key, float value, unsigned depth, int player, tuple best_move, str board=*)
    cpdef public void reset_metrics(self)
    cpdef public dict get_metrics(self)
    cpdef public dict get_cumulative_metrics(self)

# A struct that represents an entry in the transposition table
cdef struct MCTSEntry:
    float v1
    float v2
    float im_value
    int visits
    int solved_player
    bint is_expanded

cdef class TranspositionTableMCTS:
    cdef public unsigned long size
    cdef dict table
    cdef list visited
    cdef unsigned c_cache_hits
    cdef unsigned c_cache_misses
    cdef unsigned c_collisions
    cdef unsigned c_cleanups
    cdef unsigned cache_hits
    cdef unsigned cache_misses
    cdef unsigned collisions
    cdef unsigned cleanups
    cdef unsigned long num_entries

    cpdef public bint exists(self, long long  key)
    cpdef public (float, float, float, int, int, bint) get(self, long long key)
    cpdef public void put(self, long long key, float v1=*, float v2=*, int visits=*, int solved_player=*, bint is_expanded=*, float im_value=*)
    cpdef public void reset_metrics(self)
