#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True

from collections import defaultdict, OrderedDict

cdef class MoveHistory:
    cdef object table
    cpdef int get(self, tuple move)
    cpdef void update(self, tuple move, int increment)

cdef class TranspositionTable:
    cdef public long size
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


cdef class TranspositionTableMCTS:
    cdef public unsigned long size
    cdef double[:,:] table
    cdef unsigned long puts, gets, uniques

    cpdef public double[:] get(self, long long key)
    cpdef public void put(self, long long key, double v1=*, double v2=*, double visits=*, double solved_player=*, double is_expanded=*, double eval_value=*)
    cpdef public void reset_metrics(self)
    cpdef public dict get_metrics(self)
