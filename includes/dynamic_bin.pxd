# cython: language_level=3

cdef class DynamicBin:
    cdef int num_bins
    cdef list data
    cdef double min_val, max_val
    cdef list bin_counts
    cdef list bin_edges
    cdef int zero_count

    cdef void calculate_bins(self)
    cdef list get_bins(self)
    cpdef public void add_data(self, double new_data)
    cpdef public void print_bins(self)
    cpdef public void plot_bin_counts(self, str name)
    cpdef public void clear(self)
    cpdef public void plot_time_series(self, str name, int plot_width=*, int plot_height=*, bint median=*)
