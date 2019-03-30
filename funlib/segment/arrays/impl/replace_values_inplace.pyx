from cython.operator cimport dereference as deref
from libcpp.map cimport map
import logging
import numpy as np
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_values_inplace(
        np.ndarray[np.uint64_t, ndim=1, mode='c'] array,
        np.ndarray[np.uint64_t, ndim=1, mode='c'] old_values,
        np.ndarray[np.uint64_t, ndim=1, mode='c'] new_values):

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = array.size
    cdef np.npy_uint64* a = &array[0]

    cdef map[np.npy_uint64, np.npy_uint64] cmap
    for i in range(old_values.size):
        cmap[old_values[i]] = new_values[i]

    for i in range(n):
        it = cmap.find(a[i])
        if it == cmap.end():
            continue
        a[i] = deref(it).second


@cython.boundscheck(False)
@cython.wraparound(False)
def replace_values_inplace_using_mask(
        np.ndarray[np.uint64_t, ndim=1, mode='c'] mask_array,
        np.ndarray[np.uint64_t, ndim=1, mode='c'] mask_values,
        np.ndarray[np.uint64_t, ndim=1, mode='c'] new_values,
        np.ndarray[np.uint64_t, ndim=1, mode='c'] out_array):

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = mask_array.size
    cdef np.npy_uint64* mask = &mask_array[0]
    cdef np.npy_uint64* out = &out_array[0]

    cdef map[np.npy_uint64, np.npy_uint64] cmap
    for i in range(mask_values.size):
        cmap[mask_values[i]] = new_values[i]

    for i in range(n):
        it = cmap.find(mask[i])
        if it == cmap.end():
            continue
        out[i] = deref(it).second
