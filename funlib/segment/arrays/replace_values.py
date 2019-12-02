from __future__ import absolute_import
from .impl import replace_values_inplace
import numpy as np


def replace_values(
        in_array, old_values, new_values, out_array=None, inplace=None):
    '''Replace each ``old_values`` in ``array`` with the corresponding
    ``new_values``. Other values are not changed.
    '''

    # `inplace` and `out_array` cannot be both specified
    if out_array is not None:
        assert (inplace is None) or (inplace is False)
    if inplace:
        out_array = in_array

    # `in_array` should always be a numpy array
    assert isinstance(in_array, (np.ndarray))

    # `old_values` and `new_values` are converted to numpy lists
    # if they are not already
    if not isinstance(old_values, (np.ndarray, np.generic)):
        old_values = np.array(old_values, dtype=in_array.dtype)
    if not isinstance(new_values, (np.ndarray, np.generic)):
        if out_array is None:
            # if `out_array` is None, its data type is guaranteed to be
            # same as `out_array`
            new_values = np.array(new_values, dtype=in_array.dtype)
        else:
            new_values = np.array(new_values, dtype=out_array.dtype)

    assert old_values.size == new_values.size
    assert in_array.dtype == old_values.dtype
    if out_array is not None:
        assert out_array.dtype == new_values.dtype
        assert out_array.size == in_array.size

    dtype = in_array.dtype

    min_value = in_array.min()
    max_value = in_array.max()
    value_range = max_value - min_value

    # can the relabeling be done with a values map?
    # this can only be done if `out_array` is not provided and when
    # `out_array` is provided and it _is_ `in_array`
    if (out_array is None or out_array is in_array) and value_range < 1024**3:

        valid_values = np.logical_and(
            old_values >= min_value,
            old_values <= max_value)
        old_values = old_values[valid_values]
        new_values = new_values[valid_values]

        # shift old values and in_array such that they start at 0
        offset = min_value
        in_array -= offset
        old_values -= offset
        min_value -= offset
        max_value -= offset

        # replace with a values map
        values_map = np.arange(max_value + 1, dtype=dtype)
        values_map[old_values] = new_values

        inplace = out_array is in_array

        if inplace:

            in_array[:] = values_map[in_array]
            return out_array

        else:

            out_array = values_map[in_array]
            # shift back in_array
            in_array += offset
            return out_array

    else:

        # replace using C++ implementation

        if out_array is None:
            out_array = in_array.copy()

        replace_values_inplace(
            np.ravel(in_array, order='A'),
            np.ravel(old_values, order='A'),
            np.ravel(new_values, order='A'),
            np.ravel(out_array, order='A'))

        return out_array
