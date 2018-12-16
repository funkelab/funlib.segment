from __future__ import absolute_import
from .impl import replace_values_inplace
import numpy as np

def replace_values(array, old_values, new_values, inplace=False):
    '''Replace each ``old_values`` in ``array`` with the corresponding
    ``new_values``. Other values are not changed.
    '''

    assert old_values.size == new_values.size
    assert array.dtype == old_values.dtype
    assert array.dtype == new_values.dtype

    dtype = array.dtype

    old_values = np.array(old_values)
    new_values = np.array(new_values)

    min_value = array.min()
    max_value = array.max()
    value_range = max_value - min_value

    # can the relaballing be done with a values map?
    if value_range < 1024**3:

        valid_values = np.logical_and(
            old_values>=min_value,
            old_values<=max_value)
        old_values = old_values[valid_values]
        new_values = new_values[valid_values]

        # shift all values such that they start at 0
        offset = min_value
        array -= offset
        old_values -= offset
        new_values -= offset
        min_value -= offset
        max_value -= offset

        # replace with a values map
        values_map = np.arange(max_value + 1, dtype=dtype)
        values_map[old_values] = new_values

        if inplace:

            array[:] = values_map[array] + offset
            return array

        else:

            replaced = values_map[array] + offset
            array += offset
            return replaced

    else:

        # create a sparse values map
        values_map = {
            old_value: new_value
            for old_value, new_value in zip(old_values, new_values)
        }

        # replace using C++ implementation
        if not inplace:
            array = array.copy()

        replace_values_inplace(array, values_map)
        return array
