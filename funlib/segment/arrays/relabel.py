from __future__ import absolute_import
from .replace_values import replace_values
import numpy as np


def relabel(array, return_backwards_map=False, inplace=False):
    '''Relabel array, such that IDs are consecutive. Excludes 0.'''

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:

        if return_backwards_map:
            return array, 1, [0]
        else:
            return array, 1

    n = len(old_labels) + 1
    new_labels = np.arange(1, n, dtype=array.dtype)

    replaced = replace_values(array, old_labels, new_labels, inplace=inplace)

    if return_backwards_map:

        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n
