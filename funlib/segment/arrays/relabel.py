from __future__ import absolute_import
from .replace_values import replace_values
import numpy as np


def relabel(array, return_backwards_map=False, inplace=False):
    """Relabel array, such that IDs are consecutive. Excludes 0.

    Args:

        array (ndarray):

                The array to relabel.

        return_backwards_map (``bool``, optional):

                If ``True``, return an ndarray that maps new labels (indices in
                the array) to old labels.

        inplace (``bool``, optional):

                Perform the replacement in-place on ``array``.

    Returns:

        A tuple ``(relabelled, n)``, where ``relabelled`` is the relabelled
        array and ``n`` the number of unique labels found.

        If ``return_backwards_map`` is ``True``, returns ``(relabelled, n,
        backwards_map)``.
    """

    if array.size == 0:
        if return_backwards_map:
            return array, 0, []
        else:
            return array, 0

    # get all labels except 0
    old_labels = np.unique(array)
    old_labels = old_labels[old_labels != 0]

    if old_labels.size == 0:
        if return_backwards_map:
            return array, 0, [0]
        else:
            return array, 0

    n = len(old_labels)
    new_labels = np.arange(1, n + 1, dtype=array.dtype)

    replaced = replace_values(array, old_labels, new_labels, inplace=inplace)

    if return_backwards_map:
        backwards_map = np.insert(old_labels, 0, 0)
        return replaced, n, backwards_map

    return replaced, n
