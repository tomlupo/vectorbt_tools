# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for grouping."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []

GroupByT = tp.Union[None, bool, tp.Index]


@register_jitted(cache=True)
def get_group_lens_nb(groups: tp.Array1d) -> tp.GroupLens:
    """Return the count per group.

    !!! note
        Columns must form monolithic, sorted groups. For unsorted groups, use `get_group_map_nb`."""
    result = np.empty(groups.shape[0], dtype=np.int_)
    j = 0
    last_group = -1
    group_len = 0
    for i in range(groups.shape[0]):
        cur_group = groups[i]
        if cur_group < last_group:
            raise ValueError("Columns must form monolithic, sorted groups")
        if cur_group != last_group:
            if last_group != -1:
                # Process previous group
                result[j] = group_len
                j += 1
                group_len = 0
            last_group = cur_group
        group_len += 1
        if i == groups.shape[0] - 1:
            # Process last group
            result[j] = group_len
            j += 1
            group_len = 0
    return result[:j]


@register_jitted(cache=True)
def get_group_map_nb(groups: tp.Array1d, n_groups: int) -> tp.GroupMap:
    """Build the map between groups and indices.

    Returns an array with indices segmented by group and an array with group lengths.

    Works well for unsorted group arrays."""
    group_lens_out = np.full(n_groups, 0, dtype=np.int_)
    for g in range(groups.shape[0]):
        group = groups[g]
        group_lens_out[group] += 1

    group_start_idxs = np.cumsum(group_lens_out) - group_lens_out
    group_idxs_out = np.empty((groups.shape[0],), dtype=np.int_)
    group_i = np.full(n_groups, 0, dtype=np.int_)
    for g in range(groups.shape[0]):
        group = groups[g]
        group_idxs_out[group_start_idxs[group] + group_i[group]] = g
        group_i[group] += 1

    return group_idxs_out, group_lens_out


@register_jitted(cache=True)
def group_lens_select_nb(group_lens: tp.GroupLens, new_groups: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Perform indexing on a sorted array using group lengths.

    Returns indices of elements corresponding to groups in `new_groups` and a new group array."""
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    n_values = np.sum(group_lens[new_groups])
    indices_out = np.empty(n_values, dtype=np.int_)
    group_arr_out = np.empty(n_values, dtype=np.int_)
    j = 0

    for c in range(new_groups.shape[0]):
        from_r = group_start_idxs[new_groups[c]]
        to_r = group_end_idxs[new_groups[c]]
        if from_r == to_r:
            continue
        rang = np.arange(from_r, to_r)
        indices_out[j : j + rang.shape[0]] = rang
        group_arr_out[j : j + rang.shape[0]] = c
        j += rang.shape[0]
    return indices_out, group_arr_out


@register_jitted(cache=True)
def group_map_select_nb(group_map: tp.GroupMap, new_groups: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Perform indexing using group map."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    total_count = np.sum(group_lens[new_groups])
    indices_out = np.empty(total_count, dtype=np.int_)
    group_arr_out = np.empty(total_count, dtype=np.int_)
    j = 0

    for new_group_i in range(len(new_groups)):
        new_group = new_groups[new_group_i]
        group_len = group_lens[new_group]
        if group_len == 0:
            continue
        group_start_idx = group_start_idxs[new_group]
        idxs = group_idxs[group_start_idx : group_start_idx + group_len]
        indices_out[j : j + group_len] = idxs
        group_arr_out[j : j + group_len] = new_group_i
        j += group_len
    return indices_out, group_arr_out


@register_jitted(cache=True)
def group_by_evenly_nb(n: int, n_splits: int) -> tp.Array1d:
    """Get `group_by` from evenly splitting a space of values."""
    out = np.empty(n, dtype=np.int_)
    for i in range(n):
        out[i] = i * n_splits // n + n_splits // (2 * n)
    return out
