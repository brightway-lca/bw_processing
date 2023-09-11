# -*- coding: utf-8 -*-
"""
Some helpers to deal with basic arrays.

Some functions could become fixtures but at the same time it might not be the hassle
to convert them as fixtures.
"""
import numpy as np

from bw_processing.constants import UNCERTAINTY_DTYPE


def data_vector(dtype):
    return np.array([1, 2, 3], dtype=dtype)


def data_matrix(dtype):
    return np.arange(12, dtype=dtype).reshape((3, 4))


def vector_equal_with_uncertainty_dtype(A, B, equal_nan=True):
    if A.dtype != UNCERTAINTY_DTYPE or B.dtype != UNCERTAINTY_DTYPE:
        return False
    if A.shape != B.shape:
        return False
    for e, l in zip(A, B):
        for i, j in zip(e, l):
            if equal_nan:
                if np.isnan(i) and np.isnan(j):
                    continue
            if i != j:
                return False
    return True
