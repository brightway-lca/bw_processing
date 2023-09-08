# -*- coding: utf-8 -*-
"""
Unit tests for the `io_pyarrow_helper.py` module.
"""
import pytest

import numpy as np

from bw_processing.io_pyarrow_helpers import (
    pyarrow_generic_vector_table_to_numpy_generic_vector,
    pyarrow_distributions_vector_table_to_numpy_distributions_vector,
    pyarrow_generic_matrix_table_to_numpy_generic_matrix,
    pyarrow_indices_vector_table_to_numpy_indices_vector,
    numpy_indices_vector_to_pyarrow_indices_vector_table,
    numpy_generic_vector_to_pyarrow_generic_vector_table,
    numpy_distributions_vector_to_pyarrow_distributions_vector_table,
    numpy_generic_matrix_to_pyarrow_generic_matrix_table
)

from bw_processing.constants import INDICES_DTYPE, UNCERTAINTY_DTYPE

@pytest.fixture(scope="session")
def indices_vector():
    return np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)


def data_vector(dtype):
    return np.array([1, 2, 3], dtype=dtype)


def data_matrix(dtype):
    return np.arange(12, dtype=dtype).reshape((3, 4))


@pytest.fixture(scope="session")
def flip_vector():
    return np.array([True, False, False])


@pytest.fixture(scope="session")
def distribution_vector():
    return np.array([
        (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
        (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
        (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
        (5, 237, np.NaN, np.NaN, 200, 300, False),  # triangular uncertainty from 200 to 300
        (5, 2.5, np.NaN, np.NaN, 2, 3, False),  # triangular uncertainty from 2 to 3
    ],
        dtype=UNCERTAINTY_DTYPE
    )


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


def test_double_conversion_indices_vector(indices_vector):
    table = numpy_indices_vector_to_pyarrow_indices_vector_table(indices_vector)
    arr = pyarrow_indices_vector_table_to_numpy_indices_vector(table)

    assert arr.dtype == indices_vector.dtype
    assert np.array_equal(arr, indices_vector)


@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.float64])
def test_double_conversion_data_vector(dtype):
    data_vec = data_vector(dtype)
    table = numpy_generic_vector_to_pyarrow_generic_vector_table(data_vec)
    arr = pyarrow_generic_vector_table_to_numpy_generic_vector(table)

    assert arr.dtype == data_vec.dtype
    assert np.array_equal(arr, data_vec)


def test_double_conversion_flip_vector(flip_vector):
    table = numpy_generic_vector_to_pyarrow_generic_vector_table(flip_vector)
    arr = pyarrow_generic_vector_table_to_numpy_generic_vector(table)

    assert arr.dtype == flip_vector.dtype
    assert np.array_equal(arr, flip_vector)


def test_double_conversion_distribution_vector(distribution_vector):
    table = numpy_distributions_vector_to_pyarrow_distributions_vector_table(distribution_vector)
    arr = pyarrow_distributions_vector_table_to_numpy_distributions_vector(table)

    assert arr.dtype == distribution_vector.dtype
    assert vector_equal_with_uncertainty_dtype(arr, distribution_vector, equal_nan=True)


@pytest.mark.parametrize('dtype', [np.int8, np.int32, np.float64])
def test_double_conversion_data_matrix(dtype):
    data_mat = data_matrix(dtype)
    table = numpy_generic_matrix_to_pyarrow_generic_matrix_table(data_mat)
    arr = pyarrow_generic_matrix_table_to_numpy_generic_matrix(table)

    assert arr.dtype == data_mat.dtype
    assert np.array_equal(arr, data_mat)
