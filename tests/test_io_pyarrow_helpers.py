# -*- coding: utf-8 -*-
"""
Unit tests for the `io_pyarrow_helper.py` module.
"""
import numpy as np
import pytest
from helpers.basic_array_helpers import (
    data_matrix,
    data_vector,
    vector_equal_with_uncertainty_dtype,
)

from bw_processing.io_pyarrow_helpers import (
    numpy_distributions_vector_to_pyarrow_distributions_vector_table,
    numpy_generic_matrix_to_pyarrow_generic_matrix_table,
    numpy_generic_vector_to_pyarrow_generic_vector_table,
    numpy_indices_vector_to_pyarrow_indices_vector_table,
    pyarrow_distributions_vector_table_to_numpy_distributions_vector,
    pyarrow_generic_matrix_table_to_numpy_generic_matrix,
    pyarrow_generic_vector_table_to_numpy_generic_vector,
    pyarrow_indices_vector_table_to_numpy_indices_vector,
)


def test_double_conversion_indices_vector(indices_vector):
    table = numpy_indices_vector_to_pyarrow_indices_vector_table(indices_vector)
    arr = pyarrow_indices_vector_table_to_numpy_indices_vector(table)

    assert arr.dtype == indices_vector.dtype
    assert np.array_equal(arr, indices_vector)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float64])
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


def test_double_conversion_distribution_vector(distributions_vector):
    table = numpy_distributions_vector_to_pyarrow_distributions_vector_table(distributions_vector)
    arr = pyarrow_distributions_vector_table_to_numpy_distributions_vector(table)

    assert arr.dtype == distributions_vector.dtype
    assert vector_equal_with_uncertainty_dtype(arr, distributions_vector, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float64])
def test_double_conversion_data_matrix(dtype):
    data_mat = data_matrix(dtype)
    table = numpy_generic_matrix_to_pyarrow_generic_matrix_table(data_mat)
    arr = pyarrow_generic_matrix_table_to_numpy_generic_matrix(table)

    assert arr.dtype == data_mat.dtype
    assert np.array_equal(arr, data_mat)
