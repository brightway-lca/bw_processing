# -*- coding: utf-8 -*-
"""
Unit tests for saving and loading to/from parquet files but this time with options.
"""
import numpy as np
import pyarrow.parquet as pq
import pytest
from helpers.basic_array_helpers import (
    data_matrix,
    data_vector,
    vector_equal_with_uncertainty_dtype,
)

from bw_processing.errors import WrongDatatype
from bw_processing.io_parquet_helpers import (
    load_ndarray_from_parquet,
    save_arr_to_parquet,
)

ARR_LIST = [
    ("indices_vector", "vector", "indices"),
    ("flip_vector", "vector", "generic"),
]

PARQUET_OPTIONS = {
    "compression": "gzip",
    "compression_level": 4
}

@pytest.mark.parametrize("arr_fixture_name, meta_object, meta_type", ARR_LIST)
def test_save_load_parquet_file(
    arr_fixture_name, meta_object, meta_type, tmp_path_factory, request
):

    arr = request.getfixturevalue(arr_fixture_name)  # get fixture from name
    file = tmp_path_factory.mktemp("data") / (arr_fixture_name + ".parquet")

    with file as fp:
        save_arr_to_parquet(
            file=fp, arr=arr, meta_object=meta_object, meta_type=meta_type, parquet_options=PARQUET_OPTIONS
        )

    with file as fp:
        loaded_arr = load_ndarray_from_parquet(fp)

    assert arr.dtype == loaded_arr.dtype and np.array_equal(arr, loaded_arr)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float64])
def test_save_load_parquet_file_data_vector(dtype, tmp_path_factory):

    arr = data_vector(dtype=dtype)
    file = tmp_path_factory.mktemp("data") / "data_vector.parquet"

    with file as fp:
        save_arr_to_parquet(file=fp, arr=arr, meta_object="vector", meta_type="generic", parquet_options=PARQUET_OPTIONS)

    with file as fp:
        loaded_arr = load_ndarray_from_parquet(fp)

    assert arr.dtype == loaded_arr.dtype and np.array_equal(arr, loaded_arr)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float64])
def test_save_load_parquet_file_data_matrix(dtype, tmp_path_factory):

    arr = data_matrix(dtype=dtype)
    file = tmp_path_factory.mktemp("data") / "data_matrix.parquet"

    with file as fp:
        save_arr_to_parquet(file=fp, arr=arr, meta_object="matrix", meta_type="generic", parquet_options=PARQUET_OPTIONS)

    with file as fp:
        loaded_arr = load_ndarray_from_parquet(fp)

    assert arr.dtype == loaded_arr.dtype and np.array_equal(arr, loaded_arr)


def test_save_load_parquet_file_distribution_vector(
    distributions_vector, tmp_path_factory
):

    arr = distributions_vector
    file = tmp_path_factory.mktemp("data") / "distributions_vector.parquet"

    with file as fp:
        save_arr_to_parquet(
            file=fp, arr=arr, meta_object="vector", meta_type="distributions", parquet_options=PARQUET_OPTIONS
        )

    with file as fp:
        loaded_arr = load_ndarray_from_parquet(fp)

    assert vector_equal_with_uncertainty_dtype(arr, loaded_arr)


def test_save_load_parquet_file_wrong_meta_object(indices_vector, tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "indices_vector.parquet"

    with pytest.raises(NotImplementedError):
        with file as fp:
            save_arr_to_parquet(
                file=fp, arr=indices_vector, meta_object="wrong", meta_type="indices", parquet_options=PARQUET_OPTIONS
            )

    with pytest.raises(NotImplementedError):
        with file as fp:
            save_arr_to_parquet(
                file=fp, arr=indices_vector, meta_object="vector", meta_type="indices", parquet_options=PARQUET_OPTIONS
            )

        with file as fp:
            table = pq.read_table(fp)
            metadata = {"object": "wrong", "type": "indices"}
            new_table = table.replace_schema_metadata(metadata=metadata)
            pq.write_table(new_table, fp)

        with file as fp:
            load_ndarray_from_parquet(fp)


def test_save_load_parquet_file_wrong_meta_type(indices_vector, tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "indices_vector.parquet"

    with pytest.raises(NotImplementedError):
        with file as fp:
            save_arr_to_parquet(
                file=fp, arr=indices_vector, meta_object="vector", meta_type="wrong", parquet_options=PARQUET_OPTIONS
            )

    with pytest.raises(NotImplementedError):
        with file as fp:
            save_arr_to_parquet(
                file=fp, arr=indices_vector, meta_object="vector", meta_type="indices", parquet_options=PARQUET_OPTIONS
            )

        with file as fp:
            table = pq.read_table(fp)
            metadata = {"object": "vector", "type": "wrong"}
            new_table = table.replace_schema_metadata(metadata=metadata)
            pq.write_table(new_table, fp)

        with file as fp:
            load_ndarray_from_parquet(fp)


def test_save_load_parquet_file_wrong_metadata_format(indices_vector, tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "indices_vector.parquet"

    with pytest.raises(WrongDatatype):
        with file as fp:
            save_arr_to_parquet(
                file=fp, arr=indices_vector, meta_object="vector", meta_type="indices", parquet_options=PARQUET_OPTIONS
            )

        with file as fp:
            table = pq.read_table(fp)
            metadata = {}
            new_table = table.replace_schema_metadata(metadata=metadata)
            pq.write_table(new_table, fp)

        with file as fp:
            load_ndarray_from_parquet(fp)