# -*- coding: utf-8 -*-
"""
This module contains some helpers to serialize/deserialize `numpy.ndarray` objects to/from Apache `parquet` files.
We convert the `nympy.ndarray` objects to `pyarrow.Table` objects to do so.
"""
import contextlib
import os

# for annotation
from io import BufferedWriter, IOBase, RawIOBase

import numpy
import numpy as np
import pyarrow.parquet as pq

from .errors import WrongDatatype
from .io_pyarrow_helpers import (
    numpy_distributions_vector_to_pyarrow_distributions_vector_table,
    numpy_generic_matrix_to_pyarrow_generic_matrix_table,
    numpy_generic_vector_to_pyarrow_generic_vector_table,
    numpy_indices_vector_to_pyarrow_indices_vector_table,
    pyarrow_distributions_vector_table_to_numpy_distributions_vector,
    pyarrow_generic_matrix_table_to_numpy_generic_matrix,
    pyarrow_generic_vector_table_to_numpy_generic_vector,
    pyarrow_indices_vector_table_to_numpy_indices_vector,
)


def write_ndarray_to_parquet_file(
    file: BufferedWriter, arr: np.ndarray, meta_object: str, meta_type: str
):
    """
    Serialize `ndarray` objects to `file`.

    Parameters
        file (io.BufferedWriter): File to save to.
        arr (ndarray): Array to serialize.
        meta_object (str): "vector" or "matrix".
        meta_type (str): Type of object to serialize (see `io_pyarrow_helpers.py`).

    """
    table = None
    if meta_object == "matrix":
        table = numpy_generic_matrix_to_pyarrow_generic_matrix_table(arr=arr)
    elif meta_object == "vector":
        if meta_type == "indices":
            table = numpy_indices_vector_to_pyarrow_indices_vector_table(arr=arr)
        elif meta_type == "generic":
            table = numpy_generic_vector_to_pyarrow_generic_vector_table(arr=arr)
        elif meta_type == "distributions":
            table = numpy_distributions_vector_to_pyarrow_distributions_vector_table(arr=arr)
        else:
            raise NotImplementedError(f"Vector of type {meta_type} is not recognized!")
    else:
        raise NotImplementedError(f"Object {meta_object} is not recognized!")

    # Save it:
    pq.write_table(table, file)


def read_parquet_file_to_ndarray(file: RawIOBase) -> numpy.ndarray:
    """
    Read an `ndarray` from a `parquet` file.

    Args:
        file (io.RawIOBase or fsspec file object): File to read from.

    Raises:
        `WrongDatatype` if the correct metadata is not found in the `parquet` file.

    Returns:
        The corresponding `numpy` `ndarray`.
    """
    table = pq.read_table(file)

    # reading metadata from parquet file
    try:
        binary_meta_object = table.schema.metadata[b"object"]
        binary_meta_type = table.schema.metadata[b"type"]
    except KeyError:
        raise WrongDatatype(f"Parquet file {file} does not contain the right metadata format!")

    arr = None
    if binary_meta_object == b"matrix":
        arr = pyarrow_generic_matrix_table_to_numpy_generic_matrix(table=table)
    elif binary_meta_object == b"vector":
        if binary_meta_type == b"indices":
            arr = pyarrow_indices_vector_table_to_numpy_indices_vector(table=table)
        elif binary_meta_type == b"generic":
            arr = pyarrow_generic_vector_table_to_numpy_generic_vector(table=table)
        elif binary_meta_type == b"distributions":
            arr = pyarrow_distributions_vector_table_to_numpy_distributions_vector(table=table)
        else:
            raise NotImplementedError("Vector type not recognized")
    else:
        raise NotImplementedError("Metadata object not recognized")

    return arr


def save_arr_to_parquet(file: RawIOBase, arr: np.ndarray, meta_object: str, meta_type: str) -> None:
    """
    Serialize a `numpy` `ndarray` to a `parquet` `file`.

    Parameters
        file (RawIOBase): The file to save to.
        arr (ndarray): The array object to save.
        meta_object (str): "vector" or "matrix".
        meta_type (str): Type of object to serialize (see `io_pyarrow_helpers.py`).
    """
    if hasattr(file, "write"):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith(".parquet"):
            file = file + ".parquet"
        file_ctx = open(file, "wb")

    with file_ctx as fid:
        arr = np.asanyarray(arr)
        write_ndarray_to_parquet_file(fid, arr, meta_object=meta_object, meta_type=meta_type)


def load_ndarray_from_parquet(file: RawIOBase) -> np.ndarray:
    """
    Deserialize a `numpy` `ndarray` from a `parquet` `file`.

    Parameters
        file (io.RawIOBase or fsspec file object): File to read from.

    Returns
        The corresponding `numpy` `ndarray`.
    """
    if hasattr(file, "read"):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        file_ctx = open(file, "rb")

    with file_ctx as fid:
        arr = read_parquet_file_to_ndarray(fid)

    return arr
