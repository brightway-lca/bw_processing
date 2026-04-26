# -*- coding: utf-8 -*-
"""
This module contains some helpers to convert `nympy.ndarrays` to/from Apache `Arrow` `Table`.

We use `pyarrow.Table` objects to save/retrieve data into/from `parquet` format files.
We use a `metadata` section in the `pyarrow.Table` (and the `parquet` files) to be able
to recognize what type of data was serialized. Specific and generic codes exist.

The metadata object is a `dict` object that looks like this:

`{"object": "vector", "type": "generic"}`. `object` can be `vector` (`ndim == 1`) or `matrix` (`ndim == 2`),
`type` can be:

- `indices` (`dtype` is `INDICES_DTYPE`);
- `distributions` (`dtype` is `UNCERTAINTY_DTYPE`);
- `generic` (`dtype` is a common type);

"""
import numpy as np
import pyarrow as pa

from bw_processing import INDICES_DTYPE, UNCERTAINTY_DTYPE


###########
# VECTORS #
###########
def numpy_generic_vector_to_pyarrow_generic_vector_table(arr: np.ndarray) -> pa.Table:
    """
    Convert a generic (numpy) vector to a (arrow) table.

    Args:
        arr (ndarray): A numpy array that corresponds to a vector, i.e. its dimension is 1.

    See:
        `pyarrow_generic_vector_table_to_numpy_generic_vector`.

    Returns:
        The corresponding `pyarrow.Table` object.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1

    generic_schema = pa.schema(
        [pa.field(str(0), pa.from_numpy_dtype(arr.dtype))],
        metadata={"object": "vector", "type": "generic"},
    )
    table = pa.table({str(0): arr}, schema=generic_schema)

    return table


def pyarrow_generic_vector_table_to_numpy_generic_vector(table: pa.Table) -> np.ndarray:
    """
    Convert a generic (arrow) vector table to a (numpy) array.

    Args:
        table (pa.Table): A `pyarrow` table that corresponds to a vector.

    See:
        `numpy_generic_vector_to_pyarrow_generic_vector_table`.

    Returns:
        The corresponding `np.ndarray` object.
    """
    assert isinstance(table, pa.Table)
    assert len(table.schema) == 1
    assert table.schema.metadata[b"object"] == b"vector"
    assert table.schema.metadata[b"type"] == b"generic"

    # find numpy dtype
    pa_field = table.schema[0]  # pa.Field(), only one common type for all elements
    numpy_dtype = pa_field.type.to_pandas_dtype()  # numpy type

    arr = np.array(table[str(0)], dtype=numpy_dtype)  # col name is "0"

    return arr


# specific arrow schema for indices vectors
INDICES_SCHEMA = pa.schema(
    [
        pa.field(INDICES_DTYPE[0][0], pa.from_numpy_dtype(INDICES_DTYPE[0][1])),
        pa.field(INDICES_DTYPE[1][0], pa.from_numpy_dtype(INDICES_DTYPE[1][1])),
    ],
    metadata={"object": "vector", "type": "indices"},
)


def numpy_indices_vector_to_pyarrow_indices_vector_table(arr: np.ndarray) -> pa.Table:
    """
    Convert a specific indices (numpy) vector to a (arrow) table.

    Args:
        arr (ndarray): A numpy array that corresponds to an indices vector, i.e. its dimension is 1 and its
            `dtype` is `INDICES_DTYPE`.

    See:
        `pyarrow_indices_vector_table_to_numpy_indices_vector`.

    Returns:
        The corresponding `pyarrow.Table` object.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == INDICES_DTYPE

    row_indices = []
    col_indices = []
    for row, col in arr:
        row_indices.append(row)
        col_indices.append(col)

    table = pa.table(
        {
            INDICES_DTYPE[0][0]: row_indices,  # col name is "row"
            INDICES_DTYPE[1][0]: col_indices,
        },  # col name is "col"
        schema=INDICES_SCHEMA,
    )

    return table


def pyarrow_indices_vector_table_to_numpy_indices_vector(table: pa.Table) -> np.ndarray:
    """
    Convert a specific indices (arrow) vector table to a (numpy) array.

    Args:
        table (pa.Table): A `pyarrow` table that corresponds to an indices vector.

    See:
        `numpy_indices_vector_to_pyarrow_indices_vector_table`.

    Returns:
        The corresponding `np.ndarray` object.
    """
    assert isinstance(table, pa.Table)
    assert len(table.schema) == 2
    assert table.schema.metadata[b"object"] == b"vector"
    assert table.schema.metadata[b"type"] == b"indices"

    indices_array = []
    for row, col in zip(table["row"], table["col"]):
        indices_array.append((row.as_py(), col.as_py()))

    arr = np.array(indices_array, dtype=INDICES_DTYPE)

    return arr


# create UNCERTAINTY schema
NBR_UNCERTAINTY_FIELDS = len(UNCERTAINTY_DTYPE)
PA_UNCERTAINTY_FIELDS = [
    pa.field(UNCERTAINTY_DTYPE[i][0], pa.from_numpy_dtype(UNCERTAINTY_DTYPE[i][1]))
    for i in range(NBR_UNCERTAINTY_FIELDS)
]

UNCERTAINTY_SCHEMA = pa.schema(
    PA_UNCERTAINTY_FIELDS, metadata={"object": "vector", "type": "distributions"}
)

UNCERTAINTY_FIELDS_NAMES = [UNCERTAINTY_DTYPE[i][0] for i in range(NBR_UNCERTAINTY_FIELDS)]


def numpy_distributions_vector_to_pyarrow_distributions_vector_table(
    arr: np.ndarray,
) -> pa.Table:
    """
    Convert a specific distributions (numpy) vector to a (arrow) table.

    Args:
        arr (np.ndarray): A numpy array that corresponds to a distributions vector, i.e. its dimension is 1 and its
            `dtype` is `UNCERTAINTY_DTYPE`.

    See:
        `pyarrow_distributions_vector_table_to_numpy_distributions_vector`

    Returns:
        The corresponding `pyarrow.Table` object.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == UNCERTAINTY_DTYPE

    from_dict = {}
    for i in range(NBR_UNCERTAINTY_FIELDS):
        from_dict[UNCERTAINTY_FIELDS_NAMES[i]] = []

    for arr_el in arr:
        for i, el in enumerate(arr_el):
            from_dict[UNCERTAINTY_FIELDS_NAMES[i]].append(el)

    table = pa.table(from_dict, schema=UNCERTAINTY_SCHEMA)

    return table


def pyarrow_distributions_vector_table_to_numpy_distributions_vector(
    table: pa.Table,
) -> np.ndarray:
    """
    Convert a specific distributions (arrow) vector table to a (numpy) array.

    Args:
        table (pa.Table): A `pyarrow` table that corresponds to a distributions vector.

    See:
        `numpy_distributions_vector_to_pyarrow_distributions_vector_table`.

    Returns:
        The corresponding `np.ndarray` object.
    """
    assert isinstance(table, pa.Table)
    assert len(table.schema) == NBR_UNCERTAINTY_FIELDS
    assert table.schema.metadata[b"object"] == b"vector"
    assert table.schema.metadata[b"type"] == b"distributions"

    distributions_arrays_list = [
        table[UNCERTAINTY_FIELDS_NAMES[i]] for i in range(NBR_UNCERTAINTY_FIELDS)
    ]

    distributions_array = []
    for el in zip(*distributions_arrays_list):
        distributions_array.append(tuple(el[i].as_py() for i in range(NBR_UNCERTAINTY_FIELDS)))
    arr = np.array(distributions_array, dtype=UNCERTAINTY_DTYPE)

    return arr


############
# MATRICES #
############
def numpy_generic_matrix_to_pyarrow_generic_matrix_table(arr: np.ndarray) -> pa.Table:
    """
    Convert a generic (numpy) matrix to a (arrow) table.

    Args:
        arr (ndarray): A numpy array that corresponds to a generic matrix, i.e. its dimension is 2.

    See:
        `pyarrow_generic_matrix_table_to_numpy_generic_matrix`.

    Returns:
        The corresponding `pyarrow.Table` object.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2

    arr_dtype = arr.dtype
    metadata = {"object": "matrix", "type": "generic"}
    nbr_rows, nbr_cols = arr.shape
    arrays = [pa.array(arr[:, j], type=pa.from_numpy_dtype(arr_dtype)) for j in range(nbr_cols)]
    table = pa.Table.from_arrays(
        arrays=arrays,
        names=[str(j) for j in range(nbr_cols)],  # give names to each column
        metadata=metadata,
    )

    return table


def pyarrow_generic_matrix_table_to_numpy_generic_matrix(table: pa.Table) -> np.ndarray:
    """
    Convert a generic (arrow) matrix table to a (numpy) array.

    Args:
        table (pa.Table): A `pyarrow` table that corresponds to a generic matrix.

    See:
        `numpy_generic_matrix_to_pyarrow_generic_matrix_table`.

    Returns:
        The corresponding `np.ndarray` object.
    """
    assert isinstance(table, pa.Table)
    assert table.schema.metadata[b"object"] == b"matrix"
    assert table.schema.metadata[b"type"] == b"generic"

    arr = table.to_pandas().to_numpy()

    return arr
