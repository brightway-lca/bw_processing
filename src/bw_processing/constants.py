import re
from enum import Enum

import numpy as np

MAX_SIGNED_32BIT_INT = 2147483647
MAX_SIGNED_64BIT_INT = 9223372036854775807

# We could try to save space by not storing the columns
# `row_index` and `col_index`, and add them after loading from
# disk. This saves space, but is MUCH slower, as modifying
# a structured array requires a copy. So, for example, on
# EXIOBASE, it takes ~218 ms to load the technosphere,
# but -687ms to append two columns.
UNCERTAINTY_DTYPE = [
    ("uncertainty_type", np.uint8),
    ("loc", np.float32),
    ("scale", np.float32),
    ("shape", np.float32),
    ("minimum", np.float32),
    ("maximum", np.float32),
    ("negative", bool),
]
INDICES_DTYPE = [("row", np.int64), ("col", np.int64)]

NAME_RE = re.compile(r"^[\w\-\.]*$")

DEFAULT_LICENSES = [
    {
        "name": "ODC-PDDL-1.0",
        "path": "http://opendatacommons.org/licenses/pddl/",
        "title": "Open Data Commons Public Domain Dedication and License v1.0",
    }
]


class MatrixSerializeFormat(str, Enum):
    """
    Enum with the serializing formats for the vectors and matrices.
    """

    NUMPY = "numpy"  # numpy .npy format
    PARQUET = "parquet"  # Apache .parquet format


# FILE EXTENSIONS
NUMPY_SERIALIZE_FORMAT_EXTENSION = ".npy"
NUMPY_SERIALIZE_FORMAT_NAME = "npy"
PARQUET_SERIALIZE_FORMAT_EXTENSION = ".parquet"
PARQUET_SERIALIZE_FORMAT_NAME = "pqt"
