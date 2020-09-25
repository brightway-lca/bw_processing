import numpy as np
import re

# Max signed 32 bit integer, compatible with Windows
MAX_SIGNED_32BIT_INT = 2147483647

# We could try to save space by not storing the columns
# `row_index` and `col_index`, and add them after loading from
# disk. This saves space, but is MUCH slower, as modifying
# a structured array requires a copy. So, for example, on
# EXIOBASE, it takes ~218 ms to load the technosphere,
# but -687ms to append two columns.
COMMON_DTYPE = [
    ("row_value", np.int32),
    ("col_value", np.int32),
    ("row_index", np.int32),
    ("col_index", np.int32),
    ("uncertainty_type", np.uint8),
    ("amount", np.float32),
    ("loc", np.float32),
    ("scale", np.float32),
    ("shape", np.float32),
    ("minimum", np.float32),
    ("maximum", np.float32),
    ("negative", np.bool),
    ("flip", np.bool),
]
INDICES_DTYPE = [
    ("row_value", np.int32),
    ("col_value", np.int32),
    ("row_index", np.int32),
    ("col_index", np.int32),
]

NAME_RE = re.compile(r"^[\w\-\.]*$")

DEFAULT_LICENSES = [
    {
        "name": "ODC-PDDL-1.0",
        "path": "http://opendatacommons.org/licenses/pddl/",
        "title": "Open Data Commons Public Domain Dedication and License v1.0",
    }
]
