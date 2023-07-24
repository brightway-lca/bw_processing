from io import BytesIO
from pathlib import Path
from typing import Any, Union
import importlib.metadata

import numpy as np
from numpy.lib.recfunctions import repack_fields

from .array_creation import create_structured_array
from .constants import INDICES_DTYPE, NAME_RE, UNCERTAINTY_DTYPE
from .errors import InvalidName


def get_version_tuple() -> tuple:
    """Returns version as (major, minor, micro)."""

    def as_integer(version_str: str) -> Union[int, str]:
        try:
            return int(version_str)
        except ValueError:
            return version_str

    return tuple(
        as_integer(v)
        for v in importlib.metadata.version("bw_processing")
        .strip()
        .split(".")
    )


def load_bytes(obj: Any) -> Any:
    if isinstance(obj, BytesIO):
        try:
            # Go to the beginning of content
            obj.seek(0)
            return np.load(obj, allow_pickle=False)
        except ValueError:
            pass
    return obj


def check_name(name: str) -> None:
    if name is not None and not NAME_RE.match(name):
        raise InvalidName(
            "Provided name violates datapackage spec (https://frictionlessdata.io/specs/data-package/)"
        )


def check_suffix(path: Union[str, Path], suffix=str) -> str:
    """Add ``suffix``, if not already in ``path``."""
    path = Path(path)
    if not suffix.startswith("."):
        suffix = "." + suffix
    if path.suffix != suffix:
        path = path.with_suffix(path.suffix + suffix)
    return str(path)


def as_uncertainty_type(row: dict) -> int:
    if "uncertainty_type" in row:
        return row["uncertainty_type"]
    elif "uncertainty type" in row:
        return row["uncertainty type"]
    else:
        return 0


def dictionary_formatter(row: dict) -> tuple:
    """Format processed array row from dictionary input"""

    return (
        row["row"],
        # 1-d matrix
        row.get("col", row["row"]),
        row["amount"],
        as_uncertainty_type(row),
        row.get("loc", row["amount"]),
        row.get("scale", np.NaN),
        row.get("shape", np.NaN),
        row.get("minimum", np.NaN),
        row.get("maximum", np.NaN),
        row.get("negative", False),
        row.get("flip", False),
    )


def resolve_dict_iterator(iterator: Any, nrows: int = None) -> tuple:
    """Note that this function produces sorted arrays."""
    sort_fields = ["row", "col", "amount", "uncertainty_type"]
    data = (dictionary_formatter(row) for row in iterator)
    array = create_structured_array(
        data,
        INDICES_DTYPE + [("amount", np.float32)] + UNCERTAINTY_DTYPE + [("flip", bool)],
        nrows=nrows,
        sort=True,
        sort_fields=sort_fields,
    )
    return (
        array["amount"],
        # Not repacking fields would cause this multi-field index to return a view
        # All columns would be serialized
        # See https://numpy.org/doc/stable/user/basics.rec.html#indexing-structured-arrays
        repack_fields(array[["row", "col"]]),
        repack_fields(
            array[
                [
                    "uncertainty_type",
                    "loc",
                    "scale",
                    "shape",
                    "minimum",
                    "maximum",
                    "negative",
                ]
            ]
        ),
        array["flip"],
    )
