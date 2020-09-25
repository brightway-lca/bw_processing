from .constants import MAX_SIGNED_32BIT_INT, NAME_RE
from .errors import InvalidName
from io import BytesIO
from pathlib import Path
import itertools
import numpy as np


def load_bytes(obj):
    if isinstance(obj, BytesIO):
        try:
            # Go to the beginning of content
            obj.seek(0)
            return np.load(obj, allow_pickle=False)
        except ValueError:
            pass
    return obj


def check_name(name):
    if name is not None and not NAME_RE.match(name):
        raise InvalidName(
            "Provided name violates datapackage spec (https://frictionlessdata.io/specs/data-package/)"
        )


def chunked(iterable, chunk_size):
    # Black magic, see https://stackoverflow.com/a/31185097
    # and https://docs.python.org/3/library/functions.html#iter
    iterable = iter(iterable)  # Fix e.g. range from restarting
    return iter(lambda: list(itertools.islice(iterable, chunk_size)), [])


def indices_wrapper(datasource):
    for row in datasource:
        yield (row["row"], row["col"], MAX_SIGNED_32BIT_INT, MAX_SIGNED_32BIT_INT)


def dictionary_wrapper(datasource):
    for row in datasource:
        yield dictionary_formatter(row)


def as_uncertainty_type(row):
    if "uncertainty_type" in row:
        return row["uncertainty_type"]
    elif "uncertainty type" in row:
        return row["uncertainty type"]
    else:
        return 0


def dictionary_formatter(row):
    """Format processed array row from dictionary input"""

    return (
        row["row"],
        # 1-d matrix
        row.get("col", row["row"]),
        MAX_SIGNED_32BIT_INT,
        MAX_SIGNED_32BIT_INT,
        as_uncertainty_type(row),
        row["amount"],
        row.get("loc", row["amount"]),
        row.get("scale", np.NaN),
        row.get("shape", np.NaN),
        row.get("minimum", np.NaN),
        row.get("maximum", np.NaN),
        row.get("negative", False),
        row.get("flip", False),
    )


def check_suffix(path, suffix):
    """Add ``suffix``, if not already in ``path``."""
    path = Path(path)
    if not suffix.startswith("."):
        suffix = "." + suffix
    if path.suffix != suffix:
        path = path.with_suffix(path.suffix + suffix)
    return path
