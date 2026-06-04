import dataclasses
import math
from enum import Enum
from typing import Optional, Union

import numpy as np

try:
    from stats_arrays import NoUncertainty, UndefinedUncertainty
    _NO_UNCERTAINTY_IDS = (UndefinedUncertainty.id, NoUncertainty.id)
except ImportError:
    _NO_UNCERTAINTY_IDS = (0, 1)


class MatrixName(str, Enum):
    """Standard matrix names used in Brightway.

    Because this is a ``str`` enum, members can be used anywhere a plain
    string is accepted — no ``.value`` needed::

        MatrixEntry(row=1, col=4, amount=2.5)  # inside a dict keyed by MatrixName
        dp.add_entries(matrix=MatrixName.technosphere, entries=[...])

    Derived libraries may define additional matrices as plain strings;
    these three cover the core Brightway LCA workflow.
    """

    technosphere = "technosphere_matrix"
    biosphere = "biosphere_matrix"
    characterization = "characterization_matrix"

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass(frozen=True)
class MatrixEntry:
    """A single entry destined for a matrix cell.

    Multiple instances with the same (row, col) are summed during matrix
    construction, so this is not necessarily the final cell value.

    Field names and defaults match those expected by bw_processing's
    ``dictionary_formatter``. Convert to a plain dict with ``as_dict()``
    before passing to bw_processing internals.

    Args:
        row: Integer row index in the target matrix.
        col: Integer column index in the target matrix.
        amount: The numeric value to place at (row, col).
        flip: If True, multiply the value by -1 when building the matrix.
        uncertainty_type: Probability distribution type (0 = no uncertainty,
            2 = lognormal, 3 = normal, etc. — see stats_arrays for full list).
        loc: Distribution location parameter. For lognormal this is the log
            of the median; defaults to NaN (no uncertainty).
        scale: Distribution scale parameter (e.g. standard deviation).
        shape: Distribution shape parameter.
        minimum: Lower bound for distribution sampling.
        maximum: Upper bound for distribution sampling.
        negative: Whether the underlying value is negative.
    """

    row: int
    col: int
    amount: float
    flip: bool = False
    uncertainty_type: int = 0
    loc: float = math.nan
    scale: float = math.nan
    shape: float = math.nan
    minimum: float = math.nan
    maximum: float = math.nan
    negative: bool = False

    def __post_init__(self):
        if self.uncertainty_type in _NO_UNCERTAINTY_IDS:
            if math.isnan(self.loc):
                object.__setattr__(self, "loc", self.amount)
            elif self.loc != self.amount:
                raise ValueError(
                    f"uncertainty_type {self.uncertainty_type} requires loc == amount, "
                    f"got loc={self.loc} but amount={self.amount}"
                )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ArrayEntry:
    """All index/flip metadata for one persistent-array resource group.

    Unlike :class:`MatrixEntry`, which represents a single row, ``ArrayEntry``
    holds every row of a resource group together so that the 2-D scenario
    ``data`` array can be supplied directly without decomposing and
    reassembling it.

    Args:
        rows: 1-D sequence of integer row indices, one per matrix entry.
        cols: 1-D sequence of integer column indices, one per matrix entry.
        data: 2-D array of shape ``(n_entries, n_scenarios)``.
        flip: Optional 1-D boolean sequence of length ``n_entries``.
    """

    rows: Union[np.ndarray, list]
    cols: Union[np.ndarray, list]
    data: np.ndarray
    flip: Optional[Union[np.ndarray, list]] = None

    def __post_init__(self):
        rows = np.asarray(self.rows)
        cols = np.asarray(self.cols)
        data = np.asarray(self.data)

        if rows.ndim != 1:
            raise ValueError(f"`rows` must be 1-D, got shape {rows.shape}")
        if cols.shape != rows.shape:
            raise ValueError(
                f"`cols` shape {cols.shape} doesn't match `rows` shape {rows.shape}"
            )
        if data.ndim != 2:
            raise ValueError(f"`data` must be 2-D, got {data.ndim}-D")
        if data.shape[0] != len(rows):
            raise ValueError(
                f"`data` has {data.shape[0]} rows but `rows` has {len(rows)} entries"
            )
        if self.flip is not None:
            flip = np.asarray(self.flip)
            if flip.shape != rows.shape:
                raise ValueError(
                    f"`flip` shape {flip.shape} doesn't match `rows` shape {rows.shape}"
                )


def create_datapackage_from_entries(
    data: dict,
    fs=None,
    **metadata,
):
    """Create a datapackage from a dictionary of :class:`MatrixEntry` lists.

    This is the recommended high-level entry point for building datapackages
    without working directly with NumPy arrays.

    Args:
        data: Dictionary mapping matrix names to lists of :class:`MatrixEntry`
            objects. Use :class:`MatrixName` members as keys for the standard
            Brightway matrices; derived libraries may use plain strings for
            additional matrices::

                {
                    MatrixName.technosphere: [
                        MatrixEntry(row=1, col=4, amount=2.5),
                        MatrixEntry(row=2, col=5, amount=7.0, flip=True),
                    ],
                    MatrixName.biosphere: [
                        MatrixEntry(row=10, col=4, amount=0.3),
                    ],
                }

        fs: Optional filesystem. Defaults to an in-memory filesystem.
        **metadata: Additional keyword arguments passed to
            :func:`create_datapackage` (e.g. ``name``, ``id_``).

    Returns:
        A :class:`Datapackage` instance.
    """
    from bw_processing.datapackage import create_datapackage

    dp = create_datapackage(fs=fs, **metadata)
    for matrix_name, entries in data.items():
        dp.add_entries(
            matrix=matrix_name,
            entries=entries,
            name=f"{matrix_name}-data",
        )
    return dp
