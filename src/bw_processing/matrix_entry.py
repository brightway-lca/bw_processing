import dataclasses
import math
from enum import Enum
from typing import Optional

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
        rescale: Per-exchange multiplicative factor applied before matrix
            insertion. ``1.0`` (the default) leaves the value unchanged.
            Stored as a ``rescale_array`` resource (``kind="rescale"``).
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
    rescale: float = 1.0

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
        rescale: Optional 1-D float array of per-entry multiplicative factors
            (one per row). ``1.0`` leaves the value unchanged. Stored as a
            ``rescale_array`` resource (``kind="rescale"``).
    """

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    flip: Optional[np.ndarray] = None
    rescale: Optional[np.ndarray] = None

    def __post_init__(self):
        self.rows = np.asarray(self.rows)
        self.cols = np.asarray(self.cols)
        self.data = np.asarray(self.data)

        if self.rows.ndim != 1:
            raise ValueError(f"`rows` must be 1-D, got shape {self.rows.shape}")
        if not np.issubdtype(self.rows.dtype, np.integer):
            raise ValueError(f"`rows` must have integer dtype, got {self.rows.dtype}")
        if self.cols.shape != self.rows.shape:
            raise ValueError(
                f"`cols` shape {self.cols.shape} doesn't match `rows` shape {self.rows.shape}"
            )
        if not np.issubdtype(self.cols.dtype, np.integer):
            raise ValueError(f"`cols` must have integer dtype, got {self.cols.dtype}")
        if self.data.ndim != 2:
            raise ValueError(f"`data` must be 2-D, got {self.data.ndim}-D")
        if self.data.shape[0] != len(self.rows):
            raise ValueError(
                f"`data` has {self.data.shape[0]} rows but `rows` has {len(self.rows)} entries"
            )
        if self.flip is not None:
            self.flip = np.asarray(self.flip, dtype=bool)
            if self.flip.shape != self.rows.shape:
                raise ValueError(
                    f"`flip` shape {self.flip.shape} doesn't match `rows` shape {self.rows.shape}"
                )
        if self.rescale is not None:
            self.rescale = np.asarray(self.rescale, dtype=np.float32)
            if self.rescale.shape != self.rows.shape:
                raise ValueError(
                    f"`rescale` shape {self.rescale.shape} doesn't match `rows` shape {self.rows.shape}"
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
