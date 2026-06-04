import itertools

import numpy as np


def peek(iterator):
    iterator = iter(iterator)
    first = next(iterator)
    return first, itertools.chain([first], iterator)


def get_ncols(iterator):
    a, b = peek(iterator)
    return len(a), b


def chunked(iterable, chunk_size):
    # Black magic, see https://stackoverflow.com/a/31185097
    # and https://docs.python.org/3/library/functions.html#iter
    iterable = iter(iterable)  # Fix e.g. range from restarting
    return iter(lambda: list(itertools.islice(iterable, chunk_size)), [])


def create_chunked(iterable, dtype, ncols=None, bucket_size=500):
    """Create a numpy array from an iterable of indeterminate length.

    Needed when we can't determine the length of the iterable ahead of time
    (e.g. for a generator or a database cursor), so can't create the complete
    array in memory in one step.

    Creates a list of arrays with ``bucket_size`` rows until ``iterable`` is
    exhausted, then concatenates them along axis 0.

    Pass ``ncols`` for a plain 2D array; omit it for a 1D structured array.

    Args:
        iterable: Iterable of data used to populate the array.
        dtype: Numpy dtype of the created array.
        ncols: Number of columns; if None, a 1D structured array is created.
        bucket_size: Number of rows in each intermediate array.

    Returns:
        The created array. Returns a zero-length array if ``iterable`` has no data.
    """
    bucket_shape = (bucket_size,) if ncols is None else (bucket_size, ncols)
    empty_shape = (0,) if ncols is None else (0, ncols)
    arrays = []
    array = np.zeros(bucket_shape, dtype=dtype)
    for chunk in chunked(iterable, bucket_size):
        for i, row in enumerate(chunk):
            array[i] = row
        if i < bucket_size - 1:
            # .copy() releases the oversized bucket buffer immediately rather
            # than keeping it alive as a view until the final concatenation.
            arrays.append(array[: i + 1].copy())
        else:
            arrays.append(array)
            array = np.zeros(bucket_shape, dtype=dtype)
    # Empty iterable - create zero-length array.
    # Needed because we return iterators for SQL databases
    # but don't know if e.g. sometimes a database has no biosphere exchanges.
    return np.concatenate(arrays, axis=0) if arrays else np.zeros(empty_shape, dtype=dtype)


def create_structured_array(iterable, dtype, nrows=None, sort=False, sort_fields=None):
    """Create a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ for data ``iterable``. Returns a filepath of a created file (if ``filepath`` is provided, or the array.

    ``iterable`` can be data already in memory, or a generator.

    ``nrows`` can be supplied, if known. If ``iterable`` has a length, it will be determined automatically. If ``nrows`` is not known, this function generates chunked arrays until ``iterable`` is exhausted, and concatenates them.
    """
    if nrows or hasattr(iterable, "__len__"):
        if not nrows:
            nrows = len(iterable)
        array = np.zeros(nrows, dtype=dtype)
        for i, row in enumerate(iterable):
            if i > (nrows - 1):
                raise ValueError("More rows than `nrows`")
            array[i] = tuple(row)

    else:
        array = create_chunked(iterable, dtype)

    if sort:
        sort_fields = sort_fields or ()
        dtype_fields = {x[0] for x in dtype}
        order = [x for x in sort_fields if x in dtype_fields] + sorted(
            [x for x in dtype_fields if x not in sort_fields]
        )
        array.sort(order=order)

    return array


def create_array(iterable, nrows=None, dtype=np.float32):
    """Create a numpy array data ``iterable``. Returns a filepath of a created file (if ``filepath`` is provided, or the array.

    ``iterable`` can be data already in memory, or a generator.

    ``nrows`` can be supplied, if known. If ``iterable`` has a length, it will be determined automatically. If ``nrows`` is not known, this function generates chunked arrays until ``iterable`` is exhausted, and concatenates them.

    Either ``nrows`` or ``ncols`` must be specified."""
    if isinstance(iterable, np.ndarray):
        array = iterable.astype(dtype)
    elif nrows or hasattr(iterable, "__len__"):
        if not nrows:
            nrows = len(iterable)
        ncols, data = get_ncols(iterable)
        array = np.zeros((nrows, ncols), dtype=dtype)
        for i, row in enumerate(data):
            if i > (nrows - 1):
                raise ValueError("More rows than `nrows`")
            array[i, :] = tuple(row)

    else:
        try:
            ncols, data = get_ncols(iterable)
        except StopIteration:
            return np.zeros((0, 0), dtype=dtype)
        array = create_chunked(data, dtype, ncols=ncols)

    return array
