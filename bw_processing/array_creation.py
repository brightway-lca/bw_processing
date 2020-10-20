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


def create_chunked_structured_array(iterable, dtype, bucket_size=20000):
    """Create a numpy structured array from an iterable of indeterminate length.

    Needed when we can't determine the length of the iterable ahead of time (e.g. for a generator or a database cursor), so can't create the complete array in memory in on step

    Creates a list of arrays with ``bucket_size`` rows until ``iterable`` is exhausted, then concatenates them.

    Args:
        iterable: Iterable of data used to populate the array.
        dtype: Numpy dtype of the created array
        format_function: If provided, this function will be called on each row of ``iterable`` before insertion in the array.
        bucket_size: Number of rows in each intermediate array.

    Returns:.
        Returns the created array. Will return a zero-length array if ``iterable`` has no data.

    """
    arrays = []
    array = np.zeros(bucket_size, dtype=dtype)

    for chunk in chunked(iterable, bucket_size):
        for i, row in enumerate(chunk):
            array[i] = row
        if i < bucket_size - 1:
            array = array[: i + 1]
            arrays.append(array)
        else:
            arrays.append(array)
            array = np.zeros(bucket_size, dtype=dtype)

    # Empty iterable - create zero-length array
    # Needed because we return iterators for SQL databases
    # but don't know if e.g. sometime a database has
    # no biosphere exchanges
    if arrays:
        array = np.hstack(arrays)
    else:
        array = np.zeros(0, dtype=dtype)

    return array


def create_structured_array(iterable, dtype, nrows=None, sort=False, sort_fields=None):
    """Create a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ for data ``iterable``. Returns a filepath of a created file (if ``filepath`` is provided, or the array.

    ``iterable`` can be data already in memory, or a generator.

    ``nrows`` can be supplied, if known. If ``iterable`` has a length, it will be determined automatically. If ``nrows`` is not known, this function generates chunked arrays until ``iterable`` is exhausted, and concatenates them."""
    if nrows or hasattr(iterable, "__len__"):
        if not nrows:
            nrows = len(iterable)
        array = np.zeros(nrows, dtype=dtype)
        for i, row in enumerate(iterable):
            if i > (nrows - 1):
                raise ValueError("More rows than `nrows`")
            array[i] = tuple(row)

    else:
        array = create_chunked_structured_array(iterable, dtype)

    if sort:
        sort_fields = sort_fields or ()
        dtype_fields = {x[0] for x in dtype}
        order = [x for x in sort_fields if x in dtype_fields] + sorted(
            [x for x in dtype_fields if x not in sort_fields]
        )
        array.sort(order=order)

    return array


def create_chunked_array(iterable, ncols, dtype=np.float32, bucket_size=500):
    """Create a numpy array from an iterable of indeterminate length.

    Needed when we can't determine the length of the iterable ahead of time (e.g. for a generator or a database cursor), so can't create the complete array in memory in on step

    Creates a list of arrays with ``bucket_size`` rows until ``iterable`` is exhausted, then concatenates them.

    Args:
        iterable: Iterable of data used to populate the array.
        ncols: Number of columns in the created array.
        dtype: Numpy dtype of the created array
        bucket_size: Number of rows in each intermediate array.

    Returns:.
        Returns the created array. Will return a zero-length array if ``iterable`` has no data.

    """
    arrays = []
    array = np.zeros((bucket_size, ncols), dtype=dtype)

    for chunk in chunked(iterable, bucket_size):
        for i, row in enumerate(chunk):
            array[i, :] = row
        if i < bucket_size - 1:
            array = array[: i + 1, :]
            arrays.append(array)
        else:
            arrays.append(array)
            array = np.zeros((bucket_size, ncols), dtype=dtype)

    if arrays:
        array = np.hstack(arrays)
    else:
        array = np.zeros((0, ncols), dtype=dtype)

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
        ncols, data = get_ncols(iterable)
        array = create_chunked_array(data, ncols, dtype)

    return array
