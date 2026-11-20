import numpy as np

from bw_processing.array_creation import (
    chunked,
    create_chunked,
)


def test_chunked():
    c = chunked(range(600), 250)
    for x in next(c):
        pass
    assert x == 249
    for x in next(c):
        pass
    assert x == 499
    for x in next(c):
        pass
    assert x == 599


# --- create_chunked (structured 1D) ---

SIMPLE_DTYPE = [("row", np.int32), ("col", np.int32), ("amount", np.float32)]


def test_create_chunked_structured_under_one_bucket():
    data = [(i, i + 1, float(i)) for i in range(10)]
    result = create_chunked(iter(data), SIMPLE_DTYPE, bucket_size=20000)
    assert result.shape == (10,)
    assert list(result["row"]) == list(range(10))


def test_create_chunked_structured_multiple_buckets():
    data = [(i, i + 1, float(i)) for i in range(500)]
    result = create_chunked(iter(data), SIMPLE_DTYPE, bucket_size=200)
    assert result.shape == (500,)
    assert result["row"][0] == 0
    assert result["row"][499] == 499


def test_create_chunked_structured_empty():
    result = create_chunked(iter([]), SIMPLE_DTYPE)
    assert result.shape == (0,)
