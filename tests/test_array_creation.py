import numpy as np

from bw_processing.array_creation import (
    chunked,
    create_array,
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


def test_create_array_empty_generator_returns_empty_array():
    # Empty generator with no nrows previously raised StopIteration (issue #96)
    result = create_array(x for x in [])
    assert result.shape == (0, 0)
    assert result.dtype == np.float32


def test_create_array_empty_generator_respects_dtype():
    result = create_array((x for x in []), dtype=np.float64)
    assert result.dtype == np.float64


def test_create_array_nonempty_generator():
    data = ([float(i), float(i * 2)] for i in range(5))
    result = create_array(data)
    assert result.shape == (5, 2)
    assert np.allclose(result[:, 0], [0, 1, 2, 3, 4])
    assert np.allclose(result[:, 1], [0, 2, 4, 6, 8])


# --- create_chunked (plain 2D) ---

def test_create_chunked_under_one_bucket():
    data = [np.ones(3) * i for i in range(10)]
    result = create_chunked(iter(data), np.float32, ncols=3, bucket_size=500)
    assert result.shape == (10, 3)
    assert np.allclose(result[7], [7, 7, 7])


def test_create_chunked_multiple_full_buckets():
    # Previously hstack joined on axis=1 → (bucket_size, ncols*n) instead of (total, ncols)
    data = [np.ones(3) * i for i in range(1000)]
    result = create_chunked(iter(data), np.float32, ncols=3, bucket_size=500)
    assert result.shape == (1000, 3)
    assert np.allclose(result[0], [0, 0, 0])
    assert np.allclose(result[999], [999, 999, 999])


def test_create_chunked_full_plus_partial_bucket():
    # Previously hstack raised ValueError when a full and partial chunk were concatenated
    data = [np.ones(3) * i for i in range(600)]
    result = create_chunked(iter(data), np.float32, ncols=3, bucket_size=500)
    assert result.shape == (600, 3)
    assert np.allclose(result[499], [499, 499, 499])
    assert np.allclose(result[599], [599, 599, 599])


def test_create_chunked_empty_plain():
    result = create_chunked(iter([]), np.float32, ncols=4, bucket_size=500)
    assert result.shape == (0, 4)


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
