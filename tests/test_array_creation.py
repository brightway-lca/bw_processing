import numpy as np

from bw_processing.array_creation import chunked, create_array


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
