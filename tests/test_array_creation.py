from bw_processing.array_creation import chunked


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
