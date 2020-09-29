from bw_processing.proxies import ReadProxy


class C:
    called = False
    args = None
    kwargs = None

    def __call__(self, *args, **kwargs):
        self.called = True
        return args, kwargs


def test_read_proxy():
    args = (1, True)
    kwargs = {"foo": "bar", "domain": "example.com"}
    func = C()

    rp = ReadProxy(func, *args, **kwargs)
    assert not func.called
    assert func.args == func.kwargs == None

    assert "deferred" in repr(rp)

    x, y = rp()

    assert func.called
    assert x == (1, True)
    assert y == {"foo": "bar", "domain": "example.com"}
