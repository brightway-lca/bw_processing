class ReadProxy:
    """A simple proxy that defers reading until the object is executed.

    A slightly more elegant version of ``lambda : func``."""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*args, **self.kwargs)

    def __repr__(self):
        return "A deferred function that will read data only when needed"


class GenericProxy:
    """A resource that we don't understand"""

    pass
