class UndefinedInterface:
    """An interface to external data that isn't saved to disk."""

    pass


class Proxy:
    def __init__(self, func, label, kwargs):
        self.func = func
        self.label = label
        self.kwargs = kwargs

    def __call__(self):
        """Retrieve the data.

        Rewinds the file or buffer to 0, see https://github.com/brightway-lca/bw_processing/issues/9."""
        self.kwargs[self.label].seek(0)
        return self.func(**self.kwargs)
