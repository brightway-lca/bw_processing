class UndefinedInterface:
    """An interface to external data that isn't saved to disk."""

    pass


class Proxy:
    """Deferred file-read wrapper returned when ``proxy=True`` is passed to ``file_reader``.

    Stores the reader function and its arguments without executing them. The
    actual data is loaded the first time the proxy is called (i.e. when
    ``get_resource`` resolves it). The file or buffer is rewound to position 0
    before each call so that repeated calls return the same data.
    """

    def __init__(self, func, label, kwargs):
        self.func = func
        self.label = label
        self.kwargs = kwargs

    def __call__(self):
        """Retrieve the data.

        Rewinds the file or buffer to 0, see https://github.com/brightway-lca/bw_processing/issues/9.
        """
        self.kwargs[self.label].seek(0)
        return self.func(**self.kwargs)
