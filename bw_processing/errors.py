class BrightwayProcessingError(Exception):
    pass


class InconsistentFields(BrightwayProcessingError):
    """Given fields not the same for each element"""

    pass


class NonUnique(BrightwayProcessingError):
    """Nonunique elements when uniqueness is required"""

    pass


class InvalidName(BrightwayProcessingError):
    """Name fails datapackage requirements:

    A short url-usable (and preferably human-readable) name of the package. This MUST be lower-case and contain only alphanumeric characters along with ".", "_" or "-" characters."""

    pass
