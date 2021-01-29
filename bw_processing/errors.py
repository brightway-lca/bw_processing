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


class FileIntegrityError(BrightwayProcessingError):
    """MD5 hash does not agree with file contents"""

    pass


class Closed(BrightwayProcessingError):
    """Datapackage closed, can't be written to anymore."""

    pass


class LengthMismatch(BrightwayProcessingError):
    """Number of resources doesn't match the number of data objects"""

    pass


class InvalidMimetype(BrightwayProcessingError):
    """Provided mimetype missing or not understood"""

    pass
