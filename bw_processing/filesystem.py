from pathlib import Path
from typing import Union
import hashlib
import re
import unicodedata

re_slugify = re.compile(r"[^\w\s-]", re.UNICODE)
SUBSTITUTION_RE = re.compile(r"[^\w\-\.]")
MULTI_RE = re.compile(r"_{2,}")


def clean_datapackage_name(name: str) -> str:
    """Clean string ``name`` of characters not allowed in data package names.

    Replaces with underscores, and drops multiple underscores."""
    return re.sub(MULTI_RE, "_", re.sub(SUBSTITUTION_RE, "_", name).strip("_")).strip()


def safe_filename(
    string: Union[str, bytes], add_hash: bool = True, full: bool = False
) -> str:
    """Convert arbitrary strings to make them safe for filenames. Substitutes strange characters, and uses unicode normalization.

    if `add_hash`, appends hash of `string` to avoid name collisions.

    From http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename-in-python"""
    safe = re.sub(
        r"[-\s]+",
        "-",
        str(re_slugify.sub("", unicodedata.normalize("NFKD", str(string))).strip()),
    )
    if add_hash:
        if isinstance(string, str):
            string = string.encode("utf8")
        if full:
            safe += "." + hashlib.md5(string).hexdigest()
        else:
            safe += "." + hashlib.md5(string).hexdigest()[:8]
    return safe


def md5(filepath: Union[str, Path], blocksize: int = 65536) -> str:
    """Generate MD5 hash for file at `filepath`"""
    hasher = hashlib.md5()
    fo = open(filepath, "rb")
    buf = fo.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = fo.read(blocksize)
    return hasher.hexdigest()
