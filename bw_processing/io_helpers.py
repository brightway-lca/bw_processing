from .errors import InvalidMimetype
from fs.base import FS
from fs.osfs import OSFS
from fs.zipfs import ZipFS
from functools import partial
from mimetypes import guess_type
from pathlib import Path
from typing import Union, Any
import json
import numpy as np
import pandas as pd


def generic_directory_filesystem(*, dirpath: Path) -> OSFS:
    assert isinstance(dirpath, Path), "`dirpath` must be a `pathlib.Path` instance"
    if not dirpath.is_dir():
        if not dirpath.parent.is_dir():
            raise ValueError(
                "Parent directory `{}` doesn't exist".format(dirpath.parent)
            )
        dirpath.mkdir()
    return OSFS(dirpath)


def generic_zipfile_filesystem(
    *, dirpath: Path, filename: str, write: bool = True
) -> ZipFS:
    assert isinstance(dirpath, Path), "`dirpath` must be a `pathlib.Path` instance"
    if not dirpath.is_dir():
        raise ValueError("Destination directory `{}` doesn't exist".format(dirpath))
    return ZipFS(dirpath / filename, write=write)


def file_reader(
    *,
    fs: FS,
    resource: str,
    mimetype: str,
    proxy: bool = False,
    mmap_mode: Union[str, None] = None,
    **kwargs
) -> Any:
    if mimetype is None and resource.endswith(".npy"):
        mimetype = "application/octet-stream"
    elif mimetype is None:
        mimetype, _ = guess_type(resource)

    if isinstance(resource, Path):
        resource = str(resource)

    mapping = {
        "application/octet-stream": (
            np.load,
            {
                "file": fs.open(resource, mode="rb"),
                "mmap_mode": mmap_mode,
                "allow_pickle": False,
            },
        ),
        "application/json": (json.load, {"fp": fs.open(resource, encoding="utf-8")}),
        "text/csv": (pd.read_csv, {"filepath_or_buffer": fs.open(resource)}),
    }

    try:
        func, kwargs = mapping[mimetype]
    except KeyError:
        raise InvalidMimetype("Mimetype '{}' not understoof".format(mimetype))

    if proxy:
        return partial(func, **kwargs)
    else:
        return func(**kwargs)


def file_writer(*, data: Any, fs: FS, resource: str, mimetype: str, **kwargs) -> None:
    if isinstance(resource, Path):
        resource = str(resource)

    if mimetype == "application/octet-stream":
        return np.save(fs.open(resource, mode="wb"), data, allow_pickle=False)
    elif mimetype == "application/json":
        return json.dump(
            data,
            fs.open(resource, mode="w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
    elif mimetype == "text/csv":
        assert isinstance(data, pd.DataFrame)
        data.to_csv(fs.open(resource, mode="w", encoding="utf-8"))
