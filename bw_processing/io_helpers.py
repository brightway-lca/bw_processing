import json
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.zip import ZipFileSystem
from morefs.dict import DictFS

from .constants import MatrixSerializeFormat
from .errors import InvalidMimetype
from .proxies import Proxy

try:
    from .io_parquet_helpers import load_ndarray_from_parquet, save_arr_to_parquet

    PARQUET = True
except ImportError:
    load_ndarray_from_parquet = None
    save_arr_to_parquet = None
    PARQUET = False


def generic_directory_filesystem(*, dirpath: Path) -> DirFileSystem:
    assert isinstance(dirpath, Path), "`dirpath` must be a `pathlib.Path` instance"
    if not dirpath.is_dir():
        if not dirpath.parent.is_dir():
            raise ValueError("Parent directory `{}` doesn't exist".format(dirpath.parent))
        dirpath.mkdir()
    return DirFileSystem(path=dirpath, fs=LocalFileSystem())


def generic_zipfile_filesystem(
    *, dirpath: Path, filename: str, write: bool = True
) -> ZipFileSystem:
    assert isinstance(dirpath, Path), "`dirpath` must be a `pathlib.Path` instance"
    if not dirpath.is_dir():
        raise ValueError("Destination directory `{}` doesn't exist".format(dirpath))
    return ZipFileSystem(dirpath / filename, mode="w" if write else "r")


def file_reader(
    *,
    fs: AbstractFileSystem,
    resource: str,
    mimetype: str,
    proxy: bool = False,
    mmap_mode: Union[str, None] = None,
    **kwargs,
) -> Any:
    if resource.endswith(".npy"):  # TODO: constant
        mimetype = "application/octet-stream"
    elif resource.endswith(".parquet"):  # TODO: constant
        mimetype = "application/octet-stream"
    else:
        mimetype, _ = guess_type(resource)

    if mimetype == "application/octet-stream":
        if resource.endswith(".npy"):  # TODO: constant
            mimetype = "application/numpy"
        elif resource.endswith(".parquet"):  # TODO: constant
            mimetype = "application/parquet"
        else:
            raise TypeError(
                f"application/octet-stream mimetype (resource: {resource}) not recognized"
            )

    if isinstance(resource, Path):
        resource = str(resource)

    mapping = {
        "application/numpy": (
            np.load,
            "file",
            {
                "file": fs.open(resource, mode="rb"),
                "mmap_mode": mmap_mode,
                "allow_pickle": False,
            },
        ),
        "application/json": (
            json.load,
            "fp",
            {"fp": fs.open(resource, encoding="utf-8")},
        ),
        "text/csv": (
            pd.read_csv,
            "filepath_or_buffer",
            {"filepath_or_buffer": fs.open(resource)},
        ),
    }
    if PARQUET:
        mapping["application/parquet"] = (
            load_ndarray_from_parquet,
            "file",
            {
                "file": fs.open(resource, mode="rb"),
            },
        )

    try:
        func, label, kwargs = mapping[mimetype]
    except KeyError:
        raise InvalidMimetype("Mimetype '{}' not understoof".format(mimetype))

    if proxy:
        return Proxy(func, label, kwargs)
    else:
        return func(**kwargs)


def file_writer(
    *,
    data: Any,
    fs: AbstractFileSystem,
    resource: str,
    mimetype: str,
    matrix_serialize_format_type: MatrixSerializeFormat = MatrixSerializeFormat.NUMPY,  # NIKO
    meta_object: Optional[str] = None,
    meta_type: Optional[str] = None,
    **kwargs,
) -> None:
    if isinstance(resource, Path):
        resource = str(resource)

    if mimetype == "application/octet-stream":
        if matrix_serialize_format_type == MatrixSerializeFormat.NUMPY:
            with fs.open(resource, mode="wb") as fo:
                return np.save(fo, data, allow_pickle=False)
        elif matrix_serialize_format_type == MatrixSerializeFormat.PARQUET:
            if not PARQUET:
                raise ImportError("`pyarrow` library not installed")
            assert meta_type is not None
            assert meta_object is not None

            return save_arr_to_parquet(
                fs.open(resource, mode="wb"),
                data,
                meta_object=meta_object,
                meta_type=meta_type,
            )
        else:
            raise TypeError(
                f"Matrix serialize format type {matrix_serialize_format_type} is not recognized!"
            )
    elif mimetype == "application/json":
        return json.dump(
            data,
            fs.open(resource, mode="w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
    elif mimetype == "text/csv":
        assert isinstance(data, pd.DataFrame)
        data.to_csv(fs.open(resource, mode="w", encoding="utf-8"), index=False)
