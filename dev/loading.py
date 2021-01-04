from .errors import FileIntegrityError
from .filesystem import md5
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import json
import numpy as np
import pandas
import zipfile


def flexible_open(path, format, open_function=open, dirpath=None):
    """"Load data packagines in-memory or on the local filesystem. Can handle numpy, json, and csv files."""
    if dirpath is not None:
        path = Path(dirpath) / path

    if format == "npy":
        if dirpath is None:
            # Non-binary mode open inside zipfile
            return np.load(open_function(path, "r"), allow_pickle=False)
        else:
            return np.load(open_function(path, "rb"), allow_pickle=False)
    elif format == "json":
        return json.load(path)
    elif format == "csv":
        return pandas.read_csv(path)
    else:
        raise ValueError(f"format {format} not understood")


def load_package(path, check_integrity=True):
    """Load a serialized data package from ``path``, which can either be a zip archive or a directory.

    If ``path`` is already a dictionary, it is returned unchanged, with the following exception: This function will iterate over the data resources, and try to turn ``ByteIO`` objects into numpy arrays, if possible.

    There should be a good reason to save a Numpy array into a BytesIO object instead of passing it directly. Note that the BytesIO file object will be reset to the beginning (with ``seek(0)``) before being converted.

    Otherwise, we return a dictionary with the data in each resource file (with the filename acting as the key in the dictionary), and the metadata file ``datapackage.json`` (with the key ``datapackage``). The file or directory ``path`` must already exist on the filesystem.

    Metadata has one key injected into it: ``path``, which is the value of the input argument ``path``. This is useful in cases where one is dealing with multiple packages and wants to know their origins.

    This package can load the following resource file types:

        * Numpy files (resource format ``npy``)
        * JSON files (resource format ``json``)
        * CSV files (resource format ``csv``)

    CSV files are loaded with pandas. In the future, they should follow the `frictionless data CSV dialect <https://specs.frictionlessdata.io/csv-dialect/#usage>`__. The metadata file **must** specify the CSV `table schema <https://specs.frictionlessdata.io/table-schema/#language>`__ to ensure that the CSV is loaded correctly. See `the enhancement issue <https://github.com/brightway-lca/bw_processing/issues/4>`__.

    If ``check_integrity``, this function will check the integrity of the file contents using the MD5 hash in the metadata, and raise ``FileIntegrityError`` if there is no agreement. Note that integrity of files inside zip archives is never checked.

    """
    if isinstance(path, Mapping):
        # Data package already loaded, or created in memory.
        # We also support passing binary numpy data as BytesIO objects
        # on the request of the activity-browser team
        return load_bytes(path)
    else:
        path = Path(path)
        assert path.exists(), "Given path not found"

        if path.is_file() and path.suffix == ".zip":
            zf = zipfile.ZipFile(path)
            assert "datapackage.json" in zf.namelist(), "Missing datapackage"
            opener, opener_dir = zf.open, None
        elif path.is_dir():
            zf = None
            assert (path / "datapackage.json").is_file(), "Missing datapackage"
            opener, opener_dir = open, path
        else:
            raise ValueError(f"Can't load file {path}")

        result = {"datapackage": json.load(opener("datapackage.json"))}

        for resource in result["datapackage"]["resources"]:
            if zf is None and check_integrity:
                if not resource["md5"] == md5(path / resource["path"]):
                    raise FileIntegrityError(f"MD5 check failed for {path}")
            result[resource["path"]] = flexible_open(
                path=resource["path"],
                format=resource["format"],
                open_function=opener,
                dirpath=opener_dir,
            )
        return result
