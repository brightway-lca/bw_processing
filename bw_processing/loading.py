from .errors import FileIntegrityError
from .filesystem import md5
from collections.abc import Mapping
from io import BytesIO
import pandas
from pathlib import Path
import json
import numpy as np
import zipfile


def _load_npy(obj):
    return np.load(obj, allow_pickle=False)


def _load_json(obj):
    return json.load(obj)


def _load_csv(obj):
    return pandas.read_csv(obj)


mapping = {
    'npy': _load_npy,
    'json': _load_json,
    'csv': _load_csv
}


def load_package(path, check_integrity=True):
    """Load a serialized data package from ``path``, which can either be a zip archive or a directory.

    If ``path`` is already a dictionary, it is returned unchanged, with the following exception: This function will iterate over the data resources, and try to turn `ByteIO` objects into numpy arrays, if possible.

    Otherwise, we return a dictionary with the data in each resource file (with the filename acting as the key in the dictionary), and the metadata file ``datapackage.json`` (with the key ``datapackage``). The file or directory ``path`` must already exist on the filesystem.

    Metadata has one key injected into it: ``path``, which is the value of the input argument ``path``. This is useful in cases where one is dealing with multiple packages and wants to know their origins.

    This package can load the following resource file types:

        * Numpy files (file extension ``npy``)
        * JSON files (file extension ``json``)
        * CSV files (file extension ``csv``)

    File extensions are not case sensitive, but lower-case is preferred.

    CSV files are loaded with pandas. In the future, they should follow the `frictionless data CSV dialect <https://specs.frictionlessdata.io/csv-dialect/#usage>`__. The metadata file **must** specify the CSV `table schema <https://specs.frictionlessdata.io/table-schema/#language>`__ to ensure that the CSV is loaded correctly. See `the enhancement issue <https://github.com/brightway-lca/bw_processing/issues/4>`__.

    If ``check_integrity``, this function will check the integrity of the file contents using the MD5 hash in the metadata, and raise ``FileIntegrityError`` if there is no agreement. Note that integrity of files inside zip archives is never checked.

    """
    if isinstance(path, Mapping):
        # We support passing binary numpy data as BytesIO objects
        # on the request of the activity-browser team
        for k, v in path.items():
            if k == 'datapackage':
                continue
            elif not isinstance(v, Mapping):
                continue

            for x, y in v.items():
                if isinstance(y, BytesIO):
                    try:
                        k[x] = np.load(y, allow_pickle=False)
                    except ValueError:
                        pass
        return path
    else:
        path = Path(path)
        assert path.exists(), "Given path not found"

        if path.is_file() and path.suffix == ".zip":
            zf = zipfile.ZipFile(path)
            assert "datapackage.json" in zf.namelist(), "Missing datapackage"
            result = {'datapackage': json.load(zf.open("datapackage.json"))}
            for resource in result['datapackage']['resources']:
                extension = resource['path'].split(".")[-1].lower()
                try:
                    function = mapping[extension]
                    result[resource['path']] = function(zf.open(resource['path']))
                except KeyError:
                    raise KeyError("No handler for file {}".format(resource['path']))
                return result
        elif path.is_dir():
            assert (path / "datapackage.json").is_file(), "Missing datapackage"
            result = {'datapackage': json.load(open(path / "datapackage.json"))}
            for resource in result['datapackage']['resources']:
                if check_integrity:
                    if not resource['hash'] == md5(path / resource['path']):
                        raise FileIntegrityError(f"MD5 check failed for {path}")

                extension = resource['path'].split(".")[-1].lower()
                try:
                    function = mapping[extension]
                    result[resource['path']] = function(open(path / resource['path']))
                except KeyError:
                    raise KeyError("No handler for file {}".format(resource['path']))
            return result
        else:
            raise ValueError(f"Can't load file {path}")
