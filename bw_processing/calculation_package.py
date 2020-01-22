from .filesystem import md5, safe_filename
from .utils import create_datapackage_metadata, create_numpy_structured_array
from pathlib import Path
import json
import tempfile
import uuid
import zipfile


def format_calculation_resource(res):
    """Format metadata for a `datapackage resource <https://frictionlessdata.io/specs/data-resource/>`__.

    ..note:: This function is for use together with ``create_calculation_package``, it doesn't create valid `datapackage resources <https://frictionlessdata.io/specs/data-resource/>`__.

    ``res`` should be a dictionary with the following keys:

        name (str): Simple name or identifier to be used for this matrix data
        matrix (str): The name of the matrix to build. See the documentation for ``bw_calc`` for more details.
        data (iterator): Iterator to be fed into ``format_function`` to generate array rows.
        nrows (int, optional): Number of rows in array.
        path (str, optional): Filename for saved Numpy array
        format_function (callable, optional):  Function to call on each row in ``data`` to put elements in correct type and order for insertion into structured array.

    ``res`` can also have `optional keys <https://frictionlessdata.io/specs/data-resource/>`__ like ``description``, and ``title``.

    Returns:
        A dictionary ready for JSON serialization in the datapackage format.

    TODO: Think about declaring a custom JSON schema for our datapackages, see:

        * https://frictionlessdata.io/specs/profiles/
        * https://frictionlessdata.io/schemas/data-resource.json
        * https://json-schema.org/

    """
    obj = {
        # Datapackage generic
        "format": "npy",
        "mediatype": "application/octet-stream",
        "path": res.get("path") or uuid.uuid4().hex,
        "name": res["name"],
        "profile": "data-resource",
        # Brightway specific
        "matrix": res["matrix"],
    }
    # Leave separate because maybe want to add to later
    SKIP = set(obj)
    if not obj["path"].endswith(".npy"):
        obj["path"] += ".npy"
    for key, value in res.items():
        if key not in obj and key not in SKIP:
            obj[key] = value
    return obj


def create_calculation_package(
    name,
    resources,
    path=None,
    id_=None,
    metadata=None,
    replace=True,
    compress=True,
    **kwargs
):
    """Create a calculation package for use in ``bw_calc``.

    If ``path`` is ``None``, then the package is created in memory and returned as a dict. Otherwise, the datapackage is stored to disk, either as a zipfile (if ``compress``) or as a directory. The directory should already exist.

    The ``format_function`` should return a tuple of data that fits the structured array datatype, i.e.

        ("row_value", np.uint32),
        ("col_value", np.uint32),
        ("row_index", np.uint32),
        ("col_index", np.uint32),
        ("uncertainty_type", np.uint8),
        ("amount", np.float32),
        ("loc", np.float32),
        ("scale", np.float32),
        ("shape", np.float32),
        ("minimum", np.float32),
        ("maximum", np.float32),
        ("negative", np.bool),
        ("flip", np.bool),

    Args:
        name (str): Name of this calculation package
        resources (iterable): Resources is an iterable of dictionaries with the keys:
            TODO: Update based on above changes
            name (str): Simple name or identifier to be used for this matrix data
            matrix (str): The name of the matrix to build. See the documentation for ``bw_calc`` for more details.
            data (iterable): The numerical data to be stored
            nrows (int, optional):  The number of rows in ``array``. Will be counted if not provided, but with an efficiency penalty.
            format_function (callable, optional): Function that formats data to structured array columns.
        path (str, Path, or None): Location to store the created package.
        id_ (str, optional): Unique ID of this calculation package
        metadata (dict, optional): Additional metadata such as licenses, RNG seeds, etc.
        replace (bool, optional): Replace an existing calculation package with the same name and path

    Returns:
        Absolute filepath to calculation package (zip file)

    """
    # In-memory zipfile creation not currently supported
    # (see https://github.com/brightway-lca/bw_calc/issues/1)"
    if path is None:
        compress = False

    if path:
        path = Path(path)
        assert path.is_dir()
    else:
        result = {}

    if path and compress:
        archive = path / (safe_filename(name) + ".zip")
        if archive.is_file():
            if replace:
                archive.unlink()
            else:
                raise ValueError("This calculation package already exists")

        base_td = tempfile.TemporaryDirectory()
        td = Path(base_td.name)
    else:
        td = None

    for resource in resources:
        filename = resource["path"]
        if path is None:
            filepath = None
        else:
            filepath = (td if compress else path) / filename
        array = create_numpy_structured_array(
            iterable=resource.pop("data"),
            filepath=filepath,
            nrows=resource.pop("nrows", None),
            format_function=resource.pop("format_function", None),
            dtype=resource.pop("dtype", None),
        )
        if path is None:
            result[filename] = array
        else:
            resource["md5"] = md5(filepath)

    datapackage = create_datapackage_metadata(
        name=name,
        resources=resources,
        resource_function=format_calculation_resource,
        id_=id_,
        metadata=metadata,
    )
    if path is None:
        result["datapackage"] = datapackage
    else:
        dirpath = td if compress else path
        with open(dirpath / "datapackage.json", "w", encoding="utf-8") as f:
            json.dump(datapackage, f, indent=2, ensure_ascii=False)

    if compress:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in td.iterdir():
                if file.is_file():
                    zf.write(file, arcname=file.name)
        del base_td
        return archive
    elif path is None:
        return result
    else:
        return path
