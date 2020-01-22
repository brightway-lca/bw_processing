from .errors import InconsistentFields, NonUnique
from .filesystem import md5, safe_filename
from .utils import create_datapackage_metadata
from pathlib import Path
import csv
import json
import numpy as np
import tempfile
import uuid
import zipfile


def greedy_set_cover(data, exclude=None):
    """Find unique set of attributes that uniquely identifies each element in ``data``.

    Feature selection is a well known problem, and is analogous to the `set cover problem <https://en.wikipedia.org/wiki/Set_cover_problem>`__, for which there is a `well known heuristic <https://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm>`__.

    Example:

        data = [
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 2, 'b': 2, 'c': 3},
            {'a': 1, 'b': 2, 'c': 4},
        ]
        greedy_set_cover(data)
        >>> {'a', 'c'}

    Args:
        data (iterable): List of dictionaries with the same fields.
        exclude (iterable): Fields to exclude during search for uniqueness. ``id`` is Always excluded.

    Returns:
        Set of attributes (strings)

    Raises:
        NonUnique: The given fields are not enough to ensure uniqueness
    """
    exclude = set([]) if exclude is None else set(exclude)
    exclude.add("id")

    fields = {field for obj in data for field in obj if field not in exclude}
    chosen = set([])

    def values_for_fields(data, exclude, fields):
        return sorted(
            [(len({obj[field] for obj in data}), field) for field in fields],
            reverse=True,
        )

    def coverage(data, chosen):
        return len({tuple([obj[field] for field in chosen]) for obj in data})

    while coverage(data, chosen) != len(data):
        if not fields:
            raise NonUnique
        next_field = values_for_fields(data, exclude, fields)[0][1]
        fields.remove(next_field)
        chosen.add(next_field)

    return chosen


def as_unique_attributes(data, exclude=None, include=None):
    """Format ``data`` as unique set of attributes and values for use in ``create_processed_datapackage``.

    Note: Each element in ``data`` must have the attributes ``id``.

        data = [
            {},
        ]

    Args:
        data (iterable): List of dictionaries with the same fields.
        exclude (iterable): Fields to exclude during search for uniqueness. ``id`` is Always excluded.
        include (iterable): Fields to include when returning, even if not unique

    Returns:
        (list of field names as strings, dictionary of data ids to values for given field names)

    Raises:
        InconsistentFields: Not all features provides all fields.
    """
    include = set([]) if include is None else set(include)
    fields = greedy_set_cover(data, exclude)

    if len({tuple(sorted(obj.keys())) for obj in data}) > 1:
        raise InconsistentFields

    def formatter(obj, fields, include):
        return {
            key: value
            for key, value in obj.items()
            if (key in fields or key in include or key == "id")
        }

    return (
        fields.union(include).union({"id"}),
        [formatter(obj, fields, include) for obj in data],
    )


def format_processed_resource(res):
    """Format metadata for a `datapackage resource <https://frictionlessdata.io/specs/data-resource/>`__.

    ``res`` should be a dictionary with the following keys:

        name (str): Simple name or identifier to be used for this matrix data
        filename (str): Filename for saved Numpy array
        matrix (str): The name of the matrix to build. See the documentation for ``bw_calc`` for more details.
        dirpath (pathlib.Path): The directory where the datapackage and resource files are saved.

    ``res`` can also have `optional keys <https://frictionlessdata.io/specs/data-resource/>`__ like ``description``, and ``title``.

    Returns:
        A dictionary ready for JSON serialization in the datapackage format.

    TODO: Think about declaring a custom JSON schema for our datapackages, see:

        * https://frictionlessdata.io/specs/profiles/
        * https://frictionlessdata.io/schemas/data-resource.json
        * https://json-schema.org/

    """
    SKIP = {"dirpath", "filename", "data", "nrows", "format_function"}
    obj = {
        # Datapackage generic
        "format": "npy",
        "mediatype": "application/octet-stream",
        "path": res["filename"],
        "name": res["name"],
        "profile": "data-resource",
        # Brightway specific
        "matrix": res["matrix"],
    }
    # Not needed if in-memory
    if res.get("dirpath"):
        obj["md5"] = md5(res["dirpath"] / res["filename"])
    for key, value in res.items():
        if key not in obj and key not in SKIP:
            obj[key] = value
    return obj


def create_processed_datapackage(
    name, array, rows, cols, path, id_=None, metadata=None
):
    """Create a datapackage with numpy structured arrays and metadata.

    Exchanging large, dense datasets like MRIO tables is not efficient if each exchange must be listed separately. Instead, we would prefer to exchange the processed arrays used to build the matrices directly. However, these arrays use integer indices which are not consistent across computers or even Brightway projects. This function includes additional metadata to solve this problem, mapping these integer ids to enough attributes to uniquely identify each feature. Separate metadata files are included for each column in the array (i.e. the row and column indices).

    Args:
        name (str): Name of this processed package
        array (numpy structured array): The numeric data. Usually generated via ``create_numpy_structured_array``.
        rows (dict): Dictionary mapping integer indices in ``row_value`` to a dictionary of attributes.
        cols (dict): Dictionary mapping integer indices in ``col_value`` to a dictionary of attributes.
        path (str, Path): Directory to store the created package.
        id_ (str, optional): Unique ID of this processed package
        metadata (dict, optional): Additional metadata such as licenses.

    Returns:
        Filepath (as ``Path`` object) of the created package
    """
    path = Path(path)
    archive = path / (safe_filename(name) + ".zip")
    if archive.is_file():
        archive.unlink()

    base_td = tempfile.TemporaryDirectory()
    td = Path(base_td.name)

    row_labels = ["__index__"] + sorted(
        {field for obj in rows.values() for field in obj if field != "__index__"}
    )
    with open(td / "rows.csv", "w") as f:
        writer = csv.writer(f)
        for k, v in rows.items():
            writer.writerow([k] + [v.get(field) for field in row_labels[1:]])

    col_labels = ["__index__"] + sorted(
        {field for obj in cols.values() for field in obj if field != "__index__"}
    )
    with open(td / "cols.csv", "w") as f:
        writer = csv.writer(f)
        for k, v in cols.items():
            writer.writerow([k] + [v.get(field) for field in col_labels[1:]])

    array_filename = uuid.uuid4().hex + ".npy"
    np.save(td / array_filename, array, allow_pickle=False)

    resources = [
        {
            "name": "row-metadata",
            "path": "rows.csv",
            "format": "csv",
            "mediatype": "text/csv",
            "encoding": "utf-8",
            "hash": md5(td / "rows.csv"),
            "schema": {
                "array column": "row_value",
                "id column": "__index__",
                "fields": row_labels,
            },
        },
        {
            "name": "col-metadata",
            "path": "cols.csv",
            "format": "csv",
            "mediatype": "text/csv",
            "encoding": "utf-8",
            "hash": md5(td / "cols.csv"),
            "schema": {
                "array column": "col_value",
                "id column": "__index__",
                "fields": col_labels,
            },
        },
        {
            "format": "npy",
            "mediatype": "application/octet-stream",
            "path": array_filename,
            "name": "array-data",
            "profile": "data-resource",
            "hash": md5(td / array_filename),
        },
    ]

    datapackage = create_datapackage_metadata(
        name=name, resources=resources, id_=id_, metadata=metadata
    )
    with open(td / "datapackage.json", "w", encoding="utf-8") as f:
        json.dump(datapackage, f, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in td.iterdir():
            if file.is_file():
                zf.write(file, arcname=file.name)
    del base_td
    return archive
