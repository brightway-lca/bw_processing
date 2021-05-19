from .datapackage import Datapackage, load_datapackage
from .errors import NonUnique
from collections.abc import Iterable
from fs.base import FS
from typing import Union, List
import numpy as np
import pandas as pd


def _get_csv_data(
    datapackage: Union[Datapackage, FS], metadata_name: str
) -> (Datapackage, pd.DataFrame, dict, List[np.ndarray], List[int]):
    """Utility function to get CSV data from datapackage.

    Args:

        * datapackage: datapackage or `Filesystem`. Input to `load_datapackage` function.
        * metadata_name: Name identifying a CSV metadata resource in ``datapackage``

    Raises:

        * KeyError: ``metadata_name`` is not in ``datapackage``
        * ValueError: ``metadata_name`` is not CSV metadata.
        * KeyError: Resource referenced by CSV ``valid_for`` not in ``datapackage``

    Returns:

        * datapackage object
        * pandas DataFrame with CSV data
        * metadata (dict) stored with dataframe
        * list of indices arrays reference by CSV data
        * indices of arrays

    """

    dp = load_datapackage(datapackage)
    df, metadata = dp.get_resource(metadata_name)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Given metadata is not a CSV file")
    resources = [
        dp.get_resource(key + ".indices")[0][label]
        for key, label in metadata["valid_for"]
    ]
    indices = [dp._get_index(key + ".indices") for key, _ in metadata["valid_for"]]
    return dp, df, metadata, resources, indices


def reset_index(datapackage: Union[Datapackage, FS], metadata_name: str) -> Datapackage:
    """Reset the numerical indices in ``datapackage`` to sequential integers starting from zero.

    Updates the datapackage in place.

    Args:

        * datapackage: datapackage or `Filesystem`. Input to `load_datapackage` function.
        * metadata_name: Name identifying a CSV metadata resource in ``datapackage``

    Returns:

        Datapackage instance with modified data

    """
    dp, df, metadata, arrays, indices = _get_csv_data(datapackage, metadata_name)
    dp._prepare_modifications()
    unique_indices = np.unique(np.hstack(arrays))
    mapper = {k: v for k, v in zip(unique_indices, np.arange(len(unique_indices)))}

    for array in arrays:
        new = np.zeros_like(array)
        for x, y in enumerate(array):
            new[x] = mapper[y]
        array[:] = new

    dp._modified.update(indices)

    return dp


def reindex(
    datapackage: Union[Datapackage, FS],
    metadata_name: str,
    data_iterable: Iterable,
    fields: List[str] = None,
    id_field_datapackage: str = "id",
    id_field_destination: str = "id",
) -> None:
    """Use the metadata to set the integer indices in ``datapackage`` to those used in ``data_iterable``.

    Used in data exchange. Often, the integer ids provided in the data package are arbitrary, and need to be mapped to the values present in your database.

    Updates the datapackage in place.

    Args:

        * datapackage: datapackage of `Filesystem`. Input to `load_datapackage` function.
        * metadata_name: Name identifying a CSV metadata resource in ``datapackage``
        * data_iterable: Iterable which returns objects that support ``.get()``.
        * fields: Optional list of fields to use while matching
        * id_field_datapackage: String identifying the column providing an integer id in the datapackage
        * id_field_destination: String identifying the column providing an integer id in ``data_iterable``

    Raises:

        * KeyError: ``data_iterable`` is missing ``id_field_destination`` field
        * KeyError: ``metadata_name`` is missing ``id_field_datapackage`` field
        * NonUnique: Multiple objects found in ``data_iterable`` which matches fields in ``datapackage``
        * KeyError: ``metadata_name`` is not in ``datapackage``
        * KeyError: No object found in ``data_iterable`` which matches fields in ``datapackage``
        * ValueError: ``metadata_name`` is not CSV metadata.
        * ValueError: The resources given for ``metadata_name`` are not present in this ``datapackage``
        * AttributeError: ``data_iterable`` doesn't support field retrieval using ``.get()``.

    Returns:

        Datapackage instance with modified data

    """
    dp, df, metadata, arrays, indices = _get_csv_data(datapackage, metadata_name)

    if id_field_datapackage not in df.columns:
        raise KeyError(
            f"Given resource {metadata_name} is missing id column {id_field_datapackage}"
        )

    if fields is None:
        fields = sorted([x for x in df.columns if x != id_field_datapackage])

    dest_mapper = {}
    mapper = {}

    for row in data_iterable:
        key = tuple([row.get(field) for field in fields])
        index = row[id_field_destination]

        if key in dest_mapper:
            dest_mapper[key] = NonUnique
        else:
            dest_mapper[key] = index

    for i in range(len(df)):
        key = tuple([df[field][i] for field in fields])
        index = df[id_field_datapackage][i]

        if key not in dest_mapper:
            raise KeyError(f"Can't find match in `data_iterable` for {key}")
        elif dest_mapper[key] is NonUnique:
            raise NonUnique(f"No unique match in `data_iterable` for {key}")
        else:
            mapper[index] = dest_mapper[key]

    for array in arrays:
        for i in range(len(array)):
            array[i] = mapper[array[i]]

    df[id_field_datapackage] = df[id_field_datapackage].map(mapper)

    dp._modified.update(indices + [dp._get_index(metadata_name)])

    return dp
