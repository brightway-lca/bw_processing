from .datapackage import Datapackage, load_datapackage
from .errors import NonUnique
from collections.abc import Iterable
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd


def _get_csv_data(
    datapackage: Union[Datapackage, Path, str], metadata_name: str
) -> (Datapackage, pd.DataFrame, dict, List[np.ndarray]):
    """Utility function to get CSV data from datapackage.

    Args:
        datapackage: datapackage, or location of one. Input to `load_datapackage` function.
        metadata_name: Name identifying a CSV metadata resource in ``datapackage``

    Raises:
        KeyError: ``metadata_name`` is not in ``datapackage``
        ValueError: ``metadata_name`` is not CSV metadata.
        KeyError: Resource referenced by CSV ``valid_for`` not in ``datapackage``

    Returns:
        datapackage object
        pandas DataFrame with CSV data
        metadata (dict) stored with dataframe
        list of indices arrays reference by CSV data

    """
    dp = load_datapackage(datapackage)
    df, metadata = dp.get_resource(metadata_name)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Given metadata is not a CSV file")
    indices = [
        dp.get_resource(key + ".indices")[0][label]
        for key, label in metadata["valid_for"]
    ]
    return dp, df, metadata, indices


def reset_index(
    datapackage: Union[Datapackage, Path, str], metadata_name: str
) -> Datapackage:
    """Reset the numerical indices in ``datapackage`` to sequential integers starting from zero.

    Updates the datapackage in place.

    Args:
        datapackage: datapackage, or location of one. Input to `load_datapackage` function.
        metadata_name: Name identifying a CSV metadata resource in ``datapackage``

    Returns:
        Datapackage instance with modified data

    """
    dp, df, metadata, indices = _get_csv_data(datapackage, metadata_name)
    unique_indices = np.unique(np.hstack(indices))
    mapper = {k: v for k, v in zip(unique_indices, np.arange(len(unique_indices)))}

    for array in indices:
        new = np.zeros_like(array)
        for x, y in enumerate(array):
            new[x] = mapper[y]
        array[:] = new

    return dp


def reindex(
    datapackage: Union[Datapackage, Path, str],
    metadata_name: str,
    data_iterable: Iterable,
    fields: List[str] = None,
) -> None:
    """Use the metadata to set the integer indices in ``datapackage`` to those used in ``data_iterable``.

    Used in data exchange. Often, the integer ids provided in the data package are arbitrary, and need to be mapped to the values present in your database.

    Updates the datapackage in place.

    Args:
        datapackage: datapackage, or location of one. Input to `load_datapackage` function.
        metadata_name: Name identifying a CSV metadata resource in ``datapackage``
        data_iterable: Iterable which returns objects with ``id`` values and support ``.get()``.
        fields: Optional list of fields to use while matching

    Raises:
        NonUnique: Multiple objects found in ``data_iterable`` which matches fields in ``datapackage``
        ValueError: No object found in ``data_iterable`` which matches fields in ``datapackage``
        KeyError: ``metadata_name`` is not in ``datapackage``
        ValueError: ``metadata_name`` is not CSV metadata.
        ValueError: The resources given for ``metadata_name`` are not present in this ``datapackage``
        ValueError: ``data_iterable`` is not an iterable, or doesn't support field retrieval using ``.get()``.

    Returns:
        None

    """
    pass
