import warnings
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd
from fs.base import FS
from fs.memoryfs import MemoryFS

from .datapackage import DatapackageBase, create_datapackage
from .errors import LengthMismatch
from .io_helpers import file_writer


def mask_resource(obj: Any, mask: np.ndarray) -> Any:
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        return obj[mask]
    elif isinstance(obj, Iterable):
        return [item for item, check in zip(obj, mask) if check]
    else:
        raise ValueError(f"Can't mask resource:\n\t{obj}")


def update_nrows(resource: dict, data: Any) -> dict:
    if resource["format"] == "npy":
        resource["nrows"] = len(data)
    return resource


def add_resource_suffix(metadata: dict, suffix: str) -> dict:
    """Update the ``name``, ``path``, and ``group`` values to include ``suffix``. The suffix comes after the basename but after the data type suffix (e.g. indices, data).

    Given the suffix _foo" and the metadata:

        {
            "name": "sa-data-vector-from-dict.indices",
            "path": "sa-data-vector-from-dict.indices.npy",
            "group": "sa-data-vector-from-dict",
        }

    Returns:

        {
            "name": "sa-data-vector-from-dict_foo.indices",
            "path": "sa-data-vector-from-dict_foo.indices.npy",
            "group": "sa-data-vector-from-dict_foo",
        }

    """
    if not suffix:
        return metadata

    last = metadata["name"].split(".")[-1]
    rest = metadata["name"][: -len(last) - 1]

    if last not in {"indices", "data", "distributions", "flip"}:
        raise ValueError("Can't understand resource name suffix")

    rest = metadata["name"][: -len(last) - 1]
    for key in {"name", "path", "group"}:
        metadata[key] = metadata[key].replace(rest, rest + suffix)

    return metadata


def write_data_to_fs(resource: dict, data: Any, fs: FS) -> None:
    if isinstance(fs, MemoryFS):
        return
    file_writer(
        data=data,
        fs=fs,
        resource=resource["path"],
        mimetype=resource["mediatype"],
    )


def merge_datapackages_with_mask(
    first_dp: DatapackageBase,
    first_resource_group_label: str,
    second_dp: DatapackageBase,
    second_resource_group_label: str,
    mask_array: np.ndarray,
    output_fs: Optional[FS] = None,
    metadata: Optional[dict] = None,
) -> DatapackageBase:
    """Merge two resources using a Numpy boolean mask. Returns elements from ``first_dp`` where the mask is ``True``, otherwise ``second_dp``.

    Both resource arrays, and the filter mask, must have the same length.

    Both datapackages must be static, i.e. not interfaces. This is because we don't yet have the functionality to select only some of the values in a resource group in ``matrix_utils``.

    This function currently **will not** mask or filter JSON or CSV metadata.

    Args:

        * first_dp: The datapackage from whom values will be taken when ``mask_array`` is ``True``.
        * first_resource_group_label: Label of the resource group in ``first_dp`` to select values from.
        * second_dp: The datapackage from whom values will be taken when ``mask_array`` is ``False``.
        * second_resource_group_label: Label of the resource group in ``second_dp`` to select values from.
        * mask_array: Boolean numpy array
        * output_fs: Filesystem to write new datapackage to, if any.
        * metadata: Metadata for new datapackage, if any.

    Returns:

        A `Datapackage` instance. Will write the resulting datapackage to ``output_fs`` if provided.

    """
    if first_resource_group_label == second_resource_group_label:
        add_suffix = True
        warnings.warn(
            "Adding suffixes '_true' and '_false' as resource group labels are identical"
        )
    else:
        add_suffix = False

    try:
        first_dp = first_dp.groups[first_resource_group_label]
    except KeyError:
        raise ValueError(
            f"Resource group not {first_resource_group_label} not in ``first_dp``"
        )
    try:
        second_dp = second_dp.groups[second_resource_group_label]
    except KeyError:
        raise ValueError(
            f"Resource group not {second_resource_group_label} not in ``second_dp``"
        )

    DIMENSION_ERROR = """Dimension mismatch. All array lengths must be the same, but got:\n\tFirst DP: {}\n\tSecond DP: {}\n\t Mask array: {}"""
    if not (len(first_dp.data[0]) == len(first_dp.data[0]) == len(mask_array)):
        raise LengthMismatch(
            DIMENSION_ERROR.format(
                len(first_dp.data[0]), len(first_dp.data[0]), len(mask_array)
            )
        )

    if output_fs is None:
        output_fs = MemoryFS()

    if any(resource["profile"] == "interface" for resource in first_dp.resources):
        raise ValueError("Unsupported interface found in ``first_dp``")
    if any(resource["profile"] == "interface" for resource in second_dp.resources):
        raise ValueError("Unsupported interface found in ``second_dp``")

    if metadata is None:
        metadata = {}

    dp = create_datapackage(
        fs=output_fs,
        name=metadata.pop("name", None),
        id_=metadata.pop("id_", None),
        combinatorial=metadata.pop("combinatorial", None),
        sequential=metadata.pop("sequential", None),
        seed=metadata.pop("seed", None),
        sum_intra_duplicates=metadata.pop("sum_intra_duplicates", None),
        sum_inter_duplicates=metadata.pop("sum_inter_duplicates", None),
        metadata=metadata,
    )

    dp.metadata["resources"] = [
        add_resource_suffix(resource, "_true" if add_suffix else "")
        for resource in first_dp.metadata["resources"]
    ] + [
        add_resource_suffix(resource, "_false" if add_suffix else "")
        for resource in second_dp.metadata["resources"]
    ]
    dp.data = [mask_resource(obj, mask_array) for obj in first_dp.data] + [
        mask_resource(obj, ~mask_array) for obj in second_dp.data
    ]
    for resource, data in zip(dp.resources, dp.data):
        update_nrows(resource, data)
        write_data_to_fs(resource, data, output_fs)
    try:
        dp.finalize_serialization()
    except ValueError:
        pass

    return dp
