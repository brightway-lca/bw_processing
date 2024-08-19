import platform
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fsspec.implementations.ftp import FTPFileSystem
from fsspec.implementations.zip import ZipFileSystem
from morefs.dict import DictFS

from bw_processing import INDICES_DTYPE, create_datapackage, load_datapackage
from bw_processing.io_helpers import generic_directory_filesystem

_windows = platform.system() == "Windows"


class Dummy:
    pass


def add_data(dp, indices_32bit=False):
    from_dicts = [
        {
            "row": 0,
            "col": 1,
            "flip": True,
            "amount": 3.3,
            "uncertainty_type": 2,
            "loc": 2.7,
            "scale": 3.9,
        },
        {
            "row": 5,
            "col": 6,
            "flip": False,
            "amount": 8.3,
            "uncertainty_type": 7,
            "loc": 7.7,
            "scale": 8.9,
        },
    ]
    dp.add_persistent_vector_from_iterator(
        matrix="sa_matrix",
        name="sa-data-vector-from-dict",
        dict_iterator=from_dicts,
        foo="bar",
    )

    data_array = np.array([2, 7, 12])
    if indices_32bit:
        indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=[("row", np.int32), ("col", np.int32)])
    else:
        indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 0, 0], dtype=bool)
    dp.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector",
        indices_array=indices_array,
        nrows=2,  # Should be 3 - fixed automatically
        flip_array=flip_array,
    )

    json_data = [{"a": "b"}, 1, True]
    json_parameters = ["a", "foo"]
    df = pd.DataFrame(
        [
            {"id": 1, "a": 1, "c": 3, "d": 11},
            {"id": 2, "a": 2, "c": 4, "d": 11},
            {"id": 3, "a": 1, "c": 4, "d": 11},
        ]
    ).set_index(["id"])

    dp.add_persistent_array(
        matrix="sa_matrix",
        data_array=np.arange(12).reshape((3, 4)),
        indices_array=indices_array,
        name="sa-data-array",
        flip_array=flip_array,
    )

    dp.add_dynamic_vector(
        interface=Dummy(),
        indices_array=indices_array,
        matrix="sa_matrix",
        name="sa-vector-interface",
    )

    dp.add_dynamic_array(
        interface=Dummy(),
        matrix="sa_matrix",
        name="sa-array-interface",
        indices_array=indices_array,
    )
    dp.add_csv_metadata(
        dataframe=df,
        valid_for=[("sa-data-vector", "rows")],
        name="sa-data-vector-csv-metadata",
    )
    dp.add_json_metadata(
        data=json_data, valid_for="sa-data-array", name="sa-data-array-json-metadata"
    )
    dp.add_json_metadata(
        data=json_parameters,
        valid_for="sa-data-array",
        name="sa-data-array-json-parameters",
    )


def check_data(dp):
    assert len(dp.resources) == len(dp.data) == 17
    d, _ = dp.get_resource("sa-data-array-json-parameters")
    assert d == ["a", "foo"]

    d, _ = dp.get_resource("sa-data-array-json-metadata")
    assert d == [{"a": "b"}, 1, True]

    d, _ = dp.get_resource("sa-data-vector-csv-metadata")
    assert d["a"].sum() == 4

    # d, _ = dp.get_resource("presamples-indices")
    # print(d)
    # assert d["row_value"].sum() == 6

    # d, _ = dp.get_resource("sa-data")
    # assert d["col_value"].sum() == 7


def check_metadata(dp, as_tuples=True):
    expected = [
        {
            "category": "vector",
            "foo": "bar",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector-from-dict.indices",
            "matrix": "sa_matrix",
            "kind": "indices",
            "nrows": 2,
            "path": "sa-data-vector-from-dict.indices.npy",
            "group": "sa-data-vector-from-dict",
        },
        {
            "category": "vector",
            "foo": "bar",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector-from-dict.data",
            "matrix": "sa_matrix",
            "kind": "data",
            "nrows": 2,
            "path": "sa-data-vector-from-dict.data.npy",
            "group": "sa-data-vector-from-dict",
        },
        {
            "category": "vector",
            "foo": "bar",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector-from-dict.distributions",
            "matrix": "sa_matrix",
            "nrows": 2,
            "kind": "distributions",
            "path": "sa-data-vector-from-dict.distributions.npy",
            "group": "sa-data-vector-from-dict",
        },
        {
            "category": "vector",
            "foo": "bar",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector-from-dict.flip",
            "matrix": "sa_matrix",
            "kind": "flip",
            "nrows": 2,
            "path": "sa-data-vector-from-dict.flip.npy",
            "group": "sa-data-vector-from-dict",
        },
        {
            "category": "vector",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector.indices",
            "matrix": "sa_matrix",
            "kind": "indices",
            "nrows": 3,
            "path": "sa-data-vector.indices.npy",
            "group": "sa-data-vector",
        },
        {
            "category": "vector",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector.data",
            "matrix": "sa_matrix",
            "kind": "data",
            "nrows": 3,
            "path": "sa-data-vector.data.npy",
            "group": "sa-data-vector",
        },
        {
            "category": "vector",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-vector.flip",
            "matrix": "sa_matrix",
            "kind": "flip",
            "nrows": 3,
            "path": "sa-data-vector.flip.npy",
            "group": "sa-data-vector",
        },
        {
            "category": "array",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-array.indices",
            "matrix": "sa_matrix",
            "kind": "indices",
            "nrows": 3,
            "path": "sa-data-array.indices.npy",
            "group": "sa-data-array",
        },
        {
            "category": "array",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-array.data",
            "matrix": "sa_matrix",
            "kind": "data",
            "nrows": 3,
            "path": "sa-data-array.data.npy",
            "group": "sa-data-array",
        },
        {
            "category": "array",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data-array.flip",
            "matrix": "sa_matrix",
            "kind": "flip",
            "nrows": 3,
            "path": "sa-data-array.flip.npy",
            "group": "sa-data-array",
        },
        {
            "category": "vector",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-vector-interface.indices",
            "matrix": "sa_matrix",
            "kind": "indices",
            "nrows": 3,
            "path": "sa-vector-interface.indices.npy",
            "group": "sa-vector-interface",
        },
        {
            "category": "vector",
            "group": "sa-vector-interface",
            "kind": "data",
            "matrix": "sa_matrix",
            "name": "sa-vector-interface.data",
            "nrows": 3,
            "profile": "interface",
        },
        {
            "category": "array",
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-array-interface.indices",
            "matrix": "sa_matrix",
            "kind": "indices",
            "nrows": 3,
            "path": "sa-array-interface.indices.npy",
            "group": "sa-array-interface",
        },
        {
            "category": "array",
            "group": "sa-array-interface",
            "kind": "data",
            "matrix": "sa_matrix",
            "name": "sa-array-interface.data",
            "nrows": 3,
            "profile": "interface",
        },
        {
            "profile": "data-resource",
            "mediatype": "text/csv",
            "path": "sa-data-vector-csv-metadata.csv",
            "name": "sa-data-vector-csv-metadata",
            "valid_for": [("sa-data-vector", "rows")],
        },
        {
            "profile": "data-resource",
            "mediatype": "application/json",
            "path": "sa-data-array-json-metadata.json",
            "name": "sa-data-array-json-metadata",
            "valid_for": "sa-data-array",
        },
        {
            "profile": "data-resource",
            "mediatype": "application/json",
            "path": "sa-data-array-json-parameters.json",
            "name": "sa-data-array-json-parameters",
            "valid_for": "sa-data-array",
        },
    ]
    expected_as_list = deepcopy(expected)
    expected_as_list[14]["valid_for"][0] = list(expected_as_list[14]["valid_for"][0])
    if as_tuples:
        assert dp.metadata["resources"] == expected
    else:
        assert dp.metadata["resources"] == expected_as_list
    assert dp.metadata["created"].endswith("Z")
    assert isinstance(dp.metadata["licenses"], list)
    expected = {
        "profile": "data-package",
        "name": "test-fixture",
        "id": "fixture-42",
        "combinatorial": False,
        "sequential": False,
        "seed": None,
        "64_bit_indices": True,
        "sum_intra_duplicates": True,
        "sum_inter_duplicates": False,
    }
    for k, v in expected.items():
        assert dp.metadata[k] == v


def test_integration_test_in_memory():
    dp = create_datapackage(fs=None, name="test-fixture", id_="fixture-42")
    assert isinstance(dp.fs, DictFS)
    add_data(dp)

    check_metadata(dp)
    check_data(dp)


def test_integration_test_directory():
    dp = load_datapackage(
        fs_or_obj=generic_directory_filesystem(
            dirpath=Path(__file__).parent.resolve() / "fixtures" / "tfd"
        )
    )

    check_metadata(dp, False)
    check_data(dp)


@pytest.mark.slow
def test_integration_test_ftp():
    dp = load_datapackage(fs_or_obj=FTPFileSystem(host="ftp://brightway.dev/tfd/"))
    check_metadata(dp, False)
    check_data(dp)


# TBD: https://s3fs.readthedocs.io/en/latest/
# @pytest.mark.slow
# def test_integration_test_s3():
#     try:
#         import fs_s3fs
#         from botocore.exceptions import NoCredentialsError
#     except ImportError:
#         raise ImportError(
#             "https://github.com/PyFilesystem/s3fs must be installed for this test."
#         )
#     try:
#         dp = load_datapackage(fs_or_obj=open_fs("s3://bwprocessing"))
#         check_metadata(dp, False)
#         check_data(dp)
#     except NoCredentialsError:
#         raise NoCredentialsError(
#             "Supply AWS credentials (https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)"
#         )


@pytest.mark.skipif(_windows, reason="Permission errors on Windows CI")
def test_integration_test_fs_temp_directory():
    with tempfile.TemporaryDirectory() as td:
        dp = create_datapackage(
            fs=generic_directory_filesystem(dirpath=Path(td)), name="test-fixture", id_="fixture-42"
        )
        add_data(dp)
        dp.finalize_serialization()

        check_metadata(dp)
        check_data(dp)

        loaded = load_datapackage(generic_directory_filesystem(dirpath=Path(td)))

        check_metadata(loaded, False)
        check_data(loaded)


@pytest.mark.skipif(_windows, reason="Permission errors on Windows CI")
def test_integration_test_new_zipfile():
    with tempfile.TemporaryDirectory() as td:
        dp = create_datapackage(
            fs=ZipFileSystem(Path(td) / "foo.zip", mode="w"),
            name="test-fixture",
            id_="fixture-42",
        )
        add_data(dp)
        dp.finalize_serialization()

        check_metadata(dp)
        check_data(dp)

        loaded = load_datapackage(ZipFileSystem(Path(td) / "foo.zip", mode="r"))

        check_metadata(loaded, False)
        check_data(loaded)


def test_integration_test_fixture_zipfile():
    loaded = load_datapackage(
        ZipFileSystem(
            Path(__file__).parent.resolve() / "fixtures" / "test-fixture.zip",
            mode="r",
        )
    )

    check_metadata(loaded, False)
    check_data(loaded)


if __name__ == "__main__":
    # Create the test fixtures

    dirpath = Path(__file__).parent.resolve() / "fixtures"
    dirpath.mkdir(exist_ok=True)
    (dirpath / "tfd").mkdir(exist_ok=True)

    dp = create_datapackage(
        fs=generic_directory_filesystem(dirpath=dirpath / "tfd"),
        name="test-fixture",
        id_="fixture-42",
    )
    add_data(dp)
    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFileSystem(dirpath / "test-fixture.zip", mode="w"),
        name="test-fixture",
        id_="fixture-42",
    )
    add_data(dp)
    dp.finalize_serialization()

    # 32 bit fixture for backwards compatibility
    dp = create_datapackage(
        fs=ZipFileSystem(dirpath / "test-fixture-32bit.zip", mode="w"),
        name="test-fixture-32bit",
        id_="fixture-32",
    )
    del dp.metadata["64_bit_indices"]
    add_data(dp, indices_32bit=True)
    dp.finalize_serialization()
