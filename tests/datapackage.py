import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fs.memoryfs import MemoryFS
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from bw_processing import create_datapackage, load_datapackage, simple_graph
from bw_processing.constants import INDICES_DTYPE, UNCERTAINTY_DTYPE
from bw_processing.errors import (
    NonUnique,
    PotentialInconsistency,
    ShapeMismatch,
    WrongDatatype,
)

dirpath = Path(__file__).parent.resolve() / "fixtures"


class Dummy:
    pass


def add_data(dp):
    data_array = np.array([2, 7, 12])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 0, 0], dtype=bool)
    dp.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector",
        indices_array=indices_array,
        nrows=2,
        flip_array=flip_array,
    )

    json_data = [{"a": "b"}, 1, True]
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


def copy_fixture(fixture_name, dest):
    source = Path(__file__).parent.resolve() / "fixtures" / fixture_name
    for fp in source.iterdir():
        shutil.copy(fp, dest / fp.name)


def test_group_ordering_consistent():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    assert list(dp.groups) == [
        "sa-data-vector-from-dict",
        "sa-data-vector",
        "sa-data-array",
        "sa-vector-interface",
        "sa-array-interface",
    ]


def test_add_resource_with_same_name():
    dp = create_datapackage()
    add_data(dp)

    data_array = np.array([2, 7, 12])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(NonUnique):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_save_modifications(tmp_path):
    copy_fixture("tfd", tmp_path)
    dp = load_datapackage(OSFS(str(tmp_path)))

    assert dp.resources[1]["name"] == "sa-data-vector-from-dict.data"
    assert np.allclose(dp.data[1], [3.3, 8.3])

    dp.data[1][:] = 42
    dp._modified = [1]
    dp.write_modified()

    assert np.allclose(dp.data[1], 42)
    assert not dp._modified

    dp = load_datapackage(OSFS(str(tmp_path)))
    assert np.allclose(dp.data[1], 42)


def test_del_resource_filesystem(tmp_path):
    copy_fixture("tfd", tmp_path)
    dp = load_datapackage(OSFS(str(tmp_path)))
    reference_length = len(dp)
    assert "sa-vector-interface.indices.npy" in [o.name for o in tmp_path.iterdir()]
    dp.del_resource("sa-vector-interface.indices")
    assert "sa-vector-interface.indices.npy" not in [o.name for o in tmp_path.iterdir()]
    assert len(dp) == reference_length - 1
    assert len(dp.data) == reference_length - 1
    assert len(dp.metadata["resources"]) == reference_length - 1
    assert len(dp.resources) == reference_length - 1


def test_del_resource_in_memory():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    reference_length = len(dp)
    assert "sa-vector-interface.indices" in [o["name"] for o in dp.resources]
    dp.del_resource("sa-vector-interface.indices")
    assert "sa-vector-interface.indices" not in [o["name"] for o in dp.resources]
    assert len(dp) == reference_length - 1
    assert len(dp.data) == reference_length - 1
    assert len(dp.metadata["resources"]) == reference_length - 1
    assert len(dp.resources) == reference_length - 1


def test_del_resource_error_modifications(tmp_path):
    copy_fixture("tfd", tmp_path)
    dp = load_datapackage(OSFS(str(tmp_path)))
    dp._modified = [1]
    with pytest.raises(PotentialInconsistency):
        dp.del_resource(1)


def test_del_resource_group_filesystem(tmp_path):
    copy_fixture("tfd", tmp_path)
    dp = load_datapackage(OSFS(str(tmp_path)))

    reference_length = len(dp)
    assert "sa-data-vector.indices.npy" in [o.name for o in tmp_path.iterdir()]
    dp.del_resource_group("sa-data-vector")
    assert "sa-data-vector.indices.npy" not in [o.name for o in tmp_path.iterdir()]
    assert len(dp) == reference_length - 3
    assert len(dp.data) == reference_length - 3
    assert len(dp.metadata["resources"]) == reference_length - 3
    assert len(dp.resources) == reference_length - 3


def test_del_resource_group_in_memory():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    reference_length = len(dp)
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    dp.del_resource_group("sa-data-vector")
    assert "sa-data-vector.indices" not in [o["name"] for o in dp.resources]
    assert len(dp) == reference_length - 3
    assert len(dp.data) == reference_length - 3
    assert len(dp.metadata["resources"]) == reference_length - 3
    assert len(dp.resources) == reference_length - 3


def test_del_resource_group_error_modifications(tmp_path):
    copy_fixture("tfd", tmp_path)
    dp = load_datapackage(OSFS(str(tmp_path)))
    dp._modified = [1]
    with pytest.raises(PotentialInconsistency):
        dp.del_resource_group("sa-vector-interface")


def test_exclude_basic():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    reference_length = len(dp)
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    ndp = dp.exclude({"group": "sa-data-vector"})
    assert ndp is not dp
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    assert "sa-data-vector.indices" not in [o["name"] for o in ndp.resources]
    assert len(ndp) == reference_length - 3
    assert len(ndp.data) == reference_length - 3
    assert len(ndp.metadata["resources"]) == reference_length - 3
    assert len(ndp.resources) == reference_length - 3


def test_exclude_no_match():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    reference_length = len(dp)
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    ndp = dp.exclude({"foo": "bar"})
    assert ndp is not dp
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    assert "sa-data-vector.indices" in [o["name"] for o in ndp.resources]
    assert len(ndp) == reference_length


def test_exclude_multiple_filters():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    reference_length = len(dp)
    assert "sa-array-interface.indices" in [o["name"] for o in dp.resources]
    ndp = dp.exclude({"group": "sa-array-interface", "matrix": "sa_matrix"})
    assert ndp is not dp
    assert "sa-array-interface.indices" in [o["name"] for o in dp.resources]
    assert "sa-array-interface.indices" not in [o["name"] for o in ndp.resources]
    assert len(ndp) == reference_length - 2
    assert len(ndp.data) == reference_length - 2
    assert len(ndp.metadata["resources"]) == reference_length - 2
    assert len(ndp.resources) == reference_length - 2


def test_exclude_multiple_matrix():
    dp = create_datapackage()
    add_data(dp)
    assert isinstance(dp.fs, MemoryFS)

    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    ndp = dp.exclude({"matrix": "sa_matrix"})
    assert ndp is not dp
    assert "sa-data-vector.indices" in [o["name"] for o in dp.resources]
    assert "sa-data-vector.indices" not in [o["name"] for o in ndp.resources]
    assert len(ndp) == 2
    assert len(ndp.data) == 2
    assert len(ndp.metadata["resources"]) == 2
    assert len(ndp.resources) == 2


def test_add_persistent_vector_data_shapemismatch_ndimensions():
    dp = create_datapackage()
    data_array = np.array([[2, 7, 12], [4, 5, 15]])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_add_persistent_vector_data_shapemismatch_nrows():
    dp = create_datapackage()
    data_array = np.array([2, 7, 12, 15])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_add_persistent_vector_distributions_shapemismatch():
    dp = create_datapackage()
    distributions_array = np.array(
        [
            (3, 1.3, 2.5, np.NaN, np.NaN, np.NaN, False),
            (0, 1.3, 2.5, np.NaN, np.NaN, np.NaN, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            distributions_array=distributions_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_add_persistent_vector_flip_dtype():
    dp = create_datapackage()
    data_array = np.array([2, 7, 12])
    flip_array = np.array([0, 1, 0])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(WrongDatatype):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_persistent_vector_flip_shapemistmatch():
    dp = create_datapackage()
    data_array = np.array([2, 7, 12])
    flip_array = np.array([0, 1, 0, 1], dtype=bool)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=ShapeMismatch)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_persistent_array_data_shapemismatch_ndimensions():
    dp = create_datapackage()
    data_array = np.array([2, 7, 12])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_array(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_add_persistent_array_data_shapemismatch_nrows():
    dp = create_datapackage()
    data_array = np.arange(12).reshape(4, 3)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_array(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            indices_array=indices_array,
        )


def test_add_persistent_array_flip_dtype():
    dp = create_datapackage()
    data_array = np.arange(12).reshape(3, 4)
    flip_array = np.array([0, 1, 0])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(WrongDatatype):
        dp.add_persistent_array(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_persistent_array_flip_shapemistmatch():
    dp = create_datapackage()
    data_array = np.arange(12).reshape(3, 4)
    flip_array = np.array([0, 1, 0, 1], dtype=bool)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=ShapeMismatch)
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_array(
            matrix="sa_matrix",
            data_array=data_array,
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_dynamic_array_flip_dtype():
    dp = create_datapackage()
    flip_array = np.array([0, 1, 0])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(WrongDatatype):
        dp.add_dynamic_array(
            matrix="sa_matrix",
            interface=Dummy(),
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_dynamic_array_flip_shapemistmatch():
    dp = create_datapackage()
    flip_array = np.array([0, 1, 0, 1], dtype=bool)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=ShapeMismatch)
    with pytest.raises(ShapeMismatch):
        dp.add_dynamic_array(
            matrix="sa_matrix",
            interface=Dummy(),
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_dynamic_vector_flip_dtype():
    dp = create_datapackage()
    flip_array = np.array([0, 1, 0])
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
    with pytest.raises(WrongDatatype):
        dp.add_dynamic_vector(
            matrix="sa_matrix",
            interface=Dummy(),
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_add_dynamic_vector_flip_shapemistmatch():
    dp = create_datapackage()
    flip_array = np.array([0, 1, 0, 1], dtype=bool)
    indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=ShapeMismatch)
    with pytest.raises(ShapeMismatch):
        dp.add_dynamic_vector(
            matrix="sa_matrix",
            interface=Dummy(),
            name="sa-data-vector",
            flip_array=flip_array,
            indices_array=indices_array,
        )


def test_simple_graph():
    data = {
        "some_matrix": [
            (1, 2, 3.14),
            (4, 5, 17, True),
            (8, 9, 11.11, False)
        ],
        "a different life": [
            (99, 100, 101.234, False, "ignored"),
            (1001, 1002, 777, 1),
        ]
    }
    expected_resource = {
        "category": "vector",
        "profile": "data-resource",
        "format": "npy",
        "mediatype": "application/octet-stream",
        "name": "some_matrix-data.flip",
        "matrix": "some_matrix",
        "kind": "flip",
        "nrows": 3,
        "path": "some_matrix-data.flip.npy",
        "group": "some_matrix-data",
    }
    dp = simple_graph(data, name="foo", sequential=True)
    assert dp.metadata['name'] == 'foo'
    assert dp.metadata['sequential']
    assert len(dp.metadata["resources"]) == 6
    assert expected_resource in dp.metadata["resources"]

    d, _ = dp.get_resource("some_matrix-data.flip")
    assert np.allclose(d, np.array([False, True, False]))
    d, _ = dp.get_resource("some_matrix-data.indices")
    assert d['row'].sum() == 13
    assert d['col'].sum() == 16
    d, _ = dp.get_resource("some_matrix-data.data")
    assert np.allclose(d, np.array([3.14, 17, 11.11]))

    d, _ = dp.get_resource("a different life-data.flip")
    assert np.allclose(d, np.array([False, True]))
    d, _ = dp.get_resource("a different life-data.indices")
    assert d['row'].sum() == 1100
    assert d['col'].sum() == 1102
    d, _ = dp.get_resource("a different life-data.data")
    assert np.allclose(d, np.array([101.234, 777]))
