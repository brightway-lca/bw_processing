from pathlib import Path

import numpy as np
import pytest
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from bw_processing import (
    INDICES_DTYPE,
    UNCERTAINTY_DTYPE,
    create_datapackage,
    load_datapackage,
)

dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_metadata_is_the_same_object():
    dp = load_datapackage(fs_or_obj=ZipFS(str(dirpath / "test-fixture.zip")))
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")

    for k, v in fdp.metadata.items():
        if k != "resources":
            assert id(v) == id(dp.metadata[k])

    for resource in fdp.resources:
        assert any(obj for obj in dp.resources if obj is resource)


def test_indexer_is_the_same_object():
    dp = load_datapackage(fs_or_obj=ZipFS(str(dirpath / "test-fixture.zip")))
    dp.indexer = lambda x: False
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")
    assert fdp.indexer is dp.indexer


def test_data_is_the_same_object_when_not_proxy():
    dp = load_datapackage(fs_or_obj=ZipFS(str(dirpath / "test-fixture.zip")))
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")

    arr1, _ = dp.get_resource("sa-data-array.data")

    assert "sa-data-array.data" not in fdp._cache
    arr2, _ = fdp.get_resource("sa-data-array.data")

    assert np.allclose(arr1, arr2)
    assert arr1 is arr2
    assert np.shares_memory(arr1, arr2)


def test_data_is_readable_multiple_times_when_proxy_zipfs():
    dp = load_datapackage(
        fs_or_obj=ZipFS(str(dirpath / "test-fixture.zip")), proxy=True
    )
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")

    arr1, _ = dp.get_resource("sa-data-array.data")

    assert "sa-data-array.data" not in fdp._cache
    arr2, _ = fdp.get_resource("sa-data-array.data")
    assert "sa-data-array.data" in fdp._cache

    assert np.allclose(arr1, arr2)
    assert arr1.base is not arr2
    assert arr2.base is not arr1
    assert not np.shares_memory(arr1, arr2)


def test_data_is_readable_multiple_times_when_proxy_directory():
    dp = load_datapackage(fs_or_obj=OSFS(str(dirpath / "tfd")), proxy=True)
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")

    arr1, _ = dp.get_resource("sa-data-array.data")

    assert "sa-data-array.data" not in fdp._cache
    arr2, _ = fdp.get_resource("sa-data-array.data")
    assert "sa-data-array.data" in fdp._cache

    assert np.allclose(arr1, arr2)
    assert arr1.base is not arr2
    assert arr2.base is not arr1
    assert not np.shares_memory(arr1, arr2)


def test_fdp_can_load_proxy_first():
    dp = load_datapackage(
        fs_or_obj=ZipFS(str(dirpath / "test-fixture.zip")), proxy=True
    )
    fdp = dp.filter_by_attribute("matrix", "sa_matrix")

    assert "sa-data-array.data" not in fdp._cache
    arr2, _ = fdp.get_resource("sa-data-array.data")


@pytest.fixture
def erg():
    dp = create_datapackage(
        fs=None, name="frg-fixture", id_="something something danger zone"
    )

    data_array = np.arange(3)
    indices_array = np.array([(0, 1), (2, 3), (4, 5)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 0, 1], dtype=bool)
    distributions_array = np.array(
        [
            (5, 1, 2, 3, 4, 5, False),
            (4, 1, 2, 3, 4, 5, False),
            (0, 1, 2, 3, 4, 5, False),
        ],
        dtype=UNCERTAINTY_DTYPE,
    )

    dp.add_persistent_vector(
        matrix="one",
        data_array=data_array,
        name="first",
        indices_array=indices_array,
        distributions_array=distributions_array,
        nrows=3,
        flip_array=flip_array,
    )
    dp.add_persistent_array(
        matrix="two",
        data_array=np.arange(12).reshape((3, 4)),
        indices_array=indices_array,
        name="second",
    )
    return dp


def test_exclude_resource_group(erg):
    assert len(erg.resources) == 6
    result = erg.exclude({"group": "first"})
    assert len(result.resources) == 2
    assert {obj["name"] for obj in result.resources} == {
        "second.data",
        "second.indices",
    }


def test_exclude_resource_group_kind(erg):
    assert len(erg.resources) == 6
    result = erg.exclude({"group": "first", "kind": "distributions"})
    assert len(result.resources) == 5
    assert {obj["name"] for obj in result.resources} == {
        "second.data",
        "second.indices",
        "first.data",
        "first.indices",
        "first.flip",
    }
    assert not any(obj.dtype == UNCERTAINTY_DTYPE for obj in result.data)
