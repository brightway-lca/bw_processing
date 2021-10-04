from bw_processing import load_datapackage
from fs.osfs import OSFS
from fs.zipfs import ZipFS
from pathlib import Path
import numpy as np
import pytest

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
