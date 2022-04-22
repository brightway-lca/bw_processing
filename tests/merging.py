import tempfile
from pathlib import Path

import numpy as np
import pytest
from fs.memoryfs import MemoryFS
from fs.osfs import OSFS
from fs.zipfs import ZipFS

from bw_processing import (
    INDICES_DTYPE,
    DatapackageBase,
    create_datapackage,
    load_datapackage,
    merge_datapackages_with_mask,
)
from bw_processing.errors import LengthMismatch

fixture_dir = Path(__file__).parent.resolve() / "fixtures"


def test_basic_merging_functionality():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    result = merge_datapackages_with_mask(
        first_dp=first,
        first_resource_group_label="sa-data-vector",
        second_dp=second,
        second_resource_group_label="sa-data-array",
        mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
    )
    assert isinstance(result, DatapackageBase)
    assert isinstance(result.fs, MemoryFS)
    assert len(result.resources) == 5

    d, r = result.get_resource("sa-data-vector.data")

    assert r["name"] == "sa-data-vector.data"
    assert r["path"] == "sa-data-vector.data.npy"
    assert r["group"] == "sa-data-vector"
    assert r["nrows"] == 5

    assert np.allclose(d, np.array([0, 2, 4, 6, 8]))

    d, r = result.get_resource("sa-data-array.data")

    assert r["name"] == "sa-data-array.data"
    assert r["path"] == "sa-data-array.data.npy"
    assert r["group"] == "sa-data-array"
    assert r["nrows"] == 5

    assert d.shape == (5, 10)
    assert np.allclose(d[:, 0], np.array([1, 3, 5, 7, 9]) + 10)


def test_write_new_datapackage():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    with tempfile.TemporaryDirectory() as td:
        temp_fs = OSFS(td)
        result = merge_datapackages_with_mask(
            first_dp=first,
            first_resource_group_label="sa-data-vector",
            second_dp=second,
            second_resource_group_label="sa-data-array",
            mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
            output_fs=temp_fs,
        )
        result = load_datapackage(OSFS(td))

        assert isinstance(result, DatapackageBase)
        assert not isinstance(result.fs, MemoryFS)
        assert len(result.resources) == 5

        for suffix in {"indices", "data", "distributions", "flip"}:
            try:
                d, r = result.get_resource(f"sa-data-vector.{suffix}")
            except KeyError:
                continue

            assert r["name"] == f"sa-data-vector.{suffix}"
            assert r["path"] == f"sa-data-vector.{suffix}.npy"
            assert r["group"] == "sa-data-vector"
            assert r["nrows"] == 5

            if suffix == "data":
                assert np.allclose(d, np.array([0, 2, 4, 6, 8]))

            try:
                d, r = result.get_resource(f"sa-data-array.{suffix}")
            except KeyError:
                continue

            assert r["name"] == f"sa-data-array.{suffix}"
            assert r["path"] == f"sa-data-array.{suffix}.npy"
            assert r["group"] == "sa-data-array"
            assert r["nrows"] == 5

            if suffix == "data":
                assert d.shape == (5, 10)
                assert np.allclose(d[:, 0], np.array([1, 3, 5, 7, 9]) + 10)


def test_add_suffix():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_same_1.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_same_2.zip"))
    )
    with pytest.warns(UserWarning):
        result = merge_datapackages_with_mask(
            first_dp=first,
            first_resource_group_label="same",
            second_dp=second,
            second_resource_group_label="same",
            mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        )

    assert isinstance(result, DatapackageBase)
    assert len(result.resources) == 5

    for suffix in {"indices", "data", "distributions", "flip"}:
        try:
            d, r = result.get_resource(f"same_true.{suffix}")
        except KeyError:
            continue

        assert r["name"] == f"same_true.{suffix}"
        assert r["path"] == f"same_true.{suffix}.npy"
        assert r["group"] == "same_true"
        assert r["nrows"] == 5

        if suffix == "data":
            assert np.allclose(d, np.array([0, 2, 4, 6, 8]))

        try:
            d, r = result.get_resource(f"same_false.{suffix}")
        except KeyError:
            continue

        assert r["name"] == f"same_false.{suffix}"
        assert r["path"] == f"same_false.{suffix}.npy"
        assert r["group"] == "same_false"
        assert r["nrows"] == 5

        if suffix == "data":
            assert d.shape == (5, 10)
            assert np.allclose(d[:, 0], np.array([1, 3, 5, 7, 9]) + 10)


def test_wrong_resource_group_name():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    with pytest.raises(ValueError):
        merge_datapackages_with_mask(
            first_dp=first,
            first_resource_group_label="wrong",
            second_dp=second,
            second_resource_group_label="sa-data-array",
            mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        )
    with pytest.raises(ValueError):
        merge_datapackages_with_mask(
            first_dp=first,
            first_resource_group_label="sa-data-vector",
            second_dp=second,
            second_resource_group_label="wrong",
            mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        )


def test_shape_mismatch_data():
    dp1 = create_datapackage()
    data_array = np.arange(10)
    indices_array = np.array(
        [(x, y) for x, y in zip(range(10), range(10, 20))], dtype=INDICES_DTYPE
    )
    dp1.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector",
        indices_array=indices_array,
    )

    dp2 = create_datapackage()
    data_array = np.arange(5)
    indices_array = np.array(
        [(x, y) for x, y in zip(range(5), range(10, 15))], dtype=INDICES_DTYPE
    )
    dp2.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector2",
        indices_array=indices_array,
    )
    with pytest.raises(LengthMismatch):
        merge_datapackages_with_mask(
            first_dp=dp1,
            first_resource_group_label="sa-data-vector",
            second_dp=dp2,
            second_resource_group_label="sa-data-vector2",
            mask_array=np.zeros((5,), dtype=bool),
        )


def test_shape_mismatch_mask():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    with pytest.raises(LengthMismatch):
        merge_datapackages_with_mask(
            first_dp=first,
            first_resource_group_label="sa-data-vector",
            second_dp=second,
            second_resource_group_label="sa-data-array",
            mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        )


def test_new_metadata():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    result = merge_datapackages_with_mask(
        first_dp=first,
        first_resource_group_label="sa-data-vector",
        second_dp=second,
        second_resource_group_label="sa-data-array",
        mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        metadata={
            "name": "something something",
            "id_": "danger zone",
            "combinatorial": True,
            "sequential": False,
            "seed": 2000,
            "foo bar baz": True,
        },
    )

    assert result.metadata["name"] == "something_something"
    assert result.metadata["id"] == "danger zone"
    assert result.metadata["combinatorial"]
    assert not result.metadata["sequential"]
    assert result.metadata["seed"] == 2000
    assert result.metadata["foo bar baz"]


def test_default_metadata():
    first = load_datapackage(ZipFS(str(fixture_dir / "merging" / "merging_first.zip")))
    second = load_datapackage(
        ZipFS(str(fixture_dir / "merging" / "merging_second.zip"))
    )
    result = merge_datapackages_with_mask(
        first_dp=first,
        first_resource_group_label="sa-data-vector",
        second_dp=second,
        second_resource_group_label="sa-data-array",
        mask_array=np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
    )

    assert result.metadata["name"]
    assert result.metadata["id"]
    assert not result.metadata["combinatorial"]
    assert not result.metadata["sequential"]
    assert not result.metadata["seed"]


def test_interface_error():
    dp1 = create_datapackage()
    data_array = np.arange(10)
    indices_array = np.array(
        [(x, y) for x, y in zip(range(10), range(10, 20))], dtype=INDICES_DTYPE
    )
    dp1.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector",
        indices_array=indices_array,
    )

    class Dummy:
        pass

    dp2 = create_datapackage()
    indices_array = np.array(
        [(x, y) for x, y in zip(range(10), range(10, 20))], dtype=INDICES_DTYPE
    )
    dp2.add_dynamic_vector(
        interface=Dummy(),
        indices_array=indices_array,
        matrix="sa_matrix",
        name="sa-vector-interface",
    )
    with pytest.raises(ValueError):
        merge_datapackages_with_mask(
            first_dp=dp1,
            first_resource_group_label="sa-data-vector",
            second_dp=dp2,
            second_resource_group_label="sa-vector-interface",
            mask_array=np.zeros((10,), dtype=bool),
        )


def create_fixtures():
    fixture_dir.mkdir(exist_ok=True)
    (fixture_dir / "merging").mkdir(exist_ok=True)

    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "merging" / "merging_first.zip"), write=True),
        name="merging-fixture",
        id_="fixture-42",
    )
    data_array = np.arange(10)
    indices_array = np.array(
        [(x, y) for x, y in zip(range(10), range(10, 20))], dtype=INDICES_DTYPE
    )
    flip_array = np.array([x % 2 for x in range(10)], dtype=bool)
    dp.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="sa-data-vector",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "merging" / "merging_second.zip"), write=True),
        name="merging-fixture",
        id_="fixture-42",
    )
    data_array = np.repeat(data_array + 10, 10).reshape((10, 10))
    dp.add_persistent_array(
        matrix="sa_matrix",
        data_array=data_array,
        indices_array=indices_array,
        name="sa-data-array",
        flip_array=np.array([0] * 10, dtype=bool),
    )
    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "merging" / "merging_same_1.zip"), write=True),
        name="merging-fixture",
        id_="fixture-42",
    )
    data_array = np.arange(10)
    indices_array = np.array(
        [(x, y) for x, y in zip(range(10), range(10, 20))], dtype=INDICES_DTYPE
    )
    flip_array = np.array([x % 2 for x in range(10)], dtype=bool)
    dp.add_persistent_vector(
        matrix="matrix",
        data_array=data_array,
        name="same",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.finalize_serialization()

    dp = create_datapackage(
        fs=ZipFS(str(fixture_dir / "merging" / "merging_same_2.zip"), write=True),
        name="merging-fixture",
        id_="fixture-42",
    )
    data_array = np.repeat(data_array + 10, 10).reshape((10, 10))
    dp.add_persistent_array(
        matrix="matrix",
        data_array=data_array,
        indices_array=indices_array,
        name="same",
        flip_array=np.array([0] * 10, dtype=bool),
    )
    dp.finalize_serialization()


if __name__ == "__main__":
    create_fixtures()
