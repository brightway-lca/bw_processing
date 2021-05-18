from bw_processing import load_datapackage, create_datapackage
from bw_processing.errors import NonUnique
from bw_processing.indexing import reset_index, reindex
from fs.zipfs import ZipFS
from fs.osfs import OSFS
from pathlib import Path
import pytest
import numpy as np
from bw_processing.constants import INDICES_DTYPE
import pandas as pd


def add_data(dp):
    data_array = np.array([(2, 7, 12)])
    indices_array = np.array([(11, 14), (11, 15), (13, 15)], dtype=INDICES_DTYPE)
    flip_array = np.array([1, 0, 0], dtype=bool)
    dp.add_persistent_vector(
        matrix="sa_matrix",
        data_array=data_array,
        name="vector",
        indices_array=indices_array,
        nrows=2,
        flip_array=flip_array,
    )

    df = pd.DataFrame(
        [
            {"id": 11, "a": 1, "c": 3, "d": 11},
            {"id": 12, "a": 2, "c": 4, "d": 11},
            {"id": 13, "a": 1, "c": 4, "d": 11},
            {"id": 14, "a": 3, "c": 5, "d": 11},
            {"id": 15, "a": 4, "c": 5, "d": 11},
            {"id": 16, "a": 4, "c": 6, "d": 11},
        ]
    ).set_index(["id"])

    dp.add_persistent_array(
        matrix="sa_matrix",
        data_array=np.arange(12).reshape((3, 4)),
        indices_array=indices_array,
        name="array",
        flip_array=flip_array,
    )
    dp.add_csv_metadata(
        dataframe=df, valid_for=[("vector", "row")], name="vector-csv-rows"
    )
    dp.add_csv_metadata(
        dataframe=df, valid_for=[("vector", "col")], name="vector-csv-cols"
    )
    dp.add_csv_metadata(
        dataframe=df,
        valid_for=[("vector", "row"), ("vector", "col")],
        name="vector-csv-both",
    )
    dp.add_csv_metadata(
        dataframe=df, valid_for=[("array", "row")], name="array-csv-rows"
    )
    dp.add_csv_metadata(
        dataframe=df,
        valid_for=[("array", "row"), ("array", "col")],
        name="array-csv-both",
    )


def original_unchanged():
    orig = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )
    array, _ = orig.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    orig = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )
    array, _ = orig.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))


def test_reset_index_multiple():
    original_unchanged()
    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))

    reset_index(fixture, "vector-csv-rows")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    # New CSV file, so indexing starts over
    reset_index(fixture, "vector-csv-cols")
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([0, 1, 1]))
    original_unchanged()


def test_reset_index_both_row_col():
    original_unchanged()
    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))

    reset_index(fixture, "vector-csv-both")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([2, 3, 3]))


def test_reset_index_pass_only_cols():
    original_unchanged()
    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    reset_index(fixture, "vector-csv-cols")
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([0, 1, 1]))
    original_unchanged()


def test_reset_index_pass_datapackage():
    original_unchanged()
    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))

    reset_index(fixture, "vector-csv-rows")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))
    original_unchanged()


def test_reset_index_pass_array():
    original_unchanged()
    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    reset_index(fixture, "array-csv-both")

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([2, 3, 3]))
    original_unchanged()


def test_reset_index_return_object():
    original_unchanged()
    fixture = OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))

    dp = reset_index(fixture, "vector-csv-rows")

    array, _ = dp.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))
    original_unchanged()


def test_reset_index_metadata_name_error():
    fixture = OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))

    with pytest.raises(KeyError):
        reset_index(fixture, "foo")

    with pytest.raises(ValueError):
        reset_index(fixture, "vector.indices")

    # expected = {0: 1, 1: 2}
    # assert _get_mapping_dictionary(fixture, "vector-csv-metadata") == expected


if __name__ == "__main__":
    dirpath = Path(__file__).parent.resolve() / "fixtures"
    dirpath.mkdir(exist_ok=True)
    (dirpath / "indexing").mkdir(exist_ok=True)

    dp = create_datapackage(
        fs=OSFS(str(dirpath / "indexing")), name="indexing-fixture", id_="fixture-i"
    )
    add_data(dp)
    dp.finalize_serialization()

    # dp = create_datapackage(
    #     fs=ZipFS(str(dirpath / "indexing" / "test-fixture.zip"), write=True),
    #     name="test-fixture",
    #     id_="fixture-42",
    # )
    # add_data(dp)
    # dp.finalize_serialization()
