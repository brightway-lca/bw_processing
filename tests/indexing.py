from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fs.osfs import OSFS

from bw_processing import create_datapackage, load_datapackage
from bw_processing.constants import INDICES_DTYPE
from bw_processing.errors import NonUnique
from bw_processing.indexing import reindex, reset_index

### Fixture


def add_data(dp, id_field="id"):
    data_array = np.array([2, 7, 12])
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
            {id_field: 11, "a": 1, "c": 3, "d": 11},
            {id_field: 12, "a": 2, "c": 4, "d": 11},
            {id_field: 13, "a": 1, "c": 4, "d": 11},
            {id_field: 14, "a": 3, "c": 5, "d": 11},
            {id_field: 15, "a": 4, "c": 5, "d": 11},
            {id_field: 16, "a": 4, "c": 6, "d": 11},
        ]
    )

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
    dp.add_csv_metadata(
        dataframe=df,
        valid_for=[("array", "row"), ("vector", "col")],
        name="csv-multiple",
    )


@pytest.fixture
def fixture():
    dp = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )
    dp_unchanged(dp)
    return dp


def dp_unchanged(dp=None):
    if dp is None:
        dp = load_datapackage(
            OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
        )

    array, _ = dp.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    array, _ = dp.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    df, _ = dp.get_resource("vector-csv-rows")
    assert np.allclose(df["id"], np.array([11, 12, 13, 14, 15, 16]))


### reset_index


def test_reset_index_multiple_calls(fixture):
    reset_index(fixture, "vector-csv-rows")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    # New CSV file, so indexing starts over
    reset_index(fixture, "vector-csv-cols")
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([0, 1, 1]))

    dp_unchanged()


def test_reset_index_multiple_resources_referenced(fixture):
    reset_index(fixture, "csv-multiple")

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([2, 3, 3]))


def test_reset_index_modified(fixture):
    assert not fixture._modified

    reset_index(fixture, "vector-csv-rows")
    assert fixture._modified == set([fixture._get_index("vector.indices")])

    fixture = load_datapackage(
        OSFS(str(Path(__file__).parent.resolve() / "fixtures" / "indexing"))
    )
    assert not fixture._modified

    reset_index(fixture, "csv-multiple")
    assert fixture._modified == set(
        [fixture._get_index("vector.indices"), fixture._get_index("array.indices")]
    )


def test_reset_index_both_row_col(fixture):
    reset_index(fixture, "vector-csv-both")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([2, 3, 3]))


def test_reset_index_pass_only_cols(fixture):
    reset_index(fixture, "vector-csv-cols")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(array["col"], np.array([0, 1, 1]))


def test_reset_index_pass_datapackage(fixture):
    reset_index(fixture, "vector-csv-rows")

    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))


def test_reset_index_pass_array(fixture):
    reset_index(fixture, "array-csv-both")

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([2, 3, 3]))


def test_reset_index_return_object(fixture):
    dp = reset_index(fixture, "vector-csv-rows")

    array, _ = dp.get_resource("vector.indices")
    assert np.allclose(array["row"], np.array([0, 0, 1]))
    assert np.allclose(array["col"], np.array([14, 15, 15]))


def test_reset_index_metadata_name_error(fixture):
    with pytest.raises(KeyError):
        reset_index(fixture, "foo")

    with pytest.raises(ValueError):
        reset_index(fixture, "vector.indices")


### reindex


def test_reindex_normal(fixture):
    destination = [
        {"id": 21, "a": 1, "c": 3, "d": 11},
        {"id": 22, "a": 2, "c": 4, "d": 11},
        {"id": 23, "a": 1, "c": 4, "d": 11},
        {"id": 24, "a": 3, "c": 5, "d": 11},
        {"id": 25, "a": 4, "c": 5, "d": 11},
        {"id": 26, "a": 4, "c": 6, "d": 11},
    ]

    reindex(fixture, "vector-csv-rows", destination)

    array, _ = fixture.get_resource("vector.indices")
    df, _ = fixture.get_resource("vector-csv-rows")
    assert np.allclose(array["row"], np.array([21, 21, 23]))
    assert np.allclose(df["id"], np.array([21, 22, 23, 24, 25, 26]))


def test_reindex_multiple_resources(fixture):
    destination = [
        {"id": 21, "a": 1, "c": 3, "d": 11},
        {"id": 22, "a": 2, "c": 4, "d": 11},
        {"id": 23, "a": 1, "c": 4, "d": 11},
        {"id": 24, "a": 3, "c": 5, "d": 11},
        {"id": 25, "a": 4, "c": 5, "d": 11},
        {"id": 26, "a": 4, "c": 6, "d": 11},
    ]

    reindex(fixture, "csv-multiple", destination)

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([21, 21, 23]))
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["col"], np.array([24, 25, 25]))
    df, _ = fixture.get_resource("csv-multiple")
    assert np.allclose(df["id"], np.array([21, 22, 23, 24, 25, 26]))


def test_reindex_fields_subset(fixture):
    destination = [
        {"id": 21, "a": 1, "c": 3},
        {"id": 22, "a": 2, "c": 4},
        {"id": 23, "a": 1, "c": 4},
        {"id": 24, "a": 3, "c": 5},
        {"id": 25, "a": 4, "c": 5},
        {"id": 26, "a": 4, "c": 6},
    ]

    reindex(fixture, "csv-multiple", destination, fields=["a", "c"])

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([21, 21, 23]))
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["col"], np.array([24, 25, 25]))
    df, _ = fixture.get_resource("csv-multiple")
    assert np.allclose(df["id"], np.array([21, 22, 23, 24, 25, 26]))


def test_reindex_custom_id_field_datapackage():
    dp = create_datapackage()
    add_data(dp, "bar")

    destination = [
        {"id": 21, "a": 1, "c": 3, "d": 11},
        {"id": 22, "a": 2, "c": 4, "d": 11},
        {"id": 23, "a": 1, "c": 4, "d": 11},
        {"id": 24, "a": 3, "c": 5, "d": 11},
        {"id": 25, "a": 4, "c": 5, "d": 11},
        {"id": 26, "a": 4, "c": 6, "d": 11},
    ]
    array, _ = dp.get_resource("vector.indices")
    df, _ = dp.get_resource("vector-csv-rows")
    assert np.allclose(array["row"], np.array([11, 11, 13]))
    assert np.allclose(df["bar"], np.array([11, 12, 13, 14, 15, 16]))

    reindex(dp, "vector-csv-rows", destination, id_field_datapackage="bar")

    assert np.allclose(array["row"], np.array([21, 21, 23]))
    assert np.allclose(df["bar"], np.array([21, 22, 23, 24, 25, 26]))


def test_reindex_custom_id_field_destination(fixture):
    destination = [
        {"foo": 21, "a": 1, "c": 3, "d": 11},
        {"foo": 22, "a": 2, "c": 4, "d": 11},
        {"foo": 23, "a": 1, "c": 4, "d": 11},
        {"foo": 24, "a": 3, "c": 5, "d": 11},
        {"foo": 25, "a": 4, "c": 5, "d": 11},
        {"foo": 26, "a": 4, "c": 6, "d": 11},
    ]

    reindex(fixture, "csv-multiple", destination, id_field_destination="foo")

    array, _ = fixture.get_resource("array.indices")
    assert np.allclose(array["row"], np.array([21, 21, 23]))
    array, _ = fixture.get_resource("vector.indices")
    assert np.allclose(array["col"], np.array([24, 25, 25]))
    df, _ = fixture.get_resource("csv-multiple")
    assert np.allclose(df["id"], np.array([21, 22, 23, 24, 25, 26]))


def test_reindex_missing_id_in_datapackage(fixture):
    with pytest.raises(KeyError):
        reindex(fixture, "vector-csv-rows", [], id_field_datapackage="bar")


def test_reindex_missing_id_in_destination(fixture):
    destination = [
        {"foo": 21, "a": 1, "c": 3, "d": 11},
        {"foo": 22, "a": 2, "c": 4, "d": 11},
        {"foo": 23, "a": 1, "c": 4, "d": 11},
        {"foo": 24, "a": 3, "c": 5, "d": 11},
        {"foo": 25, "a": 4, "c": 5, "d": 11},
        {"foo": 26, "a": 4, "c": 6, "d": 11},
    ]
    with pytest.raises(KeyError):
        reindex(fixture, "csv-multiple", destination)


def test_reindex_destination_missing_default_id_field(fixture):
    destination = [
        {"a": 1, "c": 3, "d": 11},
        {"a": 1, "c": 4, "d": 11},
        {"a": 2, "c": 4, "d": 11},
        {"a": 1, "c": 4, "d": 11},
        {"a": 3, "c": 5, "d": 11},
        {"a": 4, "c": 5, "d": 11},
        {"a": 4, "c": 6, "d": 11},
    ]
    with pytest.raises(KeyError):
        reindex(fixture, "csv-multiple", destination)


def test_reindex_nonunique_in_destination(fixture):
    destination = [
        {"id": 20, "a": 1, "c": 3, "d": 11},
        {"id": 21, "a": 1, "c": 4, "d": 11},
        {"id": 22, "a": 2, "c": 4, "d": 11},
        {"id": 23, "a": 1, "c": 4, "d": 11},
        {"id": 24, "a": 3, "c": 5, "d": 11},
        {"id": 25, "a": 4, "c": 5, "d": 11},
        {"id": 26, "a": 4, "c": 6, "d": 11},
    ]
    with pytest.raises(NonUnique):
        reindex(fixture, "csv-multiple", destination)


def test_reindex_missing_metadata_name(fixture):
    with pytest.raises(KeyError):
        reindex(fixture, "foo", [])


def test_reindex_cant_find_in_data_iterable(fixture):
    destination = [
        {"id": 22, "a": 2, "c": 4, "d": 11},
        {"id": 23, "a": 1, "c": 4, "d": 11},
        {"id": 24, "a": 3, "c": 5, "d": 11},
        {"id": 25, "a": 4, "c": 5, "d": 11},
        {"id": 26, "a": 4, "c": 6, "d": 11},
    ]
    with pytest.raises(KeyError):
        reindex(fixture, "vector-csv-rows", destination)


def test_reindex_wrong_metadata_name_type(fixture):
    with pytest.raises(ValueError):
        reindex(fixture, "vector.indices", [])


def test_reindex_data_iterable_wrong_type(fixture):
    with pytest.raises(AttributeError):
        reindex(fixture, "csv-multiple", [1, 2, 3])


if __name__ == "__main__":
    dirpath = Path(__file__).parent.resolve() / "fixtures"
    dirpath.mkdir(exist_ok=True)
    (dirpath / "indexing").mkdir(exist_ok=True)

    dp = create_datapackage(
        fs=OSFS(str(dirpath / "indexing")), name="indexing-fixture", id_="fixture-i"
    )
    add_data(dp)
    dp.finalize_serialization()
