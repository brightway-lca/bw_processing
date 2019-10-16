from bw_processing.errors import InvalidName, NonUnique, InconsistentFields
from bw_processing import (
    as_unique_attributes,
    chunked,
    COMMON_DTYPE,
    create_calculation_package,
    create_datapackage_metadata,
    create_numpy_structured_array,
    create_processed_datapackage,
    format_calculation_resource,
    greedy_set_cover,
    NAME_RE,
)
from pathlib import Path
import numpy as np
import pytest
import tempfile


fixtures_dir = Path(__file__, "..").resolve() / "fixtures"


def test_chunked():
    c = chunked(range(600), 250)
    for x in next(c):
        pass
    assert x == 249
    for x in next(c):
        pass
    assert x == 499
    for x in next(c):
        pass
    assert x == 599


def test_create_array():
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "array.npy"
        data = [
            tuple(list(range(11)) + [False, False]),
            tuple(list(range(12, 23)) + [True, True]),
        ]
        create_numpy_structured_array(data, fp)
        result = np.load(fp)
        assert result.shape == (2,)
        assert result.dtype == COMMON_DTYPE
        assert np.allclose(result["row_value"], [0, 12])
        assert np.allclose(result["flip"], [False, True])


def test_create_array_format_function():
    def func(x, dtype):
        return (2, 4, 1, 3, 5, 7, 6, 8, 9, 11, 10, False, True)

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "array.npy"
        create_numpy_structured_array(range(10), fp, format_function=func)
        result = np.load(fp)
        assert result.shape == (10,)
        assert result.dtype == COMMON_DTYPE
        assert result["row_value"].sum() == 20


def test_create_array_specify_nrows():
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "array.npy"
        data = [tuple(list(range(11)) + [False, False])] * 200
        create_numpy_structured_array(data, fp, nrows=200)
        result = np.load(fp)
        assert result.shape == (200,)
        assert result["row_value"].sum() == 0


def test_create_array_specify_nrows_too_many():
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "array.npy"
        data = [tuple(list(range(11)) + [False, False])] * 200
        with pytest.raises(ValueError):
            create_numpy_structured_array(data, fp, nrows=100)


def test_create_array_chunk_data():
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "array.npy"
        data = [tuple(list(range(11)) + [False, False])] * 90000
        create_numpy_structured_array(data, fp)
        result = np.load(fp)
        assert result.shape == (90000,)
        assert result["row_value"].sum() == 0


def test_format_datapackage_metadata():
    expected = {"profile": "data-package", "name": "a", "id": "b", "licenses": "c"}
    result = create_datapackage_metadata(
        "a",
        [],
        resource_function=format_calculation_resource,
        id_="b",
        metadata={"licenses": "c"},
    )
    assert result["created"]
    for k, v in expected.items():
        assert result[k] == v


def test_name_re():
    assert NAME_RE.match("hey_you")
    assert not NAME_RE.match("hey_you!")


def test_format_datapackage_metadata_no_id():
    result = create_datapackage_metadata(
        "a", [], resource_function=format_calculation_resource
    )
    assert result["id"]
    assert len(result["id"]) > 16


def test_format_datapackage_metadata_default_licenses():
    result = create_datapackage_metadata(
        "a", [], resource_function=format_calculation_resource
    )
    assert result["licenses"] == [
        {
            "name": "ODC-PDDL-1.0",
            "path": "http://opendatacommons.org/licenses/pddl/",
            "title": "Open Data Commons Public Domain Dedication and License v1.0",
        }
    ]


def test_format_datapackage_metadata_invalid_name():
    with pytest.raises(InvalidName):
        create_datapackage_metadata(
            "woo!", {}, resource_function=format_calculation_resource
        )


def test_format_calculation_resource():
    given = {
        "path": "basic_array.npy",
        "name": "test-name",
        "matrix": "technosphere",
        "description": "some words",
        "foo": "bar",
    }
    expected = {
        "format": "npy",
        "mediatype": "application/octet-stream",
        "path": "basic_array.npy",
        "name": "test-name",
        "profile": "data-resource",
        "matrix": "technosphere",
        "description": "some words",
        "foo": "bar",
    }
    assert format_calculation_resource(given) == expected


def test_calculation_package():
    resources = [
        {
            "name": "first-resource",
            "path": "some-array.npy",
            "matrix": "technosphere",
            "data": [
                tuple(list(range(11)) + [False, False]),
                tuple(list(range(12, 23)) + [True, True]),
            ],
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        fp = create_calculation_package(
            name="test-package", resources=resources, path=td
        )
        # Test data in fp


def test_calculation_package_name_conflict():
    pass


def test_calculation_package_specify_id():
    pass


def test_calculation_package_metadata():
    pass


def test_greedy_set():
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 2, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 4},
    ]
    assert greedy_set_cover(data) == {"a", "c"}


def test_greedy_set_error():
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 2, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]
    with pytest.raises(NonUnique):
        greedy_set_cover(data)


def test_greedy_set_error_ids_unique():
    data = [
        {"id": 1, "a": 1, "b": 2, "c": 3},
        {"id": 2, "a": 2, "b": 2, "c": 3},
        {"id": 3, "a": 1, "b": 2, "c": 3},
    ]
    with pytest.raises(NonUnique):
        greedy_set_cover(data)


def test_greedy_set_exclude():
    data = [
        {"foo": 7, "a": 1, "b": 2, "c": 3},
        {"foo": 8, "a": 2, "b": 2, "c": 3},
        {"foo": 9, "a": 1, "b": 2, "c": 4},
    ]
    assert greedy_set_cover(data, exclude=["foo"]) == {"a", "c"}


def test_as_unique_attributes():
    data = [
        {"id": 1, "a": 1, "b": 2, "c": 3},
        {"id": 2, "a": 2, "b": 2, "c": 3},
        {"id": 3, "a": 1, "b": 2, "c": 4},
    ]
    expected = (
        {"a", "c", "id"},
        [
            {"id": 1, "a": 1, "c": 3},
            {"id": 2, "a": 2, "c": 3},
            {"id": 3, "a": 1, "c": 4},
        ],
    )
    assert as_unique_attributes(data) == expected


def test_as_unique_attributes_include_nonunique():
    data = [
        {"id": 1, "a": 1, "b": 8, "c": 3, "d": 11},
        {"id": 2, "a": 2, "b": 8, "c": 3, "d": 11},
        {"id": 3, "a": 1, "b": 8, "c": 4, "d": 11},
    ]
    expected = (
        {"a", "b", "c", "id"},
        [
            {"id": 1, "a": 1, "b": 8, "c": 3},
            {"id": 2, "a": 2, "b": 8, "c": 3},
            {"id": 3, "a": 1, "b": 8, "c": 4},
        ],
    )
    assert as_unique_attributes(data, include=["b"]) == expected


def test_as_unique_attributes_include_exclude():
    data = [
        {"id": 1, "a": 1, "b": 7, "c": 3, "d": 11},
        {"id": 2, "a": 2, "b": 8, "c": 3, "d": 11},
        {"id": 3, "a": 1, "b": 9, "c": 4, "d": 11},
    ]
    expected = (
        {"a", "b", "c", "id"},
        [
            {"id": 1, "a": 1, "b": 7, "c": 3},
            {"id": 2, "a": 2, "b": 8, "c": 3},
            {"id": 3, "a": 1, "b": 9, "c": 4},
        ],
    )
    assert as_unique_attributes(data, exclude=["b"], include=["b"]) == expected


def test_as_unique_attributes_error():
    data = [
        {"id": 1, "a": 1, "c": 3, "d": 11},
        {"id": 2, "a": 2, "c": 4, "d": 11},
        {"id": 3, "a": 1, "c": 3, "d": 11},
    ]
    with pytest.raises(NonUnique):
        as_unique_attributes(data)


def test_create_processed_datapackage():
    pass
