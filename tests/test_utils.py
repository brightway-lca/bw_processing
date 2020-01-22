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
from bw_processing.utils import dictionary_formatter, MAX_SIGNED_32BIT_INT as M
from bw_processing.errors import InvalidName
from pathlib import Path
import numpy as np
import pytest
import tempfile


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


def test_dictionary_formatter_sparse():
    given = {"row": 1, "amount": 4}
    result = dictionary_formatter(given)
    print(result)
    assert result[:7] == (1, 1, M, M, 0, 4, 4)
    assert all(np.isnan(x) for x in result[7:11])
    assert result[11:] == (False, False)


def test_dictionary_formatter_complete():
    given = {
        "row": 1,
        "col": 2,
        "uncertainty_type": 3,
        "amount": 4,
        "loc": 5,
        "scale": 6,
        "shape": 7,
        "minimum": 8,
        "maximum": 9,
        "negative": True,
        "flip": False,
    }
    expected = (1, 2, M, M, 3, 4, 5, 6, 7, 8, 9, True, False)
    assert dictionary_formatter(given) == expected


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


def test_create_datapackage_metadata():
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


def test_create_datapackage_metadata_no_id():
    result = create_datapackage_metadata(
        "a", [], resource_function=format_calculation_resource
    )
    assert result["id"]
    assert len(result["id"]) > 16


def test_create_datapackage_default_formatter():
    result = create_datapackage_metadata("a", ["b", 2])
    assert result["resources"] == ["b", 2]


def test_create_datapackage_metadata_default_licenses():
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


def test_create_datapackage_metadata_invalid_name():
    with pytest.raises(InvalidName):
        create_datapackage_metadata(
            "woo!", {}, resource_function=format_calculation_resource
        )
