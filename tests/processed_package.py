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
from bw_processing.errors import NonUnique
import pytest


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
