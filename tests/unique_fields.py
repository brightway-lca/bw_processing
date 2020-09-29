from bw_processing.errors import NonUnique
from bw_processing.unique_fields import (
    as_unique_attributes,
    as_unique_attributes_dataframe,
    greedy_set_cover,
)
import pandas as pd
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


def test_greedy_set_no_raise_error():
    data = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 2, "b": 2, "c": 3},
        {"a": 1, "b": 2, "c": 3},
    ]
    greedy_set_cover(data, raise_error=False)


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
        as_unique_attributes(data, raise_error=True)


def test_as_unique_attributes_not_raise_error():
    data = [
        {"id": 1, "a": 1, "c": 3, "d": 11},
        {"id": 2, "a": 2, "c": 4, "d": 11},
        {"id": 3, "a": 1, "c": 3, "d": 11},
    ]
    as_unique_attributes(data, raise_error=False)


def test_as_unique_attributes_dataframe():
    df = as_unique_attributes_dataframe(
        pd.DataFrame(
            [
                {"id": 1, "a": 1, "c": 3, "d": 11},
                {"id": 2, "a": 2, "c": 4, "d": 11},
                {"id": 3, "a": 1, "c": 4, "d": 11},
            ]
        ).set_index(["id"])
    )
    assert set(df.columns) == {"a", "c"}
