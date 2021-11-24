from pathlib import Path

import pytest
from fs.zipfs import ZipFS

from bw_processing import load_datapackage

dirpath = Path(__file__).parent.resolve() / "fixtures"


class Vector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __next__(self):
        return 1


class Array:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, args):
        return args


def test_list_dehydrated_interfaces():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    assert dp.dehydrated_interfaces() == ["sa-vector-interface", "sa-array-interface"]

    dp.rehydrate_interface("sa-vector-interface.data", Vector())
    assert dp.dehydrated_interfaces() == ["sa-array-interface"]


def test_rehydrate_vector_interface():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    dp.rehydrate_interface("sa-vector-interface.data", Vector())
    data, resource = dp.get_resource("sa-vector-interface.data")
    assert next(data) == 1

    expected = {
        "category": "vector",
        "group": "sa-vector-interface",
        "kind": "data",
        "matrix": "sa_matrix",
        "name": "sa-vector-interface.data",
        "profile": "interface",
        "nrows": 3,
    }
    assert resource == expected


def test_rehydrate_vector_interface_fix_name():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    dp.rehydrate_interface("sa-vector-interface", Vector())
    data, resource = dp.get_resource("sa-vector-interface.data")
    assert next(data) == 1


def test_rehydrate_vector_interface_config():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    data, resource = dp.get_resource("sa-vector-interface.data")
    resource["config"] = {"foo": "bar"}

    dp.rehydrate_interface("sa-vector-interface.data", Vector, True)
    assert dp.dehydrated_interfaces() == ["sa-array-interface"]
    data, resource = dp.get_resource("sa-vector-interface.data")
    assert data.kwargs == {"foo": "bar"}


def test_rehydrate_vector_interface_config_keyerror():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    data, resource = dp.get_resource("sa-vector-interface.data")

    with pytest.raises(KeyError):
        dp.rehydrate_interface("sa-vector-interface.data", Vector, True)


def test_rehydrate_array_interface():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    dp.rehydrate_interface("sa-array-interface.data", Array())
    data, resource = dp.get_resource("sa-array-interface.data")
    assert data[7] == 7

    expected = {
        "category": "array",
        "group": "sa-array-interface",
        "kind": "data",
        "matrix": "sa_matrix",
        "name": "sa-array-interface.data",
        "profile": "interface",
        "nrows": 3,
    }
    assert resource == expected


def test_rehydrate_array_interface_config():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    data, resource = dp.get_resource("sa-array-interface.data")
    resource["config"] = {"foo": "bar"}

    dp.rehydrate_interface("sa-array-interface.data", Array, True)
    assert dp.dehydrated_interfaces() == ["sa-vector-interface"]
    data, resource = dp.get_resource("sa-array-interface.data")
    assert data.kwargs == {"foo": "bar"}


def test_rehydrate_array_interface_config_keyerror():
    dp = load_datapackage(ZipFS(dirpath / "test-fixture.zip"))
    data, resource = dp.get_resource("sa-array-interface.data")

    with pytest.raises(KeyError):
        dp.rehydrate_interface("sa-array-interface.data", Array, True)
