# import fixtures
from fixtures.basic_arrays import *

import pytest

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def data_directory():
    dirpath = Path(__file__).parent.resolve() / "data"
    return dirpath


@pytest.fixture(scope="session")
def data_parquet_files_directory(data_directory):
    return (data_directory / "parquet_files").resolve(strict=True)


@pytest.fixture(scope="session")
def helpers_directory():
    dirpath = Path(__file__).parent.resolve() / "helpersdata"
    return dirpath

