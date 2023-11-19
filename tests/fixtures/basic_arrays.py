# -*- coding: utf-8 -*-
"""
Some basic array fixtures.

We also save these arrays into `parquet` files.
"""
import numpy as np
import pytest

from bw_processing.constants import INDICES_DTYPE, UNCERTAINTY_DTYPE


@pytest.fixture(scope="session")
def indices_vector():
    return np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)


@pytest.fixture(scope="session")
def flip_vector():
    return np.array([True, False, False])


@pytest.fixture(scope="session")
def distributions_vector():
    return np.array(
        [
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (0, 1, np.NaN, np.NaN, np.NaN, np.NaN, False),
            (
                5,
                237,
                np.NaN,
                np.NaN,
                200,
                300,
                False,
            ),  # triangular uncertainty from 200 to 300
            (5, 2.5, np.NaN, np.NaN, 2, 3, False),  # triangular uncertainty from 2 to 3
        ],
        dtype=UNCERTAINTY_DTYPE,
    )
