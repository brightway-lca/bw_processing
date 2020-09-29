from bw_processing.io_classes import (
    DirectoryIO,
    ZipfileIO,
    TemporaryDirectoryIO,
    InMemoryIO,
)
from bw_processing import create_datapackage, load_datapackage
from bw_processing.constants import MAX_SIGNED_32BIT_INT as M
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


def add_data(dp):
    sa = [
        np.array([0, 1, M, M, 2, 3.3, 4, M, M, 2.7, 3.9, False, True]),
        np.array([5, 6, M, M, 7, 8.3, 9, M, M, 7.7, 8.9, True, False]),
    ]
    indices = [np.array([0, 1, M, M]), np.array([2, 3, M, M]), np.array([4, 5, M, M])]
    data = np.arange(12).reshape((3, 4))
    json_data = [{"a": "b"}, 1, True]
    json_parameters = ["a", "foo"]
    df = pd.DataFrame(
        [
            {"id": 1, "a": 1, "c": 3, "d": 11},
            {"id": 2, "a": 2, "c": 4, "d": 11},
            {"id": 3, "a": 1, "c": 4, "d": 11},
        ]
    ).set_index(["id"])
    dp.add_structured_array(sa, "sa_matrix", name="sa-data")
    dp.add_presamples_data_array(
        data, matrix_label="sa-data", name="presamples-sa-matrix"
    )
    dp.add_presamples_indices_array(
        indices, data_array="presamples-sa-matrix", name="presamples-indices"
    )
    dp.add_csv_metadata(
        df, valid_for=[("presamples-sa-matrix", "rows")], name="presamples-csv-metadata"
    )
    dp.add_json_metadata(
        json_data, valid_for="presamples-sa-matrix", name="presamples-json-metadata"
    )
    dp.add_json_metadata(
        json_parameters, valid_for="sa-data", name="presamples-json-parameters"
    )


def check_metadata(dp):
    assert dp.metadata["resources"] == [
        {
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "sa-data",
            "matrix": "sa_matrix",
            "kind": "processed array",
            "path": "sa-data.npy",
        },
        {
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "presamples-sa-matrix",
            "matrix": "sa-data",
            "kind": "presamples",
            "path": "presamples-sa-matrix.npy",
        },
        {
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": "presamples-indices",
            "path": "presamples-indices.npy",
            "data_array": "presamples-sa-matrix",
        },
        {
            "profile": "data-resource",
            "mediatype": "text/csv",
            "path": "presamples-csv-metadata.csv",
            "name": "presamples-csv-metadata",
            "valid_for": [("presamples-sa-matrix", "rows")],
        },
        {
            "profile": "data-resource",
            "mediatype": "application/json",
            "path": "presamples-json-metadata.json",
            "name": "presamples-json-metadata",
            "valid_for": "presamples-sa-matrix",
        },
        {
            "profile": "data-resource",
            "mediatype": "application/json",
            "path": "presamples-json-parameters.json",
            "name": "presamples-json-parameters",
            "valid_for": "sa-data",
        },
    ]
    assert dp.metadata["created"].endswith("Z")
    assert isinstance(dp.metadata["licenses"], list)
    assert dp.metadata["profile"] == "data-package"
    assert dp.metadata["name"] == "the-name"
    assert dp.metadata["id"] == "the id"


def test_integration_test_in_memory():
    dp = create_datapackage(None, "the-name", "the id", compress=False)
    assert isinstance(dp.io_obj, InMemoryIO)
    add_data(dp)
    dp.finalize()

    check_metadata(dp)
