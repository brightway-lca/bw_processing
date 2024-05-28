# -*- coding: utf-8 -*-
"""
A basic example on how to use parquet files.
"""
from pathlib import Path

import numpy as np
from fsspec.implementations.zip import ZipFileSystem

import bw_processing as bwp
from bw_processing.io_helpers import generic_directory_filesystem

if __name__ == "__main__":
    print("This is a basic example on how to use parquet files.")
    print("Trying to construct a basic datapackage...")
    indices_array = np.array([(0, 1), (1, 0), (1, 1)], dtype=bwp.INDICES_DTYPE)
    data_array = np.array([1, 2, 3])
    flip_array = np.array([True, False, False])

    dirpath = Path(__file__).parent.resolve()

    # Change this if you want to OSFS vs ZIP to store the Datapackage object
    USE_OSFS = False
    # Change this if you want to use numpy or parquet format by default
    default_matrix_serialize_format_type = (
        bwp.MatrixSerializeFormat.NUMPY
    )  # bwp.MatrixSerializeFormat.PARQUET

    if USE_OSFS:
        # VERSION OSFS
        # Directory must exist for OSFS otherwise use OSFS(dirpath, create=True)!
        # Every created object will be saved in that same directory
        dp_dir = generic_directory_filesystem(dirpath=dirpath / "datapackage_1", create=True)
        dp = bwp.create_datapackage(
            fs=dp_dir, matrix_serialize_format_type=bwp.MatrixSerializeFormat.NUMPY
        )
    else:
        # VERSION ZIP
        dp_zip_file = ZipFileSystem(str(dirpath / "datapackage_2.zip"), mode="w")
        dp = bwp.create_datapackage(
            fs=dp_zip_file, matrix_serialize_format_type=bwp.MatrixSerializeFormat.NUMPY
        )  # bwp.create_datapackage(fs=dp_zip_file, serialize_type=SerializeENum.parquet)

    # Add some data to the Datapackage
    # We ask to serialize/save in parquet format specifically
    # Note that you can save some data in numpy format and other data in paquet format for the same Datapackage object
    dp.add_persistent_vector(
        matrix="some name",
        data_array=data_array,
        name="some name",
        indices_array=indices_array,
        flip_array=flip_array,
        keep_proxy=False,
        matrix_serialize_format_type=bwp.MatrixSerializeFormat.PARQUET,
    )

    dp.add_persistent_vector(
        matrix="another_matrix",
        name="another name",
        indices_array=indices_array,
        matrix_serialize_format_type=bwp.MatrixSerializeFormat.PARQUET,
    )

    from_dicts = [
        {
            "row": 0,
            "col": 1,
            "flip": True,
            "amount": 3.3,
            "uncertainty_type": 2,
            "loc": 2.7,
            "scale": 3.9,
        },
        {
            "row": 5,
            "col": 6,
            "flip": False,
            "amount": 8.3,
            "uncertainty_type": 7,
            "loc": 7.7,
            "scale": 8.9,
        },
    ]

    dp.add_persistent_vector_from_iterator(
        matrix="sa_matrix",
        name="sa-data-vector-from-dict",
        dict_iterator=from_dicts,
        foo="bar",
        matrix_serialize_format_type=bwp.MatrixSerializeFormat.PARQUET,
    )

    # to write the JSON metapackage in the file datapackage.json
    dp.finalize_serialization()
    print("Done!")
    print("=" * 80)
    print("Trying to load datapackage back...")
    # OSFS must be open! (and it was closed with finalize_serialization())

    if USE_OSFS:
        dp_dir = generic_directory_filesystem(dirpath=dirpath / "datapackage_1")
        dp2 = bwp.load_datapackage(fs_or_obj=dp_dir)
    else:
        dp_zip_file = ZipFileSystem(dirpath / "datapackage_2.zip")
        dp2 = bwp.load_datapackage(fs_or_obj=dp_zip_file)

    print("Done!")
    print("Print Datapackage resources:")
    print(dp2.resources)
