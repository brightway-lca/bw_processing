# from bw_processing import (
#     as_unique_attributes,
#     chunked,
#     COMMON_DTYPE,
#     create_package,
#     create_datapackage_metadata,
#     create_structured_array,
#     create_processed_datapackage,
#     format_calculation_resource,
#     greedy_set_cover,
#     NAME_RE,
# )
# from copy import deepcopy
# import pytest
# import tempfile


# def test_format_calculation_resource():
#     given = {
#         "path": "basic_array",
#         "name": "test-name",
#         "matrix": "technosphere",
#         "description": "some words",
#         "foo": "bar",
#     }
#     expected = {
#         "format": "npy",
#         "mediatype": "application/octet-stream",
#         "path": "basic_array.npy",
#         "name": "test-name",
#         "profile": "data-resource",
#         "matrix": "technosphere",
#         "description": "some words",
#         "foo": "bar",
#     }
#     assert format_calculation_resource(given) == expected


# def test_calculation_package():
#     resources = [
#         {
#             "name": "first-resource",
#             "path": "some-array.npy",
#             "matrix": "technosphere",
#             "data": [
#                 tuple(list(range(11)) + [False, False]),
#                 tuple(list(range(12, 23)) + [True, True]),
#             ],
#         }
#     ]
#     with tempfile.TemporaryDirectory() as td:
#         fp = create_package(
#             name="test-package", resources=resources, path=td, replace=False
#         )
#         # Test data in fp


# def test_calculation_package_directory():
#     resources = [
#         {
#             "name": "first-resource",
#             "path": "some-array.npy",
#             "matrix": "technosphere",
#             "data": [
#                 tuple(list(range(11)) + [False, False]),
#                 tuple(list(range(12, 23)) + [True, True]),
#             ],
#         }
#     ]
#     with tempfile.TemporaryDirectory() as td:
#         fp = create_package(
#             name="test-package", resources=resources, path=td, compress=False
#         )
#         # Test data in fp


# def test_calculation_package_in_memory():
#     resources = [
#         {
#             "name": "first-resource",
#             "path": "some-array.npy",
#             "matrix": "technosphere",
#             "data": [
#                 tuple(list(range(11)) + [False, False]),
#                 tuple(list(range(12, 23)) + [True, True]),
#             ],
#         }
#     ]
#     fp = create_package(name="test-package", resources=resources)
#     # Test data in fp


# def test_calculation_package_replace():
#     resources = [
#         {
#             "name": "first-resource",
#             "path": "some-array.npy",
#             "matrix": "technosphere",
#             "data": [
#                 tuple(list(range(11)) + [False, False]),
#                 tuple(list(range(12, 23)) + [True, True]),
#             ],
#         }
#     ]
#     with tempfile.TemporaryDirectory() as td:
#         create_package(
#             name="test-package", resources=deepcopy(resources), path=td
#         )
#         create_package(
#             name="test-package", resources=deepcopy(resources), path=td, replace=True
#         )


# def test_calculation_package_replace_error():
#     resources = [
#         {
#             "name": "first-resource",
#             "path": "some-array.npy",
#             "matrix": "technosphere",
#             "data": [
#                 tuple(list(range(11)) + [False, False]),
#                 tuple(list(range(12, 23)) + [True, True]),
#             ],
#         }
#     ]
#     with tempfile.TemporaryDirectory() as td:
#         create_package(
#             name="test-package", resources=deepcopy(resources), path=td
#         )
#         with pytest.raises(ValueError):
#             create_package(
#                 name="test-package",
#                 resources=deepcopy(resources),
#                 path=td,
#                 replace=False,
#             )


# def test_calculation_package_name_conflict():
#     pass


# def test_calculation_package_specify_id():
#     pass


# def test_calculation_package_metadata():
#     pass
