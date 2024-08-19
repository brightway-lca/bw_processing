# from bw_processing.loading import load_bytes
# from bw_processing import create_package
# from io import BytesIO
# from pathlib import Path
# import json
# import numpy as np
# import tempfile


# def test_load_package_in_directory():
#     with tempfile.TemporaryDirectory() as td:
#         td = Path(td)

#         resources = [
#             {
#                 "name": "first-resource",
#                 "path": "some-array.npy",
#                 "matrix": "technosphere",
#                 "data": [
#                     tuple(list(range(11)) + [False, False]),
#                     tuple(list(range(12, 23)) + [True, True]),
#                 ],
#             }
#         ]
#         with tempfile.TemporaryDirectory() as td:
#             fp = create_package(
#                 name="test-package", resources=resources, path=td, replace=False
#             )
#             # Test data in fp


# def test_load_json():
#     with tempfile.TemporaryDirectory() as td:
#         td = Path(td)
#         data = [{'foo': 'bar', }, 1, True]
#         json.dump(data, open(td / "data.json", "w"))
#         assert mapping["json"](open(td / "data.json")) == data


# # def test_load_numpy():
# #     with tempfile.TemporaryDirectory() as td:
# #         td = Path(td)
# #         data = np.arange(10)
# #         np.save(td / "array.npy", data)
# #         assert np.allclose(mapping["npy"](open(td / "array.npy")), data)


# # def
# #     resources = [
# #         {
# #             "name": "first-resource",
# #             "path": "some-array.npy",
# #             "matrix": "technosphere",
# #             "data": [
# #                 tuple(list(range(11)) + [False, False]),
# #                 tuple(list(range(12, 23)) + [True, True]),
# #             ],
# #         }
# #     ]
# #     with tempfile.TemporaryDirectory() as td:
# #         fp = create_package(
# #             name="test-package", resources=resources, path=td, compress=False
# #         )
# #         # Test data in fp
