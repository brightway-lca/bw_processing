import math
import warnings

import numpy as np
import pytest

from bw_processing import (
    ArrayEntry,
    MatrixEntry,
    MatrixName,
    create_datapackage,
    create_datapackage_from_entries,
    simple_graph,
)
from bw_processing.constants import INDICES_DTYPE, UNCERTAINTY_DTYPE


class TestMatrixName:
    def test_values(self):
        assert MatrixName.technosphere == "technosphere_matrix"
        assert MatrixName.biosphere == "biosphere_matrix"
        assert MatrixName.characterization == "characterization_matrix"

    def test_str(self):
        assert str(MatrixName.technosphere) == "technosphere_matrix"

    def test_fstring(self):
        assert f"{MatrixName.biosphere}" == "biosphere_matrix"

    def test_usable_as_dict_key(self):
        d = {MatrixName.technosphere: [1, 2, 3]}
        assert d["technosphere_matrix"] == [1, 2, 3]


class TestMatrixEntry:
    def test_required_fields(self):
        e = MatrixEntry(row=1, col=2, amount=3.0)
        assert e.row == 1
        assert e.col == 2
        assert e.amount == 3.0

    def test_defaults(self):
        e = MatrixEntry(row=1, col=2, amount=3.0)
        assert e.flip is False
        assert e.uncertainty_type == 0
        assert e.negative is False
        assert e.loc == pytest.approx(3.0)
        assert math.isnan(e.scale)
        assert math.isnan(e.shape)
        assert math.isnan(e.minimum)
        assert math.isnan(e.maximum)
        assert e.rescale == pytest.approx(1.0)

    def test_rescale_custom_value(self):
        e = MatrixEntry(row=1, col=2, amount=3.0, rescale=0.5)
        assert e.rescale == pytest.approx(0.5)

    def test_rescale_in_as_dict(self):
        e = MatrixEntry(row=1, col=2, amount=3.0, rescale=2.0)
        assert e.as_dict()["rescale"] == pytest.approx(2.0)

    def test_loc_set_to_amount_for_no_uncertainty(self):
        e = MatrixEntry(row=1, col=2, amount=5.0)
        assert e.loc == pytest.approx(5.0)

    def test_explicit_loc_matching_amount_accepted(self):
        e = MatrixEntry(row=1, col=2, amount=5.0, uncertainty_type=0, loc=5.0)
        assert e.loc == pytest.approx(5.0)

    def test_loc_mismatch_raises_for_uncertainty_type_0(self):
        with pytest.raises(ValueError, match="loc == amount"):
            MatrixEntry(row=1, col=2, amount=5.0, uncertainty_type=0, loc=9.9)

    def test_loc_mismatch_raises_for_uncertainty_type_1(self):
        with pytest.raises(ValueError, match="loc == amount"):
            MatrixEntry(row=1, col=2, amount=5.0, uncertainty_type=1, loc=9.9)

    def test_loc_not_set_when_uncertainty_type_nonzero(self):
        e = MatrixEntry(row=1, col=2, amount=5.0, uncertainty_type=2)
        assert math.isnan(e.loc)

    def test_frozen(self):
        e = MatrixEntry(row=1, col=2, amount=3.0)
        with pytest.raises(Exception):
            e.amount = 99.0

    def test_as_dict_keys(self):
        e = MatrixEntry(row=1, col=2, amount=3.0)
        d = e.as_dict()
        assert set(d.keys()) == {
            "row", "col", "amount", "flip", "uncertainty_type",
            "loc", "scale", "shape", "minimum", "maximum", "negative",
            "rescale",
        }

    def test_as_dict_values(self):
        e = MatrixEntry(row=5, col=10, amount=2.5, flip=True, uncertainty_type=2,
                        loc=0.9, scale=0.1, negative=True)
        d = e.as_dict()
        assert d["row"] == 5
        assert d["col"] == 10
        assert d["amount"] == 2.5
        assert d["flip"] is True
        assert d["uncertainty_type"] == 2
        assert d["loc"] == pytest.approx(0.9)
        assert d["scale"] == pytest.approx(0.1)
        assert d["negative"] is True


class TestCreateDatapackageFromEntries:
    def test_basic(self):
        entries = [
            MatrixEntry(row=1, col=4, amount=2.0),
            MatrixEntry(row=2, col=5, amount=7.0),
        ]
        dp = create_datapackage_from_entries({MatrixName.technosphere: entries})
        assert "technosphere_matrix-data" in dp.groups

    def test_multiple_matrices(self):
        dp = create_datapackage_from_entries({
            MatrixName.technosphere: [MatrixEntry(row=1, col=2, amount=1.0)],
            MatrixName.biosphere: [MatrixEntry(row=3, col=4, amount=0.5)],
        })
        groups = list(dp.groups.keys())
        assert "technosphere_matrix-data" in groups
        assert "biosphere_matrix-data" in groups

    def test_plain_string_matrix_name(self):
        dp = create_datapackage_from_entries({
            "custom_matrix": [MatrixEntry(row=1, col=2, amount=1.0)],
        })
        assert "custom_matrix-data" in dp.groups

    def test_data_values(self):
        entries = [
            MatrixEntry(row=1, col=4, amount=2.0),
            MatrixEntry(row=2, col=5, amount=7.0),
            MatrixEntry(row=3, col=6, amount=12.0),
        ]
        dp = create_datapackage_from_entries({MatrixName.technosphere: entries})
        group = dp.groups["technosphere_matrix-data"]

        data_resource = next(r for r in group.resources if r["kind"] == "data")
        data_idx = dp.resources.index(data_resource)
        data = dp.data[data_idx]
        assert set(data) == {2.0, 7.0, 12.0}

    def test_flip_stored(self):
        entries = [
            MatrixEntry(row=1, col=4, amount=2.0, flip=True),
            MatrixEntry(row=2, col=5, amount=7.0, flip=False),
        ]
        dp = create_datapackage_from_entries({MatrixName.technosphere: entries})
        group = dp.groups["technosphere_matrix-data"]

        flip_resource = next(r for r in group.resources if r["kind"] == "flip")
        flip_idx = dp.resources.index(flip_resource)
        flip = dp.data[flip_idx]
        assert flip.sum() == 1

    def test_uncertainty_stored(self):
        entries = [
            MatrixEntry(row=1, col=4, amount=2.0, uncertainty_type=2, loc=0.7, scale=0.1),
            MatrixEntry(row=2, col=5, amount=7.0),
        ]
        dp = create_datapackage_from_entries({MatrixName.technosphere: entries})
        group = dp.groups["technosphere_matrix-data"]

        dist_resource = next(r for r in group.resources if r["kind"] == "distributions")
        dist_idx = dp.resources.index(dist_resource)
        dist = dp.data[dist_idx]
        assert dist.dtype == np.dtype(UNCERTAINTY_DTYPE)
        uncertainty_types = set(dist["uncertainty_type"])
        assert 2 in uncertainty_types

    def test_metadata_passed_through(self):
        dp = create_datapackage_from_entries(
            {MatrixName.technosphere: [MatrixEntry(row=1, col=2, amount=1.0)]},
            name="my-package",
        )
        assert dp.metadata["name"] == "my-package"

class TestArrayEntry:
    def test_basic_construction(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)))
        assert list(e.rows) == [0, 1]
        assert list(e.cols) == [2, 3]
        assert e.data.shape == (2, 4)
        assert e.flip is None

    def test_with_flip(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)), flip=[True, False])
        assert list(e.flip) == [True, False]

    def test_numpy_inputs(self):
        rows = np.array([0, 1, 2])
        cols = np.array([3, 4, 5])
        data = np.ones((3, 10))
        e = ArrayEntry(rows=rows, cols=cols, data=data)
        assert e.data.shape == (3, 10)

    def test_fields_are_normalized_to_ndarray(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)))
        assert isinstance(e.rows, np.ndarray)
        assert isinstance(e.cols, np.ndarray)
        assert isinstance(e.data, np.ndarray)

    def test_flip_coerced_to_bool(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)), flip=[1, 0])
        assert e.flip.dtype == bool
        assert list(e.flip) == [True, False]

    def test_rows_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            ArrayEntry(rows=[[0, 1], [2, 3]], cols=[0, 1, 2, 3], data=np.ones((4, 2)))

    def test_rows_must_be_integer_dtype(self):
        with pytest.raises(ValueError, match="integer dtype"):
            ArrayEntry(rows=np.array([1.7, 2.9]), cols=np.array([3, 4]), data=np.ones((2, 3)))

    def test_cols_must_be_integer_dtype(self):
        with pytest.raises(ValueError, match="integer dtype"):
            ArrayEntry(rows=np.array([1, 2]), cols=np.array([3.0, 4.0]), data=np.ones((2, 3)))

    def test_cols_shape_mismatch(self):
        with pytest.raises(ValueError, match="cols.*rows"):
            ArrayEntry(rows=[0, 1], cols=[0, 1, 2], data=np.ones((2, 3)))

    def test_data_must_be_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones(2))

    def test_data_row_count_mismatch(self):
        with pytest.raises(ValueError, match="data.*rows"):
            ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((3, 4)))

    def test_flip_shape_mismatch(self):
        with pytest.raises(ValueError, match="flip.*rows"):
            ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)), flip=[True, False, True])

    def test_rescale_default_is_none(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)))
        assert e.rescale is None

    def test_rescale_coerced_to_float32(self):
        e = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)), rescale=[2.0, 0.5])
        assert e.rescale.dtype == np.float32
        np.testing.assert_array_almost_equal(e.rescale, [2.0, 0.5])

    def test_rescale_shape_mismatch(self):
        with pytest.raises(ValueError, match="rescale.*rows"):
            ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 4)), rescale=[1.0, 2.0, 3.0])


class TestAddArrayEntries:
    def test_single_entry(self):
        dp = create_datapackage()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=data)
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        assert len(dp.groups) == 1

    def test_indices_stored_correctly(self):
        dp = create_datapackage()
        data = np.ones((2, 3))
        entry = ArrayEntry(rows=[5, 6], cols=[7, 8], data=data)
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        idx_resource = next(r for r in group.resources if r["kind"] == "indices")
        idx = dp.data[dp.resources.index(idx_resource)]
        assert idx.dtype == np.dtype(INDICES_DTYPE)
        assert list(idx["row"]) == [5, 6]
        assert list(idx["col"]) == [7, 8]

    def test_data_stored_correctly(self):
        dp = create_datapackage()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=data)
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        data_resource = next(r for r in group.resources if r["kind"] == "data")
        stored = dp.data[dp.resources.index(data_resource)]
        np.testing.assert_array_equal(stored, data)

    def test_flip_stored(self):
        dp = create_datapackage()
        data = np.ones((2, 3))
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=data, flip=[True, False])
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        flip_resource = next(r for r in group.resources if r["kind"] == "flip")
        flip = dp.data[dp.resources.index(flip_resource)]
        assert flip[0] is np.bool_(True)
        assert flip[1] is np.bool_(False)

    def test_multiple_entries_create_multiple_groups(self):
        dp = create_datapackage()
        e1 = ArrayEntry(rows=[0], cols=[1], data=np.ones((1, 2)))
        e2 = ArrayEntry(rows=[2], cols=[3], data=np.ones((1, 5)))
        dp.add_array_entries(matrix="technosphere_matrix", entries=[e1, e2])
        assert len(dp.groups) == 2

    def test_no_flip_resource_when_flip_is_none(self):
        dp = create_datapackage()
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 3)))
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        kinds = [r["kind"] for r in group.resources]
        assert "flip" not in kinds

    def test_no_rescale_resource_when_rescale_is_none(self):
        dp = create_datapackage()
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 3)))
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        kinds = [r["kind"] for r in group.resources]
        assert "rescale" not in kinds

    def test_rescale_stored_correctly(self):
        dp = create_datapackage()
        entry = ArrayEntry(rows=[0, 1], cols=[2, 3], data=np.ones((2, 3)), rescale=[2.0, 0.5])
        dp.add_array_entries(matrix="technosphere_matrix", entries=[entry])
        group = next(iter(dp.groups.values()))
        rescale_resource = next(r for r in group.resources if r["kind"] == "rescale")
        stored = dp.data[dp.resources.index(rescale_resource)]
        np.testing.assert_array_almost_equal(stored, [2.0, 0.5])


class TestAddEntries:
    def test_no_rescale_resource_when_all_rescale_one(self):
        dp = create_datapackage()
        entries = [
            MatrixEntry(row=1, col=2, amount=1.0),
            MatrixEntry(row=3, col=4, amount=2.0),
        ]
        dp.add_entries(matrix="technosphere_matrix", entries=entries)
        group = next(iter(dp.groups.values()))
        kinds = [r["kind"] for r in group.resources]
        assert "rescale" not in kinds

    def test_rescale_resource_stored_when_rescale_set(self):
        dp = create_datapackage()
        entries = [
            MatrixEntry(row=1, col=2, amount=1.0, rescale=0.5),
            MatrixEntry(row=3, col=4, amount=2.0, rescale=2.0),
        ]
        dp.add_entries(matrix="technosphere_matrix", entries=entries)
        group = next(iter(dp.groups.values()))
        rescale_resource = next(r for r in group.resources if r["kind"] == "rescale")
        stored = dp.data[dp.resources.index(rescale_resource)]
        np.testing.assert_array_almost_equal(sorted(stored), [0.5, 2.0])

    def test_rescale_resource_written_when_only_some_entries_rescaled(self):
        dp = create_datapackage()
        entries = [
            MatrixEntry(row=1, col=2, amount=1.0),           # rescale=1.0 (default)
            MatrixEntry(row=3, col=4, amount=2.0, rescale=0.5),
        ]
        dp.add_entries(matrix="technosphere_matrix", entries=entries)
        group = next(iter(dp.groups.values()))
        idx_resource = next(r for r in group.resources if r["kind"] == "indices")
        rescale_resource = next(r for r in group.resources if r["kind"] == "rescale")
        indices = dp.data[dp.resources.index(idx_resource)]
        rescales = dp.data[dp.resources.index(rescale_resource)]
        for i, idx in enumerate(indices):
            if idx["row"] == 1:
                assert rescales[i] == pytest.approx(1.0)
            else:
                assert rescales[i] == pytest.approx(0.5)

    def test_rescale_sorted_with_data(self):
        dp = create_datapackage()
        entries = [
            MatrixEntry(row=3, col=4, amount=2.0, rescale=2.0),
            MatrixEntry(row=1, col=2, amount=1.0, rescale=0.5),
        ]
        dp.add_entries(matrix="technosphere_matrix", entries=entries)
        group = next(iter(dp.groups.values()))

        idx_resource = next(r for r in group.resources if r["kind"] == "indices")
        rescale_resource = next(r for r in group.resources if r["kind"] == "rescale")
        indices = dp.data[dp.resources.index(idx_resource)]
        rescales = dp.data[dp.resources.index(rescale_resource)]

        for i, idx in enumerate(indices):
            if idx["row"] == 1:
                assert rescales[i] == pytest.approx(0.5)
            else:
                assert rescales[i] == pytest.approx(2.0)


class TestSimpleGraphDeprecation:
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simple_graph({"technosphere": [(1, 4, 2.5)]})
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "create_datapackage_from_entries" in str(w[0].message)
