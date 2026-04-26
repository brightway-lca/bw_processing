import math
import warnings

import numpy as np
import pytest

from bw_processing import (
    MatrixEntry,
    MatrixName,
    create_datapackage_from_entries,
    simple_graph,
)
from bw_processing.constants import UNCERTAINTY_DTYPE


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
        assert math.isnan(e.loc)
        assert math.isnan(e.scale)
        assert math.isnan(e.shape)
        assert math.isnan(e.minimum)
        assert math.isnan(e.maximum)

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

class TestSimpleGraphDeprecation:
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simple_graph({"technosphere": [(1, 4, 2.5)]})
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "create_datapackage_from_entries" in str(w[0].message)
