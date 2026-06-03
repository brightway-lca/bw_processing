import numpy as np
import pytest

from bw_processing import (
    INDICES_DTYPE,
    ParamLabelField,
    ParamLabelSchema,
    StringLabelSchema,
    create_datapackage,
    generic_directory_filesystem,
    load_datapackage,
    schema_from_json_schema,
)
from bw_processing.errors import ShapeMismatch, WrongDatatype


# ---------------------------------------------------------------------------
# StringLabelSchema
# ---------------------------------------------------------------------------


def test_string_label_schema_to_json_schema():
    assert StringLabelSchema().to_json_schema() == {"type": "string"}


def test_string_label_schema_with_description():
    s = StringLabelSchema(description="A name")
    assert s.to_json_schema() == {"type": "string", "description": "A name"}


def test_string_label_schema_round_trip():
    s = StringLabelSchema(description="test")
    assert StringLabelSchema.from_json_schema(s.to_json_schema()) == s


def test_string_label_schema_validation_passes():
    StringLabelSchema().validate(["temperature", "pressure"])


def test_string_label_schema_validation_fails():
    with pytest.raises(Exception):  # jsonschema.ValidationError
        StringLabelSchema().validate([{"not": "a string"}])


# ---------------------------------------------------------------------------
# ParamLabelField / ParamLabelSchema
# ---------------------------------------------------------------------------


def test_param_label_schema_to_json_schema():
    schema = ParamLabelSchema(
        fields=[
            ParamLabelField(name="name", type="string", required=True),
            ParamLabelField(name="value", type="number", required=False),
        ]
    )
    result = schema.to_json_schema()
    assert result["type"] == "object"
    assert set(result["properties"]) == {"name", "value"}
    assert result["required"] == ["name"]
    assert "value" not in result.get("required", [])


def test_param_label_schema_no_required_omitted():
    schema = ParamLabelSchema(
        fields=[ParamLabelField(name="x", type="number", required=False)]
    )
    result = schema.to_json_schema()
    assert "required" not in result


def test_param_label_schema_field_description():
    schema = ParamLabelSchema(
        fields=[ParamLabelField(name="db", type="string", description="Database name")]
    )
    result = schema.to_json_schema()
    assert result["properties"]["db"]["description"] == "Database name"


def test_param_label_schema_round_trip():
    schema = ParamLabelSchema(
        fields=[
            ParamLabelField(name="database", type="string", required=True, description="DB"),
            ParamLabelField(name="amount", type="number", required=False),
        ]
    )
    assert ParamLabelSchema.from_json_schema(schema.to_json_schema()) == schema


def test_param_label_schema_validation_passes():
    schema = ParamLabelSchema(fields=[ParamLabelField(name="name", type="string")])
    schema.validate([{"name": "electricity"}, {"name": "heat"}])


def test_param_label_schema_validation_fails_missing_required():
    schema = ParamLabelSchema(fields=[ParamLabelField(name="name", type="string", required=True)])
    with pytest.raises(Exception):  # jsonschema.ValidationError
        schema.validate([{"other": "field"}])


def test_param_label_schema_validation_fails_wrong_type():
    schema = ParamLabelSchema(fields=[ParamLabelField(name="amount", type="number")])
    with pytest.raises(Exception):
        schema.validate([{"amount": "not-a-number"}])


# ---------------------------------------------------------------------------
# schema_from_json_schema factory
# ---------------------------------------------------------------------------


def test_schema_from_json_schema_string():
    assert isinstance(schema_from_json_schema({"type": "string"}), StringLabelSchema)


def test_schema_from_json_schema_object():
    result = schema_from_json_schema(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    )
    assert isinstance(result, ParamLabelSchema)
    assert len(result.fields) == 1
    assert result.fields[0].name == "name"
    assert result.fields[0].required is True


# ---------------------------------------------------------------------------
# add_persistent_vector with params
# ---------------------------------------------------------------------------


def _make_indices(n=1):
    return np.array([(i, i + 1) for i in range(n)], dtype=INDICES_DTYPE)


def test_persistent_vector_params_only():
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0, 1.013]),
        name="test",
    )
    kinds = {r["kind"] for r in dp.resources}
    assert "params" in kinds
    assert "param_labels" not in kinds


def test_persistent_vector_params_with_string_labels():
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0, 1.013]),
        param_labels=["temperature", "pressure"],
        name="test",
    )
    kinds = {r["kind"] for r in dp.resources}
    assert "params" in kinds
    assert "param_labels" in kinds
    labels_data, _ = dp.get_resource("test.param_labels")
    assert labels_data["values"] == ["temperature", "pressure"]
    assert "schema" not in labels_data


def test_persistent_vector_params_with_schema():
    dp = create_datapackage()
    schema = ParamLabelSchema(
        fields=[
            ParamLabelField(name="name", type="string"),
            ParamLabelField(name="namespace", type="string"),
        ]
    )
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0]),
        param_labels=[{"name": "temperature", "namespace": "IEA"}],
        param_label_schema=schema,
        name="test",
    )
    labels_data, _ = dp.get_resource("test.param_labels")
    assert labels_data["values"] == [{"name": "temperature", "namespace": "IEA"}]
    assert labels_data["schema"]["type"] == "object"
    assert "required" in labels_data["schema"]


def test_persistent_vector_params_with_string_schema():
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0]),
        param_labels=["temperature"],
        param_label_schema=StringLabelSchema(),
        name="test",
    )
    labels_data, _ = dp.get_resource("test.param_labels")
    assert labels_data["schema"] == {"type": "string"}


def test_persistent_vector_params_all_resources_in_group():
    dp = create_datapackage()
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0]),
        param_labels=["temperature"],
        name="test",
    )
    groups = dp.groups
    assert "test" in groups
    group_kinds = {r["kind"] for r in groups["test"].resources}
    assert group_kinds == {"indices", "data", "params", "param_labels"}


# ---------------------------------------------------------------------------
# Error cases for persistent_vector
# ---------------------------------------------------------------------------


def test_persistent_vector_params_wrong_dtype():
    dp = create_datapackage()
    with pytest.raises(WrongDatatype):
        dp.add_persistent_vector(
            matrix="technosphere",
            indices_array=_make_indices(),
            params_array=np.array([1, 2], dtype=int),
            name="test",
        )


def test_persistent_vector_params_must_be_1d():
    dp = create_datapackage()
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="technosphere",
            indices_array=_make_indices(),
            params_array=np.array([[1.0, 2.0]]),
            name="test",
        )


def test_persistent_vector_labels_without_params_raises():
    dp = create_datapackage()
    with pytest.raises(ValueError):
        dp.add_persistent_vector(
            matrix="technosphere",
            indices_array=_make_indices(),
            param_labels=["temperature"],
            name="test",
        )


def test_persistent_vector_schema_without_labels_raises():
    dp = create_datapackage()
    with pytest.raises(ValueError):
        dp.add_persistent_vector(
            matrix="technosphere",
            indices_array=_make_indices(),
            params_array=np.array([1.0]),
            param_label_schema=StringLabelSchema(),
            name="test",
        )


def test_persistent_vector_labels_length_mismatch():
    dp = create_datapackage()
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_vector(
            matrix="technosphere",
            indices_array=_make_indices(),
            params_array=np.array([1.0, 2.0]),
            param_labels=["only_one_label"],
            name="test",
        )


# ---------------------------------------------------------------------------
# add_persistent_array with params
# ---------------------------------------------------------------------------


def test_persistent_array_params():
    dp = create_datapackage()
    indices = _make_indices(2)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 exchanges, 3 scenarios
    params = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])  # 2 params, 3 scenarios
    dp.add_persistent_array(
        matrix="technosphere",
        indices_array=indices,
        data_array=data,
        params_array=params,
        param_labels=["temperature", "pressure"],
        name="test",
    )
    kinds = {r["kind"] for r in dp.resources}
    assert "params" in kinds
    assert "param_labels" in kinds
    params_data, _ = dp.get_resource("test.params")
    assert params_data.shape == (2, 3)


def test_persistent_array_params_column_mismatch():
    dp = create_datapackage()
    indices = _make_indices(1)
    data = np.array([[1.0, 2.0, 3.0]])  # 3 scenarios
    params = np.array([[10.0, 20.0]])   # only 2 scenarios — mismatch
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_array(
            matrix="technosphere",
            indices_array=indices,
            data_array=data,
            params_array=params,
            name="test",
        )


def test_persistent_array_params_must_be_2d():
    dp = create_datapackage()
    indices = _make_indices(1)
    data = np.array([[1.0, 2.0]])
    with pytest.raises(ShapeMismatch):
        dp.add_persistent_array(
            matrix="technosphere",
            indices_array=indices,
            data_array=data,
            params_array=np.array([1.0, 2.0]),  # 1D — wrong for array
            name="test",
        )


# ---------------------------------------------------------------------------
# Round-trip (serialize + reload)
# ---------------------------------------------------------------------------


def test_round_trip_with_params_and_labels(tmp_path):
    fs = generic_directory_filesystem(dirpath=tmp_path / "dp")
    dp = create_datapackage(fs=fs)
    schema = ParamLabelSchema(
        fields=[ParamLabelField(name="name", type="string"), ParamLabelField(name="db", type="string")]
    )
    dp.add_persistent_vector(
        matrix="technosphere",
        indices_array=_make_indices(),
        data_array=np.array([1.0]),
        params_array=np.array([25.0]),
        param_labels=[{"name": "temperature", "db": "ecoinvent"}],
        param_label_schema=schema,
        name="test",
    )
    dp.finalize_serialization()

    dp2 = load_datapackage(generic_directory_filesystem(dirpath=tmp_path / "dp"))
    params_data, _ = dp2.get_resource("test.params")
    labels_data, _ = dp2.get_resource("test.param_labels")

    np.testing.assert_array_equal(params_data, np.array([25.0]))
    assert labels_data["values"] == [{"name": "temperature", "db": "ecoinvent"}]
    reconstructed = schema_from_json_schema(labels_data["schema"])
    assert isinstance(reconstructed, ParamLabelSchema)
    assert reconstructed == schema
