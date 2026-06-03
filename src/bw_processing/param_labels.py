import jsonschema
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

VALID_FIELD_TYPES = ("string", "integer", "number", "boolean")


@dataclass
class StringLabelSchema:
    """Schema for param labels that are plain strings."""

    description: Optional[str] = None

    def to_json_schema(self) -> Dict:
        s: Dict = {"type": "string"}
        if self.description:
            s["description"] = self.description
        return s

    def validate(self, values: List) -> None:
        schema = self.to_json_schema()
        for value in values:
            jsonschema.validate(value, schema)

    @classmethod
    def from_json_schema(cls, data: Dict) -> "StringLabelSchema":
        return cls(description=data.get("description"))


@dataclass
class ParamLabelField:
    """A single field in a structured param label."""

    name: str
    type: Literal["string", "integer", "number", "boolean"] = "string"
    required: bool = True
    description: Optional[str] = None


@dataclass
class ParamLabelSchema:
    """Schema for param labels that are structured objects."""

    fields: List[ParamLabelField] = field(default_factory=list)
    description: Optional[str] = None

    def to_json_schema(self) -> Dict:
        s: Dict = {
            "type": "object",
            "properties": {
                f.name: {
                    k: v
                    for k, v in {"type": f.type, "description": f.description}.items()
                    if v is not None
                }
                for f in self.fields
            },
        }
        required = [f.name for f in self.fields if f.required]
        if required:
            s["required"] = required
        if self.description:
            s["description"] = self.description
        return s

    def validate(self, values: List) -> None:
        schema = self.to_json_schema()
        for value in values:
            jsonschema.validate(value, schema)

    @classmethod
    def from_json_schema(cls, data: Dict) -> "ParamLabelSchema":
        required = set(data.get("required", []))
        fields = []
        for name, defn in data.get("properties", {}).items():
            type_val = defn.get("type", "string")
            if type_val not in VALID_FIELD_TYPES:
                raise ValueError(
                    f"Unknown field type {type_val!r} for field {name!r}; "
                    f"must be one of {VALID_FIELD_TYPES}"
                )
            fields.append(
                ParamLabelField(
                    name=name,
                    type=type_val,
                    required=name in required,
                    description=defn.get("description"),
                )
            )
        return cls(fields=fields, description=data.get("description"))


AnyLabelSchema = Union[StringLabelSchema, ParamLabelSchema]


def schema_from_json_schema(data: Dict) -> AnyLabelSchema:
    """Reconstruct a StringLabelSchema or ParamLabelSchema from a stored JSON Schema dict."""
    if data.get("type") == "string":
        return StringLabelSchema.from_json_schema(data)
    return ParamLabelSchema.from_json_schema(data)
