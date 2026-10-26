"""
Rungate Schema Validator — JSON Schema validation for tool-call arguments.

Validates that tool-call arguments conform to a defined schema before execution.
Uses jsonschema if available, falls back to a lightweight built-in validator.

Schema format (JSON Schema compatible):
    schemas:
      web_search:
        type: object
        required: [query]
        properties:
          query:
            type: string
            minLength: 1
            maxLength: 1000
          max_results:
            type: integer
            minimum: 1
            maximum: 100

      send_email:
        type: object
        required: [to, subject, body]
        properties:
          to:
            type: string
            pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
          subject:
            type: string
            maxLength: 200
          body:
            type: string
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Try to use jsonschema if available, fall back to built-in
_HAS_JSONSCHEMA = False
try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    pass


class SchemaValidator:
    """
    Validates tool-call arguments against JSON schemas.

    Usage:
        validator = SchemaValidator({
            "web_search": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1}
                }
            }
        })

        is_valid, error = validator.validate("web_search", {"query": "test"})
        # → (True, None)

        is_valid, error = validator.validate("web_search", {})
        # → (False, "Missing required field: query")
    """

    def __init__(self, schemas: Dict[str, Dict[str, Any]]):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        for tool_name, schema in schemas.items():
            self._schemas[tool_name] = schema

    def validate(self, tool: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate tool arguments against schema.

        Returns:
            (True, None) if valid or no schema defined
            (False, error_message) if validation fails
        """
        schema = self._schemas.get(tool)
        if schema is None:
            # No schema defined — pass through
            return True, None

        if _HAS_JSONSCHEMA:
            return self._validate_jsonschema(schema, args)
        return self._validate_builtin(schema, args)

    def _validate_jsonschema(self, schema: Dict, args: Dict) -> Tuple[bool, Optional[str]]:
        """Validate using jsonschema library."""
        try:
            jsonschema.validate(instance=args, schema=schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e.message)
        except jsonschema.SchemaError as e:
            return False, f"Invalid schema: {e.message}"

    def _validate_builtin(self, schema: Dict, args: Dict) -> Tuple[bool, Optional[str]]:
        """Lightweight built-in validation (no dependencies)."""
        return self._validate_value(args, schema, path="")

    def _validate_value(
        self, value: Any, schema: Dict, path: str
    ) -> Tuple[bool, Optional[str]]:
        """Recursively validate a value against a schema."""
        expected_type = schema.get("type")

        # Type check
        if expected_type:
            if not self._check_type(value, expected_type):
                return False, f"Expected type '{expected_type}' at {path or 'root'}, got {type(value).__name__}"

        # Object validation
        if expected_type == "object":
            if not isinstance(value, dict):
                return False, f"Expected object at {path or 'root'}"

            # Required fields
            required = schema.get("required", [])
            for req_field in required:
                if req_field not in value:
                    return False, f"Missing required field: {req_field}"

            # Property validation
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in value:
                    ok, err = self._validate_value(
                        value[prop_name], prop_schema, f"{path}.{prop_name}"
                    )
                    if not ok:
                        return False, err

            # Additional properties
            if schema.get("additionalProperties") is False:
                extra = set(value.keys()) - set(properties.keys())
                if extra:
                    return False, f"Unexpected fields: {', '.join(sorted(extra))}"

        # String validation
        if expected_type == "string" and isinstance(value, str):
            min_len = schema.get("minLength")
            max_len = schema.get("maxLength")
            pattern = schema.get("pattern")

            if min_len is not None and len(value) < min_len:
                return False, f"String too short at {path}: min {min_len}, got {len(value)}"
            if max_len is not None and len(value) > max_len:
                return False, f"String too long at {path}: max {max_len}, got {len(value)}"
            if pattern is not None and not re.match(pattern, value):
                return False, f"String doesn't match pattern at {path}: {pattern}"

        # Number/integer validation
        if expected_type in ("number", "integer"):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            exclusive_min = schema.get("exclusiveMinimum")
            exclusive_max = schema.get("exclusiveMaximum")

            if minimum is not None and value < minimum:
                return False, f"Value too small at {path}: min {minimum}, got {value}"
            if maximum is not None and value > maximum:
                return False, f"Value too large at {path}: max {maximum}, got {value}"
            if exclusive_min is not None and value <= exclusive_min:
                return False, f"Value must be > {exclusive_min} at {path}, got {value}"
            if exclusive_max is not None and value >= exclusive_max:
                return False, f"Value must be < {exclusive_max} at {path}, got {value}"

        # Array validation
        if expected_type == "array" and isinstance(value, list):
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            item_schema = schema.get("items")

            if min_items is not None and len(value) < min_items:
                return False, f"Array too short at {path}: min {min_items}, got {len(value)}"
            if max_items is not None and len(value) > max_items:
                return False, f"Array too long at {path}: max {max_items}, got {len(value)}"

            if item_schema:
                for i, item in enumerate(value):
                    ok, err = self._validate_value(item, item_schema, f"{path}[{i}]")
                    if not ok:
                        return False, err

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                return False, f"Value not in enum at {path}: expected one of {schema['enum']}, got {value!r}"

        return True, None

    def _check_type(self, value: Any, expected: str) -> bool:
        """Check if a value matches the expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Unknown type, pass through
        return isinstance(value, expected_types)

    def add_schema(self, tool: str, schema: Dict[str, Any]) -> None:
        """Add or update a schema at runtime."""
        self._schemas[tool] = schema

    def remove_schema(self, tool: str) -> bool:
        """Remove a schema. Returns True if it existed."""
        return self._schemas.pop(tool, None) is not None

    def list_schemas(self) -> List[str]:
        """List all tools with schemas defined."""
        return list(self._schemas.keys())

    def get_schema(self, tool: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a tool."""
        return self._schemas.get(tool)

    def __repr__(self) -> str:
        return f"SchemaValidator(schemas={list(self._schemas.keys())})"
