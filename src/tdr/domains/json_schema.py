"""
JSON schema domain for realistic verifier-guided repair.

Provides a finite-variable encoding of JSON documents matching a schema.
Each field in the schema becomes a variable; the verifier checks schema
compliance including cross-field constraints.

This is the "realistic domain" bridge between toy CSPs and practical
LLM use cases like config repair, API response correction, etc.
"""

import json
from typing import Optional

import numpy as np
import jsonschema
from jsonschema import validate, ValidationError

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics


# ---------------------------------------------------------------------------
# Schema definition: user profile with cross-field constraints
# ---------------------------------------------------------------------------

USER_SCHEMA = {
    "type": "object",
    "properties": {
        "username": {"type": "string", "enum": ["alice", "bob", "charlie", "diana"]},
        "age": {"type": "integer", "minimum": 18, "maximum": 80},
        "role": {"type": "string", "enum": ["admin", "user", "guest", "moderator"]},
        "clearance": {"type": "string", "enum": ["low", "medium", "high"]},
        "active": {"type": "boolean"},
        "tier": {"type": "string", "enum": ["free", "pro", "enterprise"]},
        "region": {"type": "string", "enum": ["us", "eu", "apac"]},
    },
    "required": ["username", "age", "role", "clearance", "active", "tier", "region"],
}


def check_cross_field_constraints(instance: dict) -> list[str]:
    """Return list of constraint violation descriptions."""
    violations = []
    if instance.get("role") == "admin" and instance.get("clearance") != "high":
        violations.append("admin requires high clearance")
    return violations


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

FIELD_NAMES = ["username", "age", "role", "clearance", "active", "tier", "region"]
FIELD_DOMAINS = {
    "username": ["alice", "bob", "charlie", "diana"],
    "age": list(range(18, 81)),
    "role": ["admin", "user", "guest", "moderator"],
    "clearance": ["low", "medium", "high"],
    "active": [True, False],
    "tier": ["free", "pro", "enterprise"],
    "region": ["us", "eu", "apac"],
}
FIELD_TO_IDX = {name: i for i, name in enumerate(FIELD_NAMES)}
N_FIELDS = len(FIELD_NAMES)


def encode_instance(instance: dict) -> np.ndarray:
    """Convert JSON dict to integer-coded array."""
    x = np.zeros(N_FIELDS, dtype=np.int64)
    for name, idx in FIELD_TO_IDX.items():
        val = instance[name]
        domain = FIELD_DOMAINS[name]
        x[idx] = domain.index(val)
    return x


def decode_array(x: np.ndarray) -> dict:
    """Convert integer-coded array back to JSON dict."""
    instance = {}
    for name, idx in FIELD_TO_IDX.items():
        val_idx = x[idx]
        if val_idx == MASK:
            instance[name] = None
        else:
            instance[name] = FIELD_DOMAINS[name][val_idx]
    return instance


class JsonSchemaDomain(FiniteReasoningDomain):
    """JSON schema domain with cross-field constraints.

    Variables are fields of a JSON document. The verifier checks:
    1. JSON Schema compliance (types, enums, ranges)
    2. Cross-field constraints (e.g., admin requires high clearance)

    Local residuals indicate which fields participate in violations.
    """

    def __init__(self, schema: Optional[dict] = None):
        self.schema = schema or USER_SCHEMA
        self._field_names = list(self.schema["properties"].keys())
        self._n = len(self._field_names)
        self._domains = {}
        for name, props in self.schema["properties"].items():
            if "enum" in props:
                self._domains[name] = props["enum"]
            elif props["type"] == "boolean":
                self._domains[name] = [True, False]
            elif props["type"] == "integer":
                lo = props.get("minimum", 0)
                hi = props.get("maximum", 100)
                self._domains[name] = list(range(lo, hi + 1))
            elif props["type"] == "string":
                self._domains[name] = [f"val_{i}" for i in range(10)]
            else:
                self._domains[name] = [0]

    def num_variables(self) -> int:
        return self._n

    def domain_size(self, i: int) -> int:
        return len(self._domains[self._field_names[i]])

    def max_domain_size(self) -> int:
        return max(len(d) for d in self._domains.values())

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random valid JSON instance."""
        instance = {}
        for name in self._field_names:
            domain = self._domains[name]
            instance[name] = domain[rng.integers(len(domain))]
        # Enforce cross-field constraints
        if instance.get("role") == "admin":
            instance["clearance"] = "high"
        return encode_instance(instance)

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Check schema compliance and cross-field constraints.

        Returns global violation count and per-field local residuals.
        """
        if x.shape != (self._n,):
            raise ValueError(f"Expected shape ({self._n},), got {x.shape}")

        n = self._n
        violations = []

        # Check each field individually against schema
        for name, idx in FIELD_TO_IDX.items():
            if x[idx] == MASK:
                continue
            val = FIELD_DOMAINS[name][x[idx]]
            field_schema = {
                "type": "object",
                "properties": {name: self.schema["properties"][name]},
                "required": [name],
            }
            try:
                validate({name: val}, field_schema)
            except ValidationError:
                violations.append(idx)

        # Check cross-field constraints (on full decoded instance)
        if not np.any(x == MASK):
            instance = decode_array(x)
            constraint_violations = check_cross_field_constraints(instance)
            if constraint_violations:
                violations.append(FIELD_TO_IDX["role"])
                violations.append(FIELD_TO_IDX["clearance"])

        # Build diagnostics
        unique_violations = set(violations)
        global_violation = len(unique_violations)
        local_residuals = np.zeros(n, dtype=np.int64)
        for idx in violations:
            local_residuals[idx] += 1

        return VerifierDiagnostics(
            global_violation=int(global_violation),
            local_residuals=local_residuals,
        )

    def build_factors(self) -> list[Factor]:
        return []

    def enumerate_solutions(self) -> np.ndarray:
        raise NotImplementedError("JSON domain has a continuous solution space")
