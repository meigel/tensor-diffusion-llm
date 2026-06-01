"""
JSON schema v2: rich cloud infrastructure configuration domain.

Provides a finite-variable encoding of a cloud resource configuration
with realistic cross-field constraints. Designed to be hard enough
that different denoisers and policies produce distinguishable results.

Domain: 18 fields, 8 cross-field constraints, ambiguous repair targets.

Fields
------
provider, service_type, region, instance_size, cpu_cores, memory_gb,
storage_gb, encryption, backup_enabled, backup_retention_days, multi_az,
environment, auto_scaling, max_instances, logging_level, monitoring,
cost_tier, compliance

Cross-field constraints
-----------------------
1. Provider → supported regions
2. Encryption → multi_az + backup
3. Production → backup + monitoring + logging
4. Database → multi_az + backup + min storage
5. Compliance → encryption + backup + logging + monitoring
6. Free tier → instance size + scaling limits
7. Large storage → instance size + memory
8. Serverless → auto scaling + instance size
"""

import functools
from typing import Optional

import numpy as np
from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    "provider", "service_type", "region", "instance_size",
    "cpu_cores", "memory_gb", "storage_gb",
    "encryption", "backup_enabled", "backup_retention_days", "multi_az",
    "environment", "auto_scaling", "max_instances",
    "logging_level", "monitoring", "cost_tier", "compliance",
]

FIELD_DOMAINS = {
    "provider": ["aws", "gcp", "azure"],
    "service_type": ["compute", "storage", "database", "serverless", "networking"],
    "region": ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
               "ap-southeast-1", "ap-northeast-1"],
    "instance_size": ["small", "medium", "large", "xlarge"],
    "cpu_cores": [1, 2, 4, 8, 16, 32],
    "memory_gb": [1, 2, 4, 8, 16, 32, 64, 128],
    "storage_gb": [10, 50, 100, 500, 1000, 2000, 5000],
    "encryption": ["enabled", "disabled"],
    "backup_enabled": [True, False],
    "backup_retention_days": [1, 7, 14, 30, 90, 365],
    "multi_az": [True, False],
    "environment": ["dev", "staging", "production"],
    "auto_scaling": [True, False],
    "max_instances": [1, 2, 5, 10, 20, 50],
    "logging_level": ["debug", "info", "warn", "error"],
    "monitoring": ["enabled", "basic", "disabled"],
    "cost_tier": ["free", "standard", "premium", "enterprise"],
    "compliance": ["none", "soc2", "hipaa", "pci_dss"],
}

FIELD_TO_IDX = {name: i for i, name in enumerate(FIELD_NAMES)}
N_FIELDS = len(FIELD_NAMES)
MAX_DOMAIN = max(len(d) for d in FIELD_DOMAINS.values())

# Provider-region mapping
PROVIDER_REGIONS = {
    "aws": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
    "gcp": ["eu-west-1", "ap-southeast-1"],
    "azure": ["us-east-1", "eu-west-1", "eu-central-1"],
}

# Providers with region 'us-west-2' and 'ap-northeast-1' as fallback
# us-west-2 is aws-only, ap-northeast-1 is available on all


def encode_instance(instance: dict) -> np.ndarray:
    """Convert dict to integer-coded array."""
    x = np.zeros(N_FIELDS, dtype=np.int64)
    for name, idx in FIELD_TO_IDX.items():
        val = instance[name]
        x[idx] = FIELD_DOMAINS[name].index(val)
    return x


def decode_array(x: np.ndarray) -> dict:
    """Convert integer-coded array back to dict."""
    instance = {}
    for name, idx in FIELD_TO_IDX.items():
        val_idx = x[idx]
        if val_idx == MASK or val_idx < 0:
            instance[name] = None
        elif val_idx >= len(FIELD_DOMAINS[name]):
            instance[name] = f"INVALID_{val_idx}"
        else:
            instance[name] = FIELD_DOMAINS[name][val_idx]
    return instance


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def check_constraints(instance: dict) -> list[tuple[str, list[str]]]:
    """Check all cross-field constraints.

    Returns list of (constraint_description, involved_fields) tuples.
    """
    violations = []

    # --- C1: Provider-region compatibility ---
    provider = instance.get("provider")
    region = instance.get("region")
    if provider is not None and region is not None:
        supported = PROVIDER_REGIONS.get(provider, [])
        if region not in supported:
            violations.append((
                f"{provider} does not support region {region}",
                ["provider", "region"],
            ))

    # --- C2: Encryption requires multi-AZ and backup ---
    encryption = instance.get("encryption")
    if encryption == "enabled":
        if instance.get("multi_az") is False:
            violations.append((
                "encryption requires multi-AZ",
                ["encryption", "multi_az"],
            ))
        if instance.get("backup_enabled") is False:
            violations.append((
                "encryption requires backup",
                ["encryption", "backup_enabled"],
            ))

    # --- C3: Production requirements ---
    env = instance.get("environment")
    if env == "production":
        if instance.get("backup_enabled") is False:
            violations.append((
                "production requires backup",
                ["environment", "backup_enabled"],
            ))
        retention = instance.get("backup_retention_days")
        if retention is not None and retention < 30:
            violations.append((
                "production requires backup_retention_days >= 30",
                ["environment", "backup_retention_days"],
            ))
        if instance.get("monitoring") == "disabled":
            violations.append((
                "production requires monitoring",
                ["environment", "monitoring"],
            ))
        if instance.get("logging_level") == "debug":
            violations.append((
                "production cannot use debug logging",
                ["environment", "logging_level"],
            ))

    # --- C4: Database service requirements ---
    st = instance.get("service_type")
    if st == "database":
        if instance.get("multi_az") is False:
            violations.append((
                "database requires multi-AZ",
                ["service_type", "multi_az"],
            ))
        if instance.get("backup_enabled") is False:
            violations.append((
                "database requires backup",
                ["service_type", "backup_enabled"],
            ))
        storage = instance.get("storage_gb")
        if storage is not None and storage < 100:
            violations.append((
                "database requires storage_gb >= 100",
                ["service_type", "storage_gb"],
            ))

    # --- C5: Compliance requirements ---
    compliance = instance.get("compliance")
    if compliance in ("hipaa", "pci_dss"):
        if instance.get("encryption") != "enabled":
            violations.append((
                f"{compliance} requires encryption",
                ["compliance", "encryption"],
            ))
        if instance.get("backup_enabled") is False:
            violations.append((
                f"{compliance} requires backup",
                ["compliance", "backup_enabled"],
            ))
        if instance.get("monitoring") == "disabled":
            violations.append((
                f"{compliance} requires monitoring",
                ["compliance", "monitoring"],
            ))
        if instance.get("logging_level") == "debug":
            violations.append((
                f"{compliance} cannot use debug logging",
                ["compliance", "logging_level"],
            ))

    # --- C6: Free tier limits ---
    cost = instance.get("cost_tier")
    if cost == "free":
        inst_size = instance.get("instance_size")
        if inst_size not in ("small", "medium", None):
            violations.append((
                "free tier requires small or medium instance",
                ["cost_tier", "instance_size"],
            ))
        if instance.get("auto_scaling") is True:
            violations.append((
                "free tier cannot use auto-scaling",
                ["cost_tier", "auto_scaling"],
            ))
        max_inst = instance.get("max_instances")
        if max_inst is not None and max_inst > 2:
            violations.append((
                "free tier max_instances <= 2",
                ["cost_tier", "max_instances"],
            ))

    # --- C7: Large storage requires large instance ---
    storage = instance.get("storage_gb")
    if storage is not None and storage > 1000:
        inst_size = instance.get("instance_size")
        if inst_size not in ("large", "xlarge", None):
            violations.append((
                "storage > 1000 GB requires large or xlarge instance",
                ["storage_gb", "instance_size"],
            ))
        memory = instance.get("memory_gb")
        if memory is not None and memory < 16:
            violations.append((
                "storage > 1000 GB requires memory_gb >= 16",
                ["storage_gb", "memory_gb"],
            ))

    # --- C8: Serverless constraints ---
    if st == "serverless":
        if instance.get("auto_scaling") is False:
            violations.append((
                "serverless requires auto-scaling",
                ["service_type", "auto_scaling"],
            ))
        max_inst = instance.get("max_instances")
        if max_inst is not None and max_inst < 5:
            violations.append((
                "serverless requires max_instances >= 5",
                ["service_type", "max_instances"],
            ))
        inst_size = instance.get("instance_size")
        if inst_size not in ("small", None):
            violations.append((
                "serverless requires small instance size",
                ["service_type", "instance_size"],
            ))

    return violations


# ---------------------------------------------------------------------------
# Domain class
# ---------------------------------------------------------------------------

class JsonSchemaV2Domain(FiniteReasoningDomain):
    """Cloud infrastructure config domain with 8 cross-field constraints.

    18 variables, max domain size 8, 8 constraint groups.
    Designed to create ambiguous repair scenarios where fixing one
    constraint can break another.
    """

    def __init__(self):
        pass

    def num_variables(self) -> int:
        return N_FIELDS

    def domain_size(self, i: int) -> int:
        return len(FIELD_DOMAINS[FIELD_NAMES[i]])

    def max_domain_size(self) -> int:
        return MAX_DOMAIN

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a valid cloud config satisfying all constraints."""
        # Strategy: sample forward, fixing constraints greedily
        for attempt in range(100):
            provider = rng.choice(FIELD_DOMAINS["provider"])
            supported_regions = PROVIDER_REGIONS[provider]
            region = rng.choice(supported_regions)

            service_type = rng.choice(FIELD_DOMAINS["service_type"])

            # Instance size depends on service_type and constraints
            if service_type == "serverless":
                instance_size = "small"
            else:
                instance_size = rng.choice(FIELD_DOMAINS["instance_size"])

            cpu_cores = int(rng.choice(FIELD_DOMAINS["cpu_cores"]))
            memory_gb = int(rng.choice(FIELD_DOMAINS["memory_gb"]))
            storage_gb = int(rng.choice(FIELD_DOMAINS["storage_gb"]))

            # C7: Large storage → instance size constraint
            if storage_gb > 1000:
                instance_size = rng.choice(["large", "xlarge"])
                memory_gb = int(rng.choice([m for m in FIELD_DOMAINS["memory_gb"] if m >= 16]))

            # Cost tier
            cost_tier = rng.choice(FIELD_DOMAINS["cost_tier"])

            # C6: Free tier limits
            auto_scaling = bool(rng.choice(FIELD_DOMAINS["auto_scaling"]))
            if cost_tier == "free":
                instance_size = rng.choice(["small", "medium"])
                auto_scaling = False
                max_instances = int(rng.choice([1, 2]))
            else:
                max_instances = int(rng.choice(FIELD_DOMAINS["max_instances"]))

            # C8: Serverless constraints
            if service_type == "serverless":
                auto_scaling = True
                max_instances = int(rng.choice([m for m in FIELD_DOMAINS["max_instances"] if m >= 5]))

            # Environment
            environment = rng.choice(FIELD_DOMAINS["environment"])

            # C4: Database requirements
            if service_type == "database":
                multi_az = True
                backup_enabled = True
                storage_gb = int(rng.choice([s for s in FIELD_DOMAINS["storage_gb"] if s >= 100]))
            else:
                multi_az = bool(rng.choice(FIELD_DOMAINS["multi_az"]))
                backup_enabled = bool(rng.choice(FIELD_DOMAINS["backup_enabled"]))

            encryption = rng.choice(FIELD_DOMAINS["encryption"])

            # C2: Encryption → multi_az + backup
            if encryption == "enabled":
                multi_az = True
                backup_enabled = True

            # C5: Compliance
            compliance = rng.choice(FIELD_DOMAINS["compliance"])
            if compliance in ("hipaa", "pci_dss"):
                encryption = "enabled"
                backup_enabled = True

            # Backup retention
            backup_retention_days = int(rng.choice(FIELD_DOMAINS["backup_retention_days"]))

            # C3: Production requirements
            if environment == "production":
                backup_enabled = True
                if backup_retention_days < 30:
                    backup_retention_days = 30

            # C5 (continued): Compliance monitoring/logging
            logging_level = rng.choice(FIELD_DOMAINS["logging_level"])
            monitoring = rng.choice(FIELD_DOMAINS["monitoring"])
            if compliance in ("hipaa", "pci_dss") or environment == "production":
                if monitoring == "disabled":
                    monitoring = "basic"
                if logging_level == "debug":
                    logging_level = "info"

            instance = {
                "provider": provider,
                "service_type": service_type,
                "region": region,
                "instance_size": instance_size,
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "storage_gb": storage_gb,
                "encryption": encryption,
                "backup_enabled": backup_enabled,
                "backup_retention_days": backup_retention_days,
                "multi_az": multi_az,
                "environment": environment,
                "auto_scaling": auto_scaling,
                "max_instances": max_instances,
                "logging_level": logging_level,
                "monitoring": monitoring,
                "cost_tier": cost_tier,
                "compliance": compliance,
            }

            # Verify all constraints satisfied
            violations = check_constraints(instance)
            if len(violations) == 0:
                return encode_instance(instance)

        # Fallback: return something valid-ish
        raise RuntimeError("Could not generate valid instance after 100 attempts")

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Check all constraints, return global and local residuals."""
        n = self.num_variables()
        if x.shape != (n,):
            raise ValueError(f"Expected shape ({n},), got {x.shape}")

        instance = decode_array(x)
        violations = check_constraints(instance)

        # Build per-field violation counts
        field_violations = {}
        for desc, fields in violations:
            for f in fields:
                idx = FIELD_TO_IDX[f]
                field_violations[idx] = field_violations.get(idx, 0) + 1

        local_residuals = np.zeros(n, dtype=np.int64)
        for idx, count in field_violations.items():
            local_residuals[idx] = count

        return VerifierDiagnostics(
            global_violation=len(violations),
            local_residuals=local_residuals,
        )

    def build_factors(self) -> list[Factor]:
        return []

    def enumerate_solutions(self) -> np.ndarray:
        raise NotImplementedError("JSON-v2 is too large for enumeration")
