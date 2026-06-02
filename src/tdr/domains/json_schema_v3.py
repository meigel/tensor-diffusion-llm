"""
JSON schema v3: tightly-coupled cloud infrastructure domain.

Extends JSON-v2 with 7 additional constraints that create genuine
trade-offs and contradictory requirements. Designed so that verifier
repair does NOT trivially achieve 100%, enabling method comparison.

Key difference from v2: constraints share variables in ways that
make it impossible to fix all simultaneously without changing
shared fields (cost_tier, service_type, compliance) — forcing the
denoiser to make choices and the verifier to identify which
constraints to prioritize.

Total constraints: 15 groups (8 from v2 + 7 new coupled ones)
"""

import numpy as np
from typing import Optional
from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics
from tdr.domains.json_schema_v2 import (
    FIELD_NAMES, FIELD_DOMAINS, FIELD_TO_IDX, N_FIELDS, MAX_DOMAIN,
    encode_instance, decode_array,
    PROVIDER_REGIONS,
)


# ---------------------------------------------------------------------------
# Base constraint checker (inherit v2 checks + add v3 coupled constraints)
# ---------------------------------------------------------------------------

def check_constraints_v2(instance: dict) -> list[tuple[str, list[str]]]:
    """JSON-v2 constraint checks (8 groups, imported from v2 logic)."""
    violations = []

    # C1: Provider-region
    provider = instance.get("provider")
    region = instance.get("region")
    if provider is not None and region is not None:
        supported = PROVIDER_REGIONS.get(provider, [])
        if region not in supported:
            violations.append((f"{provider} does not support {region}",
                              ["provider", "region"]))

    # C2: Encryption → multi-AZ + backup
    if instance.get("encryption") == "enabled":
        if instance.get("multi_az") is False:
            violations.append(("encryption requires multi-AZ",
                              ["encryption", "multi_az"]))
        if instance.get("backup_enabled") is False:
            violations.append(("encryption requires backup",
                              ["encryption", "backup_enabled"]))

    # C3: Production
    if instance.get("environment") == "production":
        if instance.get("backup_enabled") is False:
            violations.append(("production requires backup",
                              ["environment", "backup_enabled"]))
        ret = instance.get("backup_retention_days")
        if ret is not None and ret < 30:
            violations.append(("production requires retention >= 30",
                              ["environment", "backup_retention_days"]))
        if instance.get("monitoring") == "disabled":
            violations.append(("production requires monitoring",
                              ["environment", "monitoring"]))
        if instance.get("logging_level") == "debug":
            violations.append(("production cannot use debug logging",
                              ["environment", "logging_level"]))

    # C4: Database
    if instance.get("service_type") == "database":
        if instance.get("multi_az") is False:
            violations.append(("database requires multi-AZ",
                              ["service_type", "multi_az"]))
        if instance.get("backup_enabled") is False:
            violations.append(("database requires backup",
                              ["service_type", "backup_enabled"]))
        st = instance.get("storage_gb")
        if st is not None and st < 100:
            violations.append(("database requires storage >= 100",
                              ["service_type", "storage_gb"]))

    # C5: Compliance
    comp = instance.get("compliance")
    if comp in ("hipaa", "pci_dss"):
        if instance.get("encryption") != "enabled":
            violations.append((f"{comp} requires encryption",
                              ["compliance", "encryption"]))
        if instance.get("backup_enabled") is False:
            violations.append((f"{comp} requires backup",
                              ["compliance", "backup_enabled"]))
        if instance.get("monitoring") == "disabled":
            violations.append((f"{comp} requires monitoring",
                              ["compliance", "monitoring"]))
        if instance.get("logging_level") == "debug":
            violations.append((f"{comp} cannot use debug logging",
                              ["compliance", "logging_level"]))

    # C6: Free tier
    if instance.get("cost_tier") == "free":
        sz = instance.get("instance_size")
        if sz not in ("small", "medium", None):
            violations.append(("free tier requires small/medium instance",
                              ["cost_tier", "instance_size"]))
        if instance.get("auto_scaling") is True:
            violations.append(("free tier cannot auto-scale",
                              ["cost_tier", "auto_scaling"]))
        mx = instance.get("max_instances")
        if mx is not None and mx > 2:
            violations.append(("free tier max_instances <= 2",
                              ["cost_tier", "max_instances"]))

    # C7: Large storage
    st = instance.get("storage_gb")
    if st is not None and st > 1000:
        sz = instance.get("instance_size")
        if sz not in ("large", "xlarge", None):
            violations.append(("storage > 1000GB requires large/xlarge",
                              ["storage_gb", "instance_size"]))
        mem = instance.get("memory_gb")
        if mem is not None and mem < 16:
            violations.append(("storage > 1000GB requires memory >= 16",
                              ["storage_gb", "memory_gb"]))

    # C8: Serverless
    if instance.get("service_type") == "serverless":
        if instance.get("auto_scaling") is False:
            violations.append(("serverless requires auto-scaling",
                              ["service_type", "auto_scaling"]))
        mx = instance.get("max_instances")
        if mx is not None and mx < 5:
            violations.append(("serverless requires max_instances >= 5",
                              ["service_type", "max_instances"]))
        sz = instance.get("instance_size")
        if sz not in ("small", None):
            violations.append(("serverless requires small instance",
                              ["service_type", "instance_size"]))

    return violations


def check_constraints_v3(instance: dict) -> list[tuple[str, list[str]]]:
    """All JSON-v3 constraint checks: v2 base + 7 coupled constraints."""

    violations = check_constraints_v2(instance)

    # --- C9: Free tier + compliance conflict ---
    # Free tier (C6) limits monitoring and backup, but hipaa/pci_dss (C5)
    # requires monitoring+backup. These are incompatible by design.
    comp = instance.get("compliance")
    cost = instance.get("cost_tier")
    if cost == "free" and comp in ("hipaa", "pci_dss"):
        violations.append((
            "free tier incompatible with compliance (conflicting requirements)",
            ["cost_tier", "compliance"],
        ))

    # --- C10: Database + free tier conflict ---
    # Database (C4) requires storage >= 100, multi-AZ, backup.
    # Free tier (C6) limits instance size and auto-scaling.
    # If both active, storage constraint may still be satisfiable but
    # multi-AZ is a premium feature incompatible with free.
    if instance.get("service_type") == "database" and cost == "free":
        violations.append((
            "database incompatible with free tier (multi-AZ is premium)",
            ["service_type", "cost_tier"],
        ))

    # --- C11: Production + encryption requires 90-day retention ---
    env = instance.get("environment")
    if env == "production" and instance.get("encryption") == "enabled":
        ret = instance.get("backup_retention_days")
        if ret is not None and ret < 90:
            violations.append((
                "production + encryption requires retention >= 90 days",
                ["environment", "encryption", "backup_retention_days"],
            ))

    # --- C12: Multi-AZ + free tier conflict ---
    # Multi-AZ is a premium feature, incompatible with free tier.
    if instance.get("multi_az") is True and cost == "free":
        violations.append((
            "multi-AZ incompatible with free tier",
            ["multi_az", "cost_tier"],
        ))

    # --- C13: Serverless + compliance conflict ---
    # Serverless (C8) requires auto-scaling and small instance.
    # Compliance (C5) requires encryption and specific logging.
    # If both are active, the main conflict is that serverless
    # environments typically cannot meet HIPAA logging requirements.
    if instance.get("service_type") == "serverless" and comp in ("hipaa", "pci_dss"):
        violations.append((
            "serverless incompatible with compliance (logging requirements)",
            ["service_type", "compliance"],
        ))

    # --- C14: Very large storage requires enterprise cost tier ---
    st = instance.get("storage_gb")
    if st is not None and st > 2000:
        if cost not in ("enterprise", "premium", None):
            violations.append((
                "storage > 2000GB requires enterprise or premium tier",
                ["storage_gb", "cost_tier"],
            ))

    # --- C15: Max instances > 20 requires enterprise ---
    mx = instance.get("max_instances")
    if mx is not None and mx > 20 and cost != "enterprise":
        violations.append((
            "max_instances > 20 requires enterprise tier",
            ["max_instances", "cost_tier"],
        ))

    return violations


# ---------------------------------------------------------------------------
# Domain class
# ---------------------------------------------------------------------------

class JsonSchemaV3Domain(FiniteReasoningDomain):
    """Cloud infra config with 15 constraint groups, designed to be hard.

    Unlike v2, constraints are tightly coupled: fixing one can break
    another. Cost_tier and compliance are shared across many constraint
    groups, creating situations where no single-step repair works.
    """

    def num_variables(self) -> int:
        return N_FIELDS

    def domain_size(self, i: int) -> int:
        return len(FIELD_DOMAINS[FIELD_NAMES[i]])

    def max_domain_size(self) -> int:
        return MAX_DOMAIN

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a valid config, handling contradictions by ordering."""
        for attempt in range(200):
            provider = rng.choice(FIELD_DOMAINS["provider"])
            region = rng.choice(PROVIDER_REGIONS[provider])

            service_type = rng.choice(FIELD_DOMAINS["service_type"])
            environment = rng.choice(FIELD_DOMAINS["environment"])
            cost_tier = rng.choice(FIELD_DOMAINS["cost_tier"])
            compliance = rng.choice(FIELD_DOMAINS["compliance"])

            # Resolve C9: free + compliance conflict
            if cost_tier == "free" and compliance in ("hipaa", "pci_dss"):
                # Flip one to resolve
                if rng.random() < 0.5:
                    cost_tier = "standard"
                else:
                    compliance = "none"

            # Resolve C10: database + free
            if service_type == "database" and cost_tier == "free":
                cost_tier = "standard"

            # Resolve C12: multi-AZ + free
            multi_az = bool(rng.choice([True, False]))
            if cost_tier == "free":
                multi_az = False

            # C13: serverless + compliance
            if service_type == "serverless" and compliance in ("hipaa", "pci_dss"):
                if rng.random() < 0.5:
                    service_type = "compute"
                else:
                    compliance = "none"

            # Encryption
            encryption = rng.choice(FIELD_DOMAINS["encryption"])
            backup_enabled = bool(rng.choice([True, False]))

            # C2: Encryption → multi-az + backup
            if encryption == "enabled":
                multi_az = True
                backup_enabled = True

            # C4: Database → multi-az + backup
            if service_type == "database":
                multi_az = True
                backup_enabled = True

            # C5: Compliance → encryption + backup
            if compliance in ("hipaa", "pci_dss"):
                encryption = "enabled"
                backup_enabled = True

            # C15: Enterprise needed for large max_instances
            max_instances = int(rng.choice(FIELD_DOMAINS["max_instances"]))
            if max_instances > 20:
                cost_tier = "enterprise"

            # C6: Free tier limits
            instance_size = rng.choice(FIELD_DOMAINS["instance_size"])
            auto_scaling = bool(rng.choice([True, False]))
            if cost_tier == "free":
                instance_size = rng.choice(["small", "medium"])
                auto_scaling = False
                max_instances = min(max_instances, 2)

            # C8: Serverless
            if service_type == "serverless":
                auto_scaling = True
                max_instances = max(max_instances, 5)
                instance_size = "small"

            # C14: Large storage → enterprise
            storage_gb = int(rng.choice(FIELD_DOMAINS["storage_gb"]))
            if storage_gb > 2000:
                if cost_tier not in ("enterprise", "premium"):
                    cost_tier = rng.choice(["enterprise", "premium"])

            # C7: Large storage → instance size
            if storage_gb > 1000:
                instance_size = rng.choice(["large", "xlarge"])
                memory_gb = int(rng.choice([m for m in FIELD_DOMAINS["memory_gb"] if m >= 16]))
            else:
                memory_gb = int(rng.choice(FIELD_DOMAINS["memory_gb"]))

            # C4: Database min storage
            if service_type == "database":
                storage_gb = max(storage_gb, 100)

            # C3: Production requirements
            backup_retention_days = int(rng.choice(FIELD_DOMAINS["backup_retention_days"]))
            logging_level = rng.choice(FIELD_DOMAINS["logging_level"])
            monitoring = rng.choice(FIELD_DOMAINS["monitoring"])

            if environment == "production":
                backup_enabled = True
                backup_retention_days = max(backup_retention_days, 30)
                if monitoring == "disabled":
                    monitoring = "basic"
                if logging_level == "debug":
                    logging_level = "info"

            # C11: Production + encryption → 90-day retention
            if environment == "production" and encryption == "enabled":
                backup_retention_days = max(backup_retention_days, 90)

            # C5 cont: Compliance logging
            if compliance in ("hipaa", "pci_dss"):
                if monitoring == "disabled":
                    monitoring = "basic"
                if logging_level == "debug":
                    logging_level = "info"

            cpu_cores = int(rng.choice(FIELD_DOMAINS["cpu_cores"]))

            instance = {
                "provider": provider, "region": region,
                "service_type": service_type,
                "instance_size": instance_size,
                "cpu_cores": cpu_cores, "memory_gb": memory_gb,
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

            violations = check_constraints_v3(instance)
            if len(violations) == 0:
                return encode_instance(instance)

        raise RuntimeError("Could not generate valid instance after 200 attempts")

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Check all 15 constraint groups with local residuals."""
        n = self.num_variables()
        if x.shape != (n,):
            raise ValueError(f"Expected shape ({n},), got {x.shape}")

        instance = decode_array(x)
        violations = check_constraints_v3(instance)

        field_counts = {}
        for desc, fields in violations:
            for f in fields:
                idx = FIELD_TO_IDX[f]
                field_counts[idx] = field_counts.get(idx, 0) + 1

        local_residuals = np.zeros(n, dtype=np.int64)
        for idx, count in field_counts.items():
            local_residuals[idx] = count

        return VerifierDiagnostics(
            global_violation=len(violations),
            local_residuals=local_residuals,
        )

    def build_factors(self) -> list[Factor]:
        return []

    def enumerate_solutions(self) -> np.ndarray:
        raise NotImplementedError("JSON-v3 too large for enumeration")
