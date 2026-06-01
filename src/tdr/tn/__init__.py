"""
Tensor-network backends for marginal computation.
"""

from tdr.tn.brute_force_backend import BruteForceMarginalBackend
from tdr.tn.marginals import MarginalBackend, ContractionMarginalBackend
from tdr.tn.factors import (
    Factor,
    condition_factor,
    condition_all_factors,
    join_factors,
    contract_marginal,
    contract_all_marginals,
)

__all__ = [
    "BruteForceMarginalBackend",
    "MarginalBackend",
    "ContractionMarginalBackend",
    "Factor",
    "condition_factor",
    "condition_all_factors",
    "join_factors",
    "contract_marginal",
    "contract_all_marginals",
]
