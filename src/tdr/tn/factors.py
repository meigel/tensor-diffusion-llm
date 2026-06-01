"""
Factor utilities for tensor-network inference.

Provides operations on Factor objects:
  - condition_factor: slice a factor table at observed variable values
  - join_factors:     multiply factors via einsum, aligning by shared variables
  - contract_marginal: compute single-variable marginal by variable elimination
  - contract_all_marginals: compute all single-variable marginals

These form the building blocks for the ContractionMarginalBackend and
future tnreason integration.

Variable elimination
--------------------
Given factors {ψ_a(x_{∂a})} over a set of variables, compute:

    q_i(v) = Σ_{x_{V\{i}}} Π_a ψ_a(x_{∂a}) / Z

where Z = Σ_x Π_a ψ_a(x_{∂a}) is the partition function.

The elimination proceeds in a greedy variable order (all variables
except the target), at each step joining all factors involving the
eliminated variable and summing it out. This yields the exact
marginal for small factor graphs.
"""

from typing import Optional

import numpy as np

from tdr.domains.base import Factor
from tdr import MASK


def condition_factor(factor: Factor, observations: dict[int, int]) -> Factor:
    """Condition a factor on observed variable values.

    Slices the factor table at the observed values for any variables
    that appear in both the factor and the observations dict.

    Example:
        factor = Factor((3, 7, 11), table_4x4x4)
        conditioned = condition_factor(factor, {3: 1, 11: 2})
        # Returns Factor((7,), table_4) where table_4[v] = ψ(1, v, 2)

    Args:
        factor: The factor to condition.
        observations: Dict mapping variable index → observed value.

    Returns:
        A new Factor over the subset of variables NOT in observations.
    """
    # Build a slicing tuple: either an integer (observed) or slice(None) (free)
    slicing = []
    remaining_vars = []
    for var in factor.variables:
        if var in observations:
            slicing.append(observations[var])
        else:
            slicing.append(slice(None))
            remaining_vars.append(var)

    sliced_table = factor.table[tuple(slicing)]

    # If all variables were observed, the result is scalar — reshape to 1-D
    # with a single element so downstream operations work
    if not remaining_vars:
        return Factor(variables=(), table=np.asarray(sliced_table, dtype=np.float64))

    return Factor(variables=tuple(remaining_vars), table=np.ascontiguousarray(sliced_table))


def condition_all_factors(
    factors: list[Factor],
    x_masked: np.ndarray,
) -> list[Factor]:
    """Condition all factors on a masked state array.

    Observed entries (x_masked[i] != MASK) are used to slice factor tables.
    Masked entries are left free.

    Args:
        factors: List of Factors making up the tensor network.
        x_masked: State array, shape (n,); entries MASK or domain values.

    Returns:
        List of conditioned Factor objects (observed variables removed).
    """
    observations = {}
    for i, val in enumerate(x_masked):
        if val != MASK:
            observations[i] = int(val)

    return [condition_factor(f, observations) for f in factors]


def join_factors(factors: list[Factor]) -> Factor:
    """Join multiple factors into a single factor via einsum.

    Multiplies factor tables elementwise, aligning by shared variable
    indices. Equivalent to building the product over all factors:

        Ψ(x) = Π_a ψ_a(x_{∂a})

    Uses numpy einsum for efficient contraction.

    Args:
        factors: List of Factor objects to multiply.

    Returns:
        A single Factor over the union of all variable indices.

    Raises:
        ValueError: If factors list is empty.
    """
    if not factors:
        raise ValueError("Cannot join empty list of factors")

    if len(factors) == 1:
        return factors[0]

    # Collect all unique variables in order of first appearance
    all_vars: list[int] = []
    seen: set[int] = set()
    for f in factors:
        for v in f.variables:
            if v not in seen:
                seen.add(v)
                all_vars.append(v)

    # Build einsum equation: each factor's variables get a lowercase letter
    # Use 'a' through 'z' (enough for 26 variables — fine for small domains)
    import string
    letters = string.ascii_lowercase
    if len(all_vars) > len(letters):
        raise ValueError(f"Too many variables ({len(all_vars)}) for einsum letters")

    var_to_letter = {v: letters[i] for i, v in enumerate(all_vars)}

    subscripts: list[str] = []
    operands: list[np.ndarray] = []
    for f in factors:
        if not f.variables:
            # Scalar factor — just multiply in
            subscripts.append("")
            operands.append(f.table.ravel()[0] * np.ones(1))
        else:
            subscript = "".join(var_to_letter[v] for v in f.variables)
            subscripts.append(subscript)
            operands.append(f.table)

    result_subscript = "".join(var_to_letter[v] for v in all_vars)
    equation = ",".join(subscripts) + "->" + result_subscript

    joined = np.einsum(equation, *operands, dtype=np.float64)
    return Factor(variables=tuple(all_vars), table=joined)


def _eliminate_one_var(
    factors: list[Factor],
    var: int,
) -> list[Factor]:
    """Eliminate one variable by joining all factors involving it, then summing it out.

    This is one step of variable elimination.

    Args:
        factors: Current list of Factors.
        var: Variable index to eliminate.

    Returns:
        Updated list of Factors with var eliminated.
    """
    # Find all factors involving var
    involved = [f for f in factors if var in f.variables]

    if not involved:
        return factors  # var not in any factor — nothing to do

    remaining = [f for f in factors if var not in f.variables]

    # Join all involved factors
    if len(involved) == 1:
        joint = involved[0]
    else:
        joint = join_factors(involved)

    # Sum out the variable
    idx = joint.variables.index(var)
    new_table = joint.table.sum(axis=idx)
    new_vars = tuple(v for v in joint.variables if v != var)

    if len(new_vars) > 0:
        remaining.append(Factor(variables=new_vars, table=np.ascontiguousarray(new_table)))
    # If len(new_vars) == 0, the result is a scalar — we discard it
    # because it contributes to the partition function Z uniformly
    # (same factor for all values of remaining variables)

    return remaining


def _infer_domain_size(factors: list[Factor]) -> int:
    """Infer the maximum domain size from factor tables.

    Looks at all factor table shapes to determine the domain size.
    """
    max_d = 0
    for f in factors:
        for s in f.table.shape:
            max_d = max(max_d, s)
    return max_d


def contract_marginal(
    factors: list[Factor],
    target_var: int,
    domain_size: Optional[int] = None,
):
    """Compute the marginal distribution over a single variable.

    Uses variable elimination with a natural elimination order
    (all variables except target_var).

    Args:
        factors: List of Factor objects (already conditioned on observations).
        target_var: Variable index to compute marginal for.
        domain_size: Max domain size. Inferred from factor shapes if not given.

    Returns:
        q:      Marginal distribution array of shape (d,),
                or None if the state is a contradiction.
        logZ:   Log partition function (natural log).
        status: 'ok' or 'contradiction'.
    """
    # Check if target_var appears in any factor
    appears = any(target_var in f.variables for f in factors)
    if not appears:
        d = domain_size or _infer_domain_size(factors)
        q = np.full(d, 1.0 / d, dtype=np.float64)
        return q, 0.0, "ok"

    # Collect all unique variables
    all_vars = sorted(set(v for f in factors for v in f.variables))

    # Elimination order: everything except target_var
    elim_order = [v for v in all_vars if v != target_var]

    remaining = list(factors)

    for var in elim_order:
        remaining = _eliminate_one_var(remaining, var)

    # After elimination, remaining factors should all involve target_var
    # Join them and normalize
    if not remaining:
        d = domain_size or _infer_domain_size(factors)
        q = np.full(d, 1.0 / d, dtype=np.float64)
        return q, 0.0, "ok"

    joint = join_factors(remaining)

    # If joint involves multiple variables, marginalize down to target_var
    # (this can happen with non-optimal elimination orders)
    while len(joint.variables) > 1:
        other_vars = [v for v in joint.variables if v != target_var]
        if not other_vars:
            break
        v = other_vars[0]
        idx = joint.variables.index(v)
        joint.table = joint.table.sum(axis=idx)
        joint.variables = tuple(vv for vv in joint.variables if vv != v)

    # joint should now be a 1-D array over target_var
    if len(joint.variables) == 1 and joint.variables[0] == target_var:
        marginal = joint.table
    else:
        # Scalar — shouldn't happen but handle gracefully
        d = domain_size or _infer_domain_size(factors)
        return np.full(d, 1.0 / d, dtype=np.float64), 0.0, "ok"

    Z = np.sum(marginal)
    if Z == 0:
        d = domain_size or _infer_domain_size(factors)
        return None, -np.inf, "contradiction"

    return marginal / Z, np.log(Z), "ok"


def contract_all_marginals(
    factors: list[Factor],
    n: int,
    domain_size: Optional[int] = None,
):
    """Compute marginals for all variables 0..n-1.

    Args:
        factors: Conditioned factor list.
        n:       Total number of variables (some may not appear in factors).
        domain_size: Max domain size. Inferred from factor shapes if not given.

    Returns:
        q:      Array of shape (n, d) with marginal probabilities.
        logZ:   Log partition function (from first computed marginal).
        status: 'ok' or 'contradiction'.
    """
    d = domain_size or _infer_domain_size(factors)

    # Check for contradiction by computing first marginal
    first_var = next((i for i in range(n) if any(i in f.variables for f in factors)), 0)
    q0, logZ, status = contract_marginal(factors, first_var, d)

    if status == "contradiction":
        q = np.full((n, d), -1.0, dtype=np.float64)
        return q, -np.inf, "contradiction"

    q_arr = np.zeros((n, d), dtype=np.float64)
    q_arr[first_var] = q0

    for i in range(n):
        if i == first_var:
            continue
        qi, _, _ = contract_marginal(factors, i, d)
        if qi is not None:
            q_arr[i] = qi
        else:
            q_arr[i] = np.full(d, 1.0 / d, dtype=np.float64)

    # Check if all variables observed (no unobserved vars in factors)
    all_vars = set()
    for f in factors:
        all_vars.update(f.variables)
    if not all_vars:
        status = "all_solved"

    return q_arr, logZ, status
