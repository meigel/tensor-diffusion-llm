"""
Verifier: computes global violation and local residuals for a domain.

Re-exports VerifierDiagnostics for convenience.
"""

from tdr.domains.base import VerifierDiagnostics, FiniteReasoningDomain

# Re-export for convenience.
__all__ = ["VerifierDiagnostics", "FiniteReasoningDomain"]
