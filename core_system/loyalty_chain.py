from typing import Any

from core_system.api import SutazAiAPI


class SutazAiLoyaltyChain:
    def verify(self, decision):
        """Check loyalty across sutazai realities"""
        realities = SutazAiAPI.list_realities()
        return all(self._check_reality_loyalty(r, decision) for r in realities)

    def _check_reality_loyalty(self, reality: str, decision: Any) -> bool:
        """Validate loyalty in a specific reality (implementation)."""
        return SutazAiAPI.validate_reality(reality, decision)
