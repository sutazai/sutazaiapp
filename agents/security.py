"""
FounderProtectionSystem - Implements security protocols to authorize new skill acquisition.
"""


class FounderProtectionSystem:
    def authorize_learning(self, skill):
        """
        Determine if learning a particular skill is authorized.

        Args:
            skill (dict): A dictionary expected to contain a 'name' and an 'impact' key.

        Returns:
            bool: True if skill acquisition is authorized; otherwise, False.
        """
        # Example logic: Only allow if the 'impact' meets or exceeds a threshold.
        return skill.get("impact", 0) >= 7.0
