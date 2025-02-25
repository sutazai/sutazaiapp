#!/usr/bin/env python3
# cSpell:ignore Sutaz SutazAI

"""
SutazAI Validator Module

This module provides a mechanism for registering and executing system-wide
validation checks. Each check is defined by a name, a validator function, and
a flag indicating whether the check is critical. The `run_checks` method
executes each registered check and yields its status.
"""


class SutazAIValidator:
    def __init__(self):
        """
        Initialize a new validator instance with an empty list of checks.
        """
        self.checks = []

    def add_check(self, name, validator, critical=True):
        """
        Register a new validation check.

        Args:
            name (str): A unique identifier for the check.
            validator (callable): A callable that performs the validation. It should return
                                  a truthy value if the check passes, and a falsy value if it fails.
            critical (bool, optional): Whether this check is critical for system functionality. Defaults to True.
        """
        self.checks.append(
            {"name": name, "validator": validator, "critical": critical}
        )

    def run_checks(self):
        """
        Execute all registered validation checks.

        Yields:
            dict: A dictionary for each check with the following keys:
                - component (str): The name of the check.
                - status (str): "PASS", "FAIL", or an error message if the check raised an exception.
                - critical (bool): Whether the check is critical.
        """
        for check in self.checks:
            try:
                result = check["validator"]()
                status = "PASS" if result else "FAIL"
            except Exception as e:
                status = f"ERROR: {str(e)}"

            yield {
                "component": check["name"],
                "status": status,
                "critical": check["critical"],
            }
