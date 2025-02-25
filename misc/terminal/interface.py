"""
Terminal Interface Module for SutazAI

Provides core terminal interaction capabilities.
"""


class SutazAiTerminal:
    """
    Primary terminal interface for SutazAI system.

    Manages user interactions, command processing,
    and system communication through terminal.
    """

    def __init__(self, config=None):
        """
        Initialize terminal interface.

        Args:
            config (dict, optional): Configuration settings
        """
        self.config = config or {}

    def display(self, message: str):
        """
        Display a message in the terminal.

        Args:
            message (str): Message to display
        """
        print(message)

    def get_input(self, prompt: str) -> str:
        """
        Get user input from terminal.

        Args:
            prompt (str): Input prompt

        Returns:
            str: User input
        """
        return input(prompt)
