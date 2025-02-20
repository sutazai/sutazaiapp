"""
Communication Agent Module

Handles notification and communication mechanisms.
"""


class NotificationAPI:
    """
    Manages system notifications and communication channels.
    """

    def __init__(self):
        """
        Initialize notification system.
        """
        self.notifications = []

    def send_notification(self, message: str, channel: str = "default"):
        """
        Send a notification through specified channel.

        Args:
            message (str): Notification content
            channel (str, optional): Notification channel
        """
        print(f"[{channel.upper()}] {message}")
        self.notifications.append({"message": message, "channel": channel})
