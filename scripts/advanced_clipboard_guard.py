#!/usr/bin/env python3.11
"""Advanced clipboard monitoring and sanitization module for SutazAI."""

import logging
import os
import re
from re import Pattern
from typing import Optional, list
from typing import List


logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ClipboardGuard:
    """Monitors and sanitizes clipboard content for security."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the clipboard guard.

        Args:
        config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.patterns: List[Pattern] = []
        self.replacements: List[str] = []

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            else:
            self._load_default_patterns()

            def _load_config(self, config_path: str) -> None:
                """Load patterns from configuration file.

                Args:
                config_path: Path to configuration file
                """
                try:
                    import yaml

                    with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                    for pattern in config.get("patterns", []):
                        self.add_pattern(
                        pattern["regex"],
                        pattern.get("replacement", "[REDACTED]"),
                        )

                        except Exception as e:
                            self.logger.error(
                                "Failed to load config: %s",
                                str(e))
                            self._load_default_patterns()

                            def _load_default_patterns(self) -> None:
                                """Load default security patterns."""
                                # Add default patterns
                                self.add_pattern(
                                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                                "[EMAIL]",
                                )
                                self.add_pattern(
                                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                                "[PHONE]",
                                )
                                self.add_pattern(
                                r"\b\d{16}\b",
                                "[CARD_NUMBER]",
                                )

                                def add_pattern(
                                    self,
                                    pattern: str,
                                    replacement: str) -> None:
                                    """Add a new pattern for content sanitization.

                                    Args:
                                    pattern: Regular expression pattern
                                    replacement: Replacement text
                                    """
                                    try:
                                        compiled = re.compile(pattern)
                                        self.patterns.append(compiled)
                                        self.replacements.append(replacement)
                                        self.logger.debug(
                                            "Added pattern: %s",
                                            pattern)

                                        except Exception as e:
                                            self.logger.error(
                                                "Failed to add pattern: %s",
                                                str(e))

                                            def sanitize_content(
                                                self,
                                                content: str) -> str:
                                                """Sanitize content by applying security patterns.

                                                Args:
                                                content: Content to sanitize

                                                Returns:
                                                str: Sanitized content
                                                """
                                                sanitized = content

                                                try:
                                                    for pattern, replacement in zip(
                                                        self.patterns,
                                                        self.replacements):
                                                        sanitized = pattern.sub(
                                                            replacement,
                                                            sanitized)

                                                    return sanitized

                                                    except Exception as e:
                                                        self.logger.error(
                                                            "Failed to sanitize content: %s",
                                                            str(e))
                                                    return content

                                                    def monitor_clipboard(
                                                        self) -> None:
                                                        """Start monitoring clipboard for sensitive content."""
                                                        try:
                                                            import pyperclip

                                                            last_content = pyperclip.paste()

                                                            while True:
                                                                try:
                                                                    current_content = pyperclip.paste()
                                                                    if current_content != last_content:
                                                                        sanitized = self.sanitize_content(
                                                                            current_content)
                                                                        if sanitized != current_content:
                                                                            pyperclip.copy(
                                                                                sanitized)
                                                                            self.logger.info(
                                                                                "Sanitized clipboard content")
                                                                            last_content = sanitized
                                                                            except Exception as e:
                                                                                self.logger.error(
                                                                                    "Clipboard error: %s",
                                                                                    str(e))

                                                                                except ImportError:
                                                                                    self.logger.error(
                                                                                    "pyperclip not installed. Run: pip install pyperclip",
                                                                                    )
                                                                                    except Exception as e:
                                                                                        self.logger.error(
                                                                                            "Monitor error: %s",
                                                                                            str(e))


                                                                                        def main():
                                                                                            """Main entry point for clipboard monitoring."""
                                                                                            guard = ClipboardGuard()

                                                                                            # Add custom patterns if needed
                                                                                            guard.add_pattern(
                                                                                            r"\b(
                                                                                                ?:password|pwd|pass)\s*[:=]\s*\S+\b",
                                                                                            "[PASSWORD]",
                                                                                            )

                                                                                            # Start monitoring
                                                                                            print(
                                                                                                "Starting clipboard monitor (Ctrl+C to stop)...")
                                                                                            guard.monitor_clipboard()


                                                                                            if __name__ == "__main__":
                                                                                                main()
