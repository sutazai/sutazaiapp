#!/usr/bin/env python3.11
import functools
import os
import sys
from typing import Any, Callable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from otp_manager import OTPManager


class OTPValidator:
    def __init__(
        self,
        config_path="/opt/sutazaiapp/config/external_calls.toml"):
        """
        Initialize OTP validator with configuration

        Args:
        config_path (str): Path to external calls configuration
        """
        self.otp_manager = OTPManager()
        self.config_path = config_path
        self.external_calls_enabled = self._load_config()

        def _load_config(self) -> bool:
            """
            Load external calls configuration

            Returns:
            bool: Whether external calls are enabled
            """
            try:
                import toml

                config = toml.load(self.config_path)
            return config.get(
                "security",
                {}).get("external_calls_enabled",
                False)
            except Exception as e:
                print(f"Config load error: {e}")
            return False

            def validate_external_call(self, func: Callable) -> Callable:
                """
                Decorator to validate external calls with OTP

                Args:
                func (Callable): Function to be decorated

                Returns:
                Callable: Decorated function with OTP validation
                """

                @functools.wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    # If external calls disabled, block
                    if not self.external_calls_enabled:
                    raise PermissionError("External calls are disabled")

                    # Prompt for OTP
                    print("External call requires OTP validation.")
                    user_otp = input("Enter OTP: ")

                    # Validate OTP
                    last_otp_info = {}  # Replace with secure OTP retrieval
                    if not self.otp_manager.validate_otp(
                        user_otp,
                        last_otp_info):
                    raise ValueError("Invalid OTP. External call blocked.")

                return func(*args, **kwargs)

            return wrapper


            # Example usage
            def main():
                validator = OTPValidator()

                @validator.validate_external_call
                def example_external_call():
                    print("External call successful!")

                    try:
                        example_external_call()
                        except (PermissionError, ValueError) as e:
                            print(f"Call blocked: {e}")


                            if __name__ == "__main__":
                                main()
