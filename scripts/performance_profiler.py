#!/usr/bin/env python3
"""Performance profiling module for SutazAI."""

import cProfile
import logging
import os
import pstats
from typing import Any, Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class PerformanceProfiler:
    """Handles performance profiling for SutazAI components."""

    def __init__(self, output_dir: str = "profiles"):
        """Initialize the performance profiler.

        Args:
            output_dir: Directory to store profiling results
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def profile_function(
        self,
        func: Callable,
        *args,
        output_file: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Profile a function execution.

        Args:
            func: Function to profile
            output_file: Optional file to save profiling results
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Any: Result of the profiled function
        """
        if output_file is None:
            output_file = os.path.join(
                self.output_dir,
                f"{func.__name__}_{self.get_timestamp()}.prof",
            )

        # Create profiler
        profiler = cProfile.Profile()

        try:
            # Profile the function
            result = profiler.runcall(func, *args, **kwargs)

            # Save profiling stats
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumulative")
            stats.dump_stats(output_file)

            self.logger.info("Saved profiling results to: %s", output_file)
            return result

        except Exception as e:
            self.logger.error("Profiling failed: %s", str(e))
            raise

    def get_timestamp(self) -> str:
        """Get current timestamp string.

        Returns:
            str: Formatted timestamp string
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def analyze_profile(self, profile_file: str) -> None:
        """Analyze and print profiling results.

        Args:
            profile_file: Path to the profile results file
        """
        try:
            stats = pstats.Stats(profile_file)
            stats.sort_stats("cumulative")

            # Print top 20 time-consuming functions
            self.logger.info("Top 20 time-consuming functions:")
            stats.print_stats(20)

        except Exception as e:
            self.logger.error("Failed to analyze profile: %s", str(e))


def main():
    """Example usage of the performance profiler."""

    def example_function():
        """Example function to profile."""
        total = 0
        for i in range(1000000):
            total += i
        return total

    profiler = PerformanceProfiler()

    # Profile the example function
    result = profiler.profile_function(example_function)

    # Analyze the latest profile
    latest_profile = max(
        [f for f in os.listdir(profiler.output_dir) if f.endswith(".prof")],
        key=lambda x: os.path.getctime(os.path.join(profiler.output_dir, x)),
    )
    profiler.analyze_profile(os.path.join(profiler.output_dir, latest_profile))


if __name__ == "__main__":
    main()
