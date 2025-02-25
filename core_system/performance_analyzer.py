import json
import pstats


class PerformanceAnalyzer:
    """
    Advanced performance analysis tool for interpreting profiling results.
    """

    def __init__(self, profile_file: str = "profile_results.prof"):
        """
        Initialize the performance analyzer.

        Args:
            profile_file (str): Path to the profiling results file
        """
        self.profile_file = profile_file
        self.stats = None
        self.analysis_results = {
            "total_time": 0,
            "top_time_consuming_functions": [],
            "performance_recommendations": [],
        }

    def load_profile(self):
        """
        Load the profiling statistics.
        """
        try:
            self.stats = pstats.Stats(self.profile_file)
            self.stats.sort_stats("cumulative")
        except Exception as e:
            print(f"Error loading profile: {e}")
            return False
        return True

    def analyze_performance(self, top_n: int = 10):
        """
        Perform comprehensive performance analysis.

        Args:
            top_n (int): Number of top time-consuming functions to analyze

        Returns:
            Dict containing performance analysis results
        """
        if not self.stats:
            if not self.load_profile():
                return self.analysis_results

        # Capture top time-consuming functions
        top_functions = []
        for entry in self.stats.stats.values():
            func_name = f"{entry[4][0]}:{entry[4][1]} ({entry[4][2]})"
            total_time = entry[2]
            cumulative_time = entry[3]

            top_functions.append(
                {
                    "function": func_name,
                    "total_time": total_time,
                    "cumulative_time": cumulative_time,
                }
            )

        # Sort and truncate to top_n
        top_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
        self.analysis_results["top_time_consuming_functions"] = top_functions[:top_n]

        # Calculate total time
        self.analysis_results["total_time"] = sum(
            func["cumulative_time"] for func in top_functions
        )

        # Generate performance recommendations
        self._generate_recommendations()

        return self.analysis_results

    def _generate_recommendations(self):
        """
        Generate performance improvement recommendations.
        """
        recommendations = []

        for func in self.analysis_results["top_time_consuming_functions"]:
            if (
                func["cumulative_time"] > 0.5
            ):  # Threshold for significant time consumption
                recommendations.append(
                    {
                        "function": func["function"],
                        "time_consumed": func["cumulative_time"],
                        "recommendation": self._get_optimization_suggestion(
                            func["function"]
                        ),
                    }
                )

        self.analysis_results["performance_recommendations"] = recommendations

    def _get_optimization_suggestion(self, function_name: str) -> str:
        """
        Provide specific optimization suggestions based on function name.

        Args:
            function_name (str): Name of the function to provide suggestions for

        Returns:
            str: Optimization suggestion
        """
        suggestions = {
            "verify_system": "Consider optimizing system verification by parallelizing checks or reducing redundant operations.",
            "optimize_system": "Review system optimization logic for potential performance bottlenecks.",
            "validate_system": "Implement more efficient validation techniques, potentially using caching or incremental validation.",
            "database": "Optimize database queries, add indexing, and consider query caching.",
            "network": "Review network-related operations for potential optimization, such as connection pooling.",
        }

        # Find the most relevant suggestion
        for key, suggestion in suggestions.items():
            if key in function_name.lower():
                return suggestion

        return "Consider refactoring the function for improved performance."

    def save_analysis(self, output_file: str = "performance_analysis.json"):
        """
        Save performance analysis results to a JSON file.

        Args:
            output_file (str): Path to save the analysis results
        """
        try:
            with open(output_file, "w") as f:
                json.dump(self.analysis_results, f, indent=2)
            print(f"Performance analysis saved to {output_file}")
        except Exception as e:
            print(f"Error saving performance analysis: {e}")

    def print_analysis(self):
        """
        Print a human-readable performance analysis report.
        """
        print("\n🚀 SutazAI Performance Analysis Report 🚀")
        print(
            f"Total Execution Time: {self.analysis_results['total_time']:.4f} seconds"
        )

        print("\n📊 Top Time-Consuming Functions:")
        for i, func in enumerate(
            self.analysis_results["top_time_consuming_functions"], 1
        ):
            print(f"{i}. {func['function']}")
            print(f"   Total Time: {func['cumulative_time']:.4f} seconds")

        print("\n🔧 Performance Recommendations:")
        for rec in self.analysis_results["performance_recommendations"]:
            print(f"- {rec['function']}")
            print(f"  Time Consumed: {rec['time_consumed']:.4f} seconds")
            print(f"  Suggestion: {rec['recommendation']}\n")


def main():
    """
    Main function to run performance analysis.
    """
    analyzer = PerformanceAnalyzer()
    analyzer.analyze_performance()
    analyzer.save_analysis()
    analyzer.print_analysis()


if __name__ == "__main__":
    main()
