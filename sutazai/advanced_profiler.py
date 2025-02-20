import os
import sys
import time
import cProfile
import pstats
import io
import tracemalloc
import linecache
import traceback
from typing import Dict, List, Any, Optional, Callable

class AdvancedProfiler:
    """
    A comprehensive performance profiling tool for SutazAI.
    Provides in-depth performance analysis and memory tracking.
    """

    def __init__(self, output_dir: str = 'profiling_results'):
        """
        Initialize the advanced profiler.

        Args:
            output_dir (str): Directory to store profiling results
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def profile_function(
        self, 
        func: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Profile a function's performance and memory usage.

        Args:
            func (Callable): Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Dict[str, Any]: Detailed profiling results
        """
        # Memory profiling
        tracemalloc.start()
        start_memory = tracemalloc.take_snapshot()

        # Time and CPU profiling
        profiler = cProfile.Profile()
        start_time = time.time()
        
        try:
            result = profiler.runcall(func, *args, **kwargs)
        except Exception as e:
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        end_time = time.time()
        
        # Collect profiling stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative').print_stats(10)
        
        # Memory analysis
        end_memory = tracemalloc.take_snapshot()
        memory_stats = self._analyze_memory_usage(start_memory, end_memory)
        
        tracemalloc.stop()

        return {
            'function_name': func.__name__,
            'execution_time': end_time - start_time,
            'cpu_profile': stats_stream.getvalue(),
            'memory_stats': memory_stats,
            'result': result
        }

    def _analyze_memory_usage(
        self, 
        start_snapshot: tracemalloc.Snapshot, 
        end_snapshot: tracemalloc.Snapshot
    ) -> Dict[str, Any]:
        """
        Analyze memory usage between two snapshots.

        Args:
            start_snapshot (tracemalloc.Snapshot): Initial memory snapshot
            end_snapshot (tracemalloc.Snapshot): Final memory snapshot

        Returns:
            Dict[str, Any]: Memory usage analysis
        """
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        memory_analysis = {
            'total_memory_increase': 0,
            'top_memory_consumers': []
        }

        for stat in top_stats[:10]:
            frame = stat.traceback[0]
            filename = frame.filename
            lineno = frame.lineno
            line = linecache.getline(filename, lineno).strip()
            
            memory_consumer = {
                'filename': filename,
                'line_number': lineno,
                'line_content': line,
                'memory_increase': stat.size_diff
            }
            
            memory_analysis['total_memory_increase'] += stat.size_diff
            memory_analysis['top_memory_consumers'].append(memory_consumer)

        return memory_analysis

    def profile_project(
        self, 
        project_root: str, 
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Profile entire project performance and memory usage.

        Args:
            project_root (str): Root directory of the project
            exclude_patterns (Optional[List[str]]): Patterns to exclude from profiling

        Returns:
            Dict[str, Any]: Comprehensive project profiling results
        """
        exclude_patterns = exclude_patterns or [
            '.git', '__pycache__', 'venv', 'env', 'node_modules'
        ]

        project_profile = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'project_root': project_root,
            'file_profiles': []
        }

        for root, dirs, files in os.walk(project_root):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            file_profile = self.profile_function(f.read)
                            file_profile['file_path'] = file_path
                            project_profile['file_profiles'].append(file_profile)
                    except Exception as e:
                        project_profile['file_profiles'].append({
                            'file_path': file_path,
                            'error': str(e)
                        })

        # Generate report
        report_path = os.path.join(
            self.output_dir, 
            f'project_profile_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            import json
            json.dump(project_profile, f, indent=2)

        return project_profile

def main():
    profiler = AdvancedProfiler()
    
    # Example usage
    def test_function(n):
        return sum(i**2 for i in range(n))

    profile_result = profiler.profile_function(test_function, 10000)
    print("Function Profiling Result:", profile_result)

    # Uncomment to profile entire project
    # project_profile = profiler.profile_project('/path/to/project')
    # print("Project Profiling Result:", project_profile)

if __name__ == '__main__':
    main() 