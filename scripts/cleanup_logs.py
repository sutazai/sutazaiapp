#!/usr/bin/env python3.11
"""
Cleanup Logs Script for SutazAI

This script compresses log files in \
    the logs/ directory that are older than a specified
threshold (default 1 day) to reduce disk I/O and improve system performance.
Compressed files are stored with a .gz extension.
"""

import gzip
import os
import shutil
import time

# Directory containing logs
logs_dir = os.path.join(os.getcwd(), "logs")
# Threshold for file age in seconds (1 day = 86400 seconds)
threshold = 86400


def compress_old_logs():
    now = time.time()
    for root, _, files in os.walk(logs_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Skip if already compressed
            if filepath.endswith(".gz"):
            continue
            try:
                mtime = os.path.getmtime(filepath)
                except Exception as e:
                    print(f"Failed to get mtime for {filepath}: {e}")
                continue
                if now - mtime > threshold:
                    compressed_filepath = filepath + ".gz"
                    try:
                        with (
                        open(filepath, "rb") as f_in,
                        gzip.open(compressed_filepath, "wb") as f_out,
                        ):
                        shutil.copyfileobj(f_in, f_out)
                        os.remove(filepath)
                        print(f"Compressed and removed: {filepath}")
                        except Exception as e:
                            print(f"Error compressing {filepath}: {e}")


                            def main():
                                compress_old_logs()
                                print("Log cleanup completed.")


                                if __name__ == "__main__":
                                    main()
