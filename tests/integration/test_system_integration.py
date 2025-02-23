#!/usr/bin/env python3
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def test_dummy():
    """Dummy test to ensure test suite is working"""
    assert True

def main():
    pytest.main(["-v", __file__])

if __name__ == "__main__":
    main()
