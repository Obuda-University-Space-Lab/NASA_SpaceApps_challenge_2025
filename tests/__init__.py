"""
Test configuration and utilities
"""

import pytest
import sys
from pathlib import Path

# Configure test environment
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add src directory to Python path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))