# Copyright 2025 Intrinsic Innovation LLC

"""
Test configuration file for pytest.

- Ensures the project root is added to `sys.path` so local imports work
  during testing (e.g., importing internal modules from outside the test folder).
- This setup allows seamless running of pytest from any directory.
"""

import atexit
import sys
from pathlib import Path
import pytest
import random
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Sets the random seed for numpy and random libraries before each test.
    The `autouse=True` flag ensures this fixture is used by every test
    automatically, without needing to be explicitly requested.
    """
    # A fixed seed for reproducibility
    seed = 42  
    
    random.seed(seed)
    np.random.seed(seed)
