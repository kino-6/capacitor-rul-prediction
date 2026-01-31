"""
Pytest configuration for true_rul tests
"""

import sys
from pathlib import Path

# Add parent project root to path for nasa_pcoe_eda imports
# This must happen at module level before any test imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
