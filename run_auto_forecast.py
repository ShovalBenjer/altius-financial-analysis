#!/usr/bin/env python3
"""Run the auto-forecast agent."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.auto_forecast import main

if __name__ == "__main__":
    main()