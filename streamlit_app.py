"""
Root Streamlit Entrypoint
Allows running directly from project root:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import main

if __name__ == "__main__":
    main()
