"""
Bug-Detection-System Package Init
"""

import os
import sys

# Ensure torch DLLs on Windows are discoverable before any subsystem imports torch or sentence-transformers
torch_lib = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
    if torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED
)

__version__ = "1.0.0"
