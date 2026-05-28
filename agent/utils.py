import sys
from pathlib import Path

def get_resource_path(relative_path: str | Path) -> Path:
    """Get the absolute path to a resource, supporting PyInstaller.

    When running as a PyInstaller bundle, resources are unpacked into
    a temporary directory stored in sys._MEIPASS.
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return Path(sys._MEIPASS) / relative_path
    return Path(relative_path)
