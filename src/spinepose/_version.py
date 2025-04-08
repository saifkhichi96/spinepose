# spinepose/_version.py

import os

# This assumes that the VERSION file is in the parent directory
# relative to where this file lives. Adjust the path if needed.
VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION")

with open(VERSION_FILE, "r", encoding="utf-8") as f:
    __version__ = f.read().strip()
