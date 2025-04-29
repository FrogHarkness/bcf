import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["os", "numpy", "pandas", "tkinter", "gurobipy"],
    "includes": ["numpy", "pandas"],
    "include_files": [],
    "excludes": []
}

# GUI applications require a different base on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="PeerReviewer",
    version="1.0",
    description="Peer Reviewer Assignment Tool",
    options={"build_exe": build_exe_options},
    executables=[Executable("prer_.py", base=base, target_name="PeerReviewer")]
)

print("To build the executable, run: python build_exe.py build")
print("The executable will be in the build directory") 