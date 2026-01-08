"""
Install dependencies step by step
Purpose: Install packages one by one to identify any issues
"""

import subprocess
import sys

# Packages we need (in order of importance)
packages = [
    "numpy",
    "scipy", 
    "pandas",
    "scikit-learn",
    # "librosa",  # Commented out - may have issues with Python 3.13
    # "soundfile",  # Commented out - may have issues with Python 3.13
]

print("Installing packages one by one...")
print("=" * 60)

for package in packages:
    print(f"\nInstalling {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[OK] {package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}")
        print(f"Error: {e}")
        print("\nTrying with --no-build-isolation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-build-isolation"])
            print(f"[OK] {package} installed with --no-build-isolation")
        except:
            print(f"[SKIP] {package} - will need alternative solution")

print("\n" + "=" * 60)
print("Installation complete!")
print("\nNote: If librosa/soundfile fail, we can use scipy only (already working)")

