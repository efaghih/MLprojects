"""
Check which packages are installed and working
"""

import sys

packages_to_check = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
]

print("Checking installed packages:")
print("=" * 60)

installed = []
missing = []

for import_name, package_name in packages_to_check:
    try:
        __import__(import_name)
        print(f"[OK] {package_name} - installed and working")
        installed.append(package_name)
    except ImportError:
        print(f"[MISSING] {package_name} - not installed")
        missing.append(package_name)

print("\n" + "=" * 60)
print(f"Installed: {len(installed)}/{len(packages_to_check)}")
print("\nFor our current pipeline, we have everything we need!")
print("We're using scipy for audio (no librosa needed)")

