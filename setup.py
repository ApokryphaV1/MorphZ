from setuptools import setup, find_packages
from pathlib import Path
from typing import List
import re


def read_requirements(path: str = "requirements.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def read_version(init_path: str = "src/morphZ/__init__.py") -> str:
    # Deprecated: version now provided by setuptools_scm from git tags
    return "0.0.0"


def read_readme(path: str = "README.md") -> tuple[str, str | None]:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8"), "text/markdown"
    return "KDE-based density estimation and approximation package.", None


setup(
    name="morphZ",
    use_scm_version=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
