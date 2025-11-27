#!/usr/bin/env python3
"""
Prepare documentation assets by copying the README and `examples/` tree.

Files are copied into docs/_auto/, and names with whitespace are normalised to
use underscores so that Jupyter Book paths stay stable.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
AUTO_DIR = DOCS_DIR / "_auto"
SOURCE_EXAMPLES = ROOT / "examples"
README_SRC = ROOT / "README.md"


def _sanitize_segment(segment: str) -> str:
    """Replace whitespace with underscores to keep doc paths tidy."""
    return re.sub(r"\s+", "_", segment)


def _copy_examples() -> None:
    """Copy the examples folder while normalising whitespace in names."""
    dest_root = AUTO_DIR / "examples"
    dest_root.mkdir(parents=True, exist_ok=True)

    for src in SOURCE_EXAMPLES.rglob("*"):
        rel_parts = [_sanitize_segment(part) for part in src.relative_to(SOURCE_EXAMPLES).parts]
        dest = dest_root.joinpath(*rel_parts)
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            if dest.suffix == ".ipynb":
                _ensure_notebook_title(dest, rel_parts[-1])


def _ensure_notebook_title(notebook_path: Path, filename: str) -> None:
    """Ensure each copied notebook has a first-level title."""
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    with notebook_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cells = data.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if source.strip().startswith("#"):
                break
    else:
        heading_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "id": f"generated-{uuid.uuid4().hex}",
            "source": [f"# {title}\n"],
        }
        cells.insert(0, heading_cell)
        data["cells"] = cells
        with notebook_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)
            fh.write("\n")


def main() -> None:
    if AUTO_DIR.exists():
        shutil.rmtree(AUTO_DIR)
    AUTO_DIR.mkdir(parents=True, exist_ok=True)

    if not README_SRC.exists():
        raise FileNotFoundError(f"Could not locate README at {README_SRC}")
    shutil.copy2(README_SRC, AUTO_DIR / "README.md")

    if SOURCE_EXAMPLES.exists():
        _copy_examples()


if __name__ == "__main__":
    main()
