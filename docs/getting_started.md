# Getting Started

## Installation

### From PyPI (recommended)

```bash
python -m pip install morphZ
```

### From a local checkout

Use an editable install if you plan to modify the source:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Running Tests

Test the installation with `pytest`:

```bash
python -m pip install -e ".[test]"
pytest
```

## Building the Documentation

This project uses [Jupyter Book](https://jupyterbook.org) and keeps generated
artifacts out of version control.  To build the docs locally:

```bash
python -m pip install jupyter-book
./docs/build_docs.sh
```

The helper script copies the main `README.md` and the `examples/` directory into
`docs/_auto/` (which is Git-ignored) so that the book always renders the latest
content.  HTML output is written to `docs/_build/html`.
