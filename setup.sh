#!/bin/bash
set -e

echo "Setting up pysrf development environment"

if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found. Install pyenv first:"
    echo "  curl https://pyenv.run | bash"
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "poetry not found. Installing."
    curl -sSL https://install.python-poetry.org | python3 -
fi

PYTHON_VERSION="3.12.4"
echo "Checking Python ${PYTHON_VERSION}"
if ! pyenv versions | grep -q "${PYTHON_VERSION}"; then
    echo "Installing Python ${PYTHON_VERSION}"
    pyenv install ${PYTHON_VERSION}
fi

echo "Setting local Python version"
pyenv local ${PYTHON_VERSION}

echo "Installing dependencies"
poetry install --no-root --all-extras

echo "Building and installing pysrf (editable)"
if ! poetry run pip install -e . --no-build-isolation; then
    echo "Editable install failed. A C compiler is required to build the Cython extension." >&2
    exit 1
fi

echo "Running tests"
poetry run pytest tests/ -v

echo "Setup complete. Run commands with: poetry run <command>"
