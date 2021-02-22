#!/usr/bin/env bash

# Setup
# bash scripts/setup.sh

# Linting
autoflake -r src --recursive --in-place --remove-all-unused-imports --exclude=__init__.py
flake8 src
isort -rc src
black src --line-length=120

# Test coverage
pytest --ignore data_generation --cov=src/typesetter --cov-report html ${@}

# Type anno coverage
mypy src

# Docstring coverage
interrogate -c pyproject.toml src