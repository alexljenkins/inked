#!/usr/bin/env bash

set -e
set -x

bash scripts/test.sh --cov=src/typesetter --cov-report html ${@} # --cov=src/api