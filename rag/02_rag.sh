#!/usr/bin/env bash
# Foreground wrapper: exit codes and logs remain visible to the caller.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python}" -m rag.02_rag "$@"
