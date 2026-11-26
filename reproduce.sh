#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PHYEVO_PYTHON:-python}" -B "$release_dir/scripts/reproduce.py" "$@"
