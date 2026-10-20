#!/bin/sh
# Builds bitpattern/bdd.abi3.so from src/ in place, for development.
set -e
cd "$(dirname "$0")"
MOJO=${MOJO:-/home/stef/bitpattern/.venv/bin/mojo}
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)
"$MOJO" build src/lib.mojo -I src -D VERSION="$VERSION" --emit shared-lib -o bitpattern/bdd.abi3.so "$@"
