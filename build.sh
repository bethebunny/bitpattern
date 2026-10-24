#!/bin/sh
# Builds bitpattern/bdd.abi3.so from src/ in place, for development.
set -e
cd "$(dirname "$0")"
MOJO=${MOJO:-.venv/bin/mojo}
VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml)
if [ "$(uname)" = Darwin ]; then
    # Python's symbols resolve when Python loads the module, as on Linux.
    set -- -Xlinker -undefined -Xlinker dynamic_lookup "$@"
fi
"$MOJO" build src/lib.mojo -I src -D VERSION="$VERSION" --emit shared-lib -o bitpattern/bdd.abi3.so "$@"
