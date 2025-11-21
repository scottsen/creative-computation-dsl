#!/usr/bin/env bash
# Progressive Reveal CLI Wrapper for Morphogen Project
#
# This script provides easy access to the reveal tool for exploring
# Morphogen codebase files at different levels of detail.
#
# Installation: pip install git+https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400
# Or: Save gist files locally and pip install -e .
#
# Usage:
#   ./scripts/reveal.sh 0 src/morphogen/domains/audio.py        # Metadata
#   ./scripts/reveal.sh 1 src/morphogen/domains/audio.py        # Structure
#   ./scripts/reveal.sh 2 docs/specifications/SPEC-AUDIO.md     # Preview
#   ./scripts/reveal.sh 3 SPECIFICATION.md --page-size 50       # Full content

set -euo pipefail

LEVEL=${1:-1}
shift || true

# Check if reveal is installed
if ! command -v reveal &> /dev/null; then
    echo "Error: reveal tool not installed"
    echo ""
    echo "Install from gist:"
    echo "  pip install git+https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400"
    echo ""
    echo "Or manually:"
    echo "  1. Clone gist: git clone https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400 reveal-cli"
    echo "  2. Install: cd reveal-cli && pip install -e ."
    exit 1
fi

# Run reveal with specified level
exec reveal --level "$LEVEL" "$@"
