#!/bin/bash
# Verify a GGUF file's SHA256 against the publisher's LFS oid.
# Usage: ./verify-checksum.sh <file> <expected-sha256>

set -euo pipefail

FILE="${1:?Usage: verify-checksum.sh <file> <expected-sha256>}"
EXPECTED="${2:?Expected SHA256 required}"

if [ ! -f "$FILE" ]; then
    echo "ERROR: File not found: $FILE"
    exit 1
fi

echo "Computing SHA256 for $(basename "$FILE")..."
ACTUAL=$(sha256sum "$FILE" | awk '{print $1}')

echo "Expected: $EXPECTED"
echo "Actual:   $ACTUAL"

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "✓ SHA256 matches — file integrity verified"
    exit 0
else
    echo "✗ SHA256 MISMATCH — file is corrupt. Re-download required."
    exit 1
fi
