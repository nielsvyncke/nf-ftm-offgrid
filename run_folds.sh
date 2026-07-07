#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for i in {0..4}; do
    bash "$SCRIPT_DIR/train.sh" --model inr --fold $i "$@"
done
