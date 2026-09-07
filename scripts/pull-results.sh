#!/usr/bin/env bash
# Pull benchmark JSON off the device.
set -euo pipefail

DEVICE="${DEVICE:-00008120-0002695E3A68C01E}"   # Ixhen's iPhone 14 Pro Max
BUNDLE="${BUNDLE:-com.ixdev.benchapp}"
DEST="${1:-results/device-pull}"

mkdir -p "$DEST"

xcrun devicectl device copy from \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE" \
  --username mobile \
  --source Documents \
  --destination "$DEST"

echo "pulled to $DEST"
