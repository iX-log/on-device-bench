#!/usr/bin/env bash
# Download the pre-converted Core ML models so you can skip conversion.
set -euo pipefail

REPO="iX-log/on-device-bench"
TAG="v0.1-models"
ASSET="whisper-base-encoder-models.zip"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
DEST="${DEST:-models}"
FORCE=0

for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE=1
done

PACKAGES=(
  "whisper-base-encoder-fp16.mlpackage"
  "whisper-base-encoder-int8.mlpackage"
  "whisper-base-encoder-int4.mlpackage"
)

all_present=1
for pkg in "${PACKAGES[@]}"; do
  [ -d "${DEST}/${pkg}" ] || all_present=0
done

if [ "$all_present" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
  echo "All three .mlpackage directories already exist in ${DEST}/. Use --force to redownload."
  exit 0
fi

mkdir -p "$DEST"
TMP_ZIP="$(mktemp -t fetch-models).zip"
trap 'rm -f "$TMP_ZIP"' EXIT

echo "Downloading ${URL}"
curl -L -f -o "$TMP_ZIP" "$URL"

if [ ! -s "$TMP_ZIP" ]; then
  echo "Download failed: ${TMP_ZIP} is empty." >&2
  exit 1
fi

if ! unzip -tq "$TMP_ZIP" >/dev/null; then
  echo "Download failed: ${TMP_ZIP} is not a valid zip file." >&2
  exit 1
fi

unzip -oq "$TMP_ZIP" -d "$DEST"

echo "Landed in ${DEST}/:"
for pkg in "${PACKAGES[@]}"; do
  if [ -d "${DEST}/${pkg}" ]; then
    echo "  ${pkg} ($(du -sh "${DEST}/${pkg}" | cut -f1))"
  else
    echo "  ${pkg} MISSING" >&2
  fi
done
