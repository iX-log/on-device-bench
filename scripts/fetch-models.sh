#!/usr/bin/env bash
# Download the pre-converted Core ML models so you can skip conversion,
# and place them where both the Python side and Xcode expect them.
set -euo pipefail

REPO="iX-log/on-device-bench"
TAG="v0.1-models"
ASSET="whisper-base-encoder-models.zip"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
DEST="${DEST:-models}"
XCODE_DEST="${XCODE_DEST:-ios/BenchApp/BenchApp}"
FORCE=0

for arg in "$@"; do
  [ "$arg" = "--force" ] && FORCE=1
done

PACKAGES=(
  "whisper-base-encoder-fp16.mlpackage"
  "whisper-base-encoder-int8.mlpackage"
  "whisper-base-encoder-int4.mlpackage"
)

present_in() {
  local dir="$1" pkg
  for pkg in "${PACKAGES[@]}"; do
    [ -d "${dir}/${pkg}" ] || return 1
  done
  return 0
}

if present_in "$DEST" && [ "$FORCE" -eq 0 ]; then
  echo "All three .mlpackage directories already exist in ${DEST}/. Use --force to redownload."
else
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
fi

# Xcode's BenchApp target is a file-system-synchronized group rooted at
# ios/BenchApp/BenchApp: anything placed there is picked up automatically,
# so a plain copy is what "Copy items if needed" used to do by hand.
mkdir -p "$XCODE_DEST"
for pkg in "${PACKAGES[@]}"; do
  if [ ! -d "${XCODE_DEST}/${pkg}" ] || [ "$FORCE" -eq 1 ]; then
    rm -rf "${XCODE_DEST:?}/${pkg}"
    cp -R "${DEST}/${pkg}" "${XCODE_DEST}/${pkg}"
  fi
done

echo "Landed in ${DEST}/ and ${XCODE_DEST}/:"
for pkg in "${PACKAGES[@]}"; do
  if [ -d "${DEST}/${pkg}" ] && [ -d "${XCODE_DEST}/${pkg}" ]; then
    echo "  ${pkg} ($(du -sh "${DEST}/${pkg}" | cut -f1))"
  else
    echo "  ${pkg} MISSING" >&2
  fi
done
