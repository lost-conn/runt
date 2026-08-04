#!/usr/bin/env bash
# Build the AudioWorklet's wasm module (DESIGN §8).
#
# Run as a trunk `pre_build` hook — see Trunk.toml — so `trunk build` and
# `trunk serve` produce a complete `dist/` with no second command to remember.
# The recipe is the one verified in `spikes/audio/FINDINGS.md`: a raw `cdylib`
# with a C ABI and **zero imports**, built separately from the wasm-bindgen
# bundle and dropped beside it by two `copy-file` links in index.html.
#
# There is deliberately no wasm-bindgen step. `AudioWorkletGlobalScope` has no
# `TextEncoder`/`TextDecoder`, which wasm-bindgen's glue requires
# (rustwasm/wasm-bindgen#2367); exporting plain `extern "C"` functions requires
# neither.
set -euo pipefail
cd "$(dirname "$0")"

PROFILE=wasm-worklet
OUT=../../target/wasm32-unknown-unknown/${PROFILE}/runt_audio_worklet.wasm

cargo build -p runt-audio-worklet --target wasm32-unknown-unknown --profile "${PROFILE}"

# Optional. `wasm-opt -Oz` took another 15–25% off in the spike; it is not
# required and CI does not have it.
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz "${OUT}" -o "${OUT}"
  echo "worklet: wasm-opt applied"
fi

printf 'worklet: %s bytes (%s gzipped) → %s\n' \
  "$(stat -c%s "${OUT}")" \
  "$(gzip -9 -c "${OUT}" | wc -c)" \
  "${OUT}"
