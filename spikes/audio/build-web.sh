#!/usr/bin/env bash
# Build the worklet wasm and drop it next to the page.
#
# There is deliberately no wasm-bindgen / trunk step for the worklet module:
# see FINDINGS.md "why no wasm-bindgen". The output is a raw cdylib with a C
# ABI and zero imports.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release -p spike-worklet --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/spike_worklet.wasm web/spike_worklet.wasm

# Optional: `wasm-opt -Oz` typically takes another 15-25% off. Not required.
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz web/spike_worklet.wasm -o web/spike_worklet.wasm
  echo "wasm-opt applied"
fi

printf 'wasm: %s bytes (%s gzipped)\n' \
  "$(stat -c%s web/spike_worklet.wasm)" \
  "$(gzip -9 -c web/spike_worklet.wasm | wc -c)"
echo "serve with:  python3 -m http.server 8099 --directory web"
