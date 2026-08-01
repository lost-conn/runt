# Audio spike — findings

Date: 2026-08-01 · Environment: Chrome 151 headless + native x86-64 Linux ·
Status: **§8 risk retired.** Verdict adopted into DESIGN.md §8.

## Verdict

**Path A — the synth running *inside* the AudioWorklet, params via
`postMessage`, no SharedArrayBuffer — is the baseline.** Path B (SAB ring
buffer) works but is strictly worse on our constraints and is not adopted.

| | Path A (worklet-resident) | Path B (SAB ring) |
|---|---|---|
| Works on GH Pages as-is | **yes** | no — needs COI shim |
| Added buffering latency | **0** | 160 ms at glitch-free ring size |
| Glitches, 55.6 s soak | **1 / 20 832 quanta (0.005%)** | 0 @160 ms ring; 0.08–0.17% @32/10.7 ms |
| Moving parts | worklet only | worklet + worker + ring + service worker |

## Key numbers (measured, not estimated)

- **Latency (100 postMessage round trips, 0 lost):** median **6.6 ms**, p95
  10.8, max 11.4, min 0.4. The spread is exactly Chrome's 512-frame render
  block — a message waits for the next audio callback, nothing more.
  Realistic event→sound ≈ **25–55 ms on real hardware**, dominated by the OS
  audio stack. Acceptable for game SFX ("tight" to "passable").
- **Worklet wasm:** 200 134 B raw / **83 738 B gzipped**; 6.2 ms to compile;
  fetched lazily on first audio start. (`wasm-opt` not installed here; would
  cut another ~15–25%.)
- **CPU per 128-frame stereo quantum:** native **10.10 µs**, wasm **11.99 µs**
  (1.19×) against a 2 666 µs budget = **0.45% of one core**. Audio is not a
  perf risk; voice count is bounded by taste, not CPU.
- **Path B ring tradeoff (from cursor deltas):** 8192 fr → 160.0 ms, 0
  underruns; 2048 fr → 32.0 ms, 0.08%; 1024 fr → 10.7 ms, 0.17%. Low latency
  and clean audio are mutually exclusive on this path.
- **cpal native:** 1126 callbacks / 12 s, **0 xruns**.

## Determinism — honest result

- Holds **within a platform**: separate native processes agree
  (`0xcc9ec2a6ec256bfd`); 3/3 identical runs inside wasm
  (`0xfe98d88ffabef45d`). Zero subnormals, zero NaN/Inf.
- **Native ≠ wasm**: max 1.2e-6 relative / ~1e-10 absolute divergence —
  last-few-ULP rounding from differing libm `sin`/`tan` accumulated through
  IIR state. Perceptually identical. Matches DESIGN §4's existing stance
  (same build + platform); §8 must not promise cross-platform bit-identity —
  and now doesn't.

## COOP/COEP story for GitHub Pages

- Path A needs nothing. The enabling trick: **`WebAssembly.Module` is
  structured-cloneable** — compile on the main thread, pass through
  `processorOptions`, instantiate *synchronously* in the processor
  constructor; live on the first `process()` call. The module has **zero
  imports** (verified).
- wasm-bindgen is deliberately absent from the worklet: its glue needs
  `TextEncoder`/`TextDecoder`, which `AudioWorkletGlobalScope` lacks
  (wasm-bindgen#2367, open since 2020). Its official `wasm_audio_worklet`
  example is built on rayon threads → SAB → exactly what GH Pages can't do.
- coi-serviceworker v0.1.7 **does** work (verified `crossOriginIsolated:
  true` from a header-less static server) but costs: first-load reload,
  sticky registration, secure-context requirement, COEP breaks cross-origin
  embeds. **Not adopted.**

## fundsp 0.23.0 landmines (for future agents)

1. `hacker`/`hacker32` modules are **gone** → `prelude`/`prelude32`/
   `prelude64`. All tutorials and stale LLM knowledge say `hacker32`.
2. docs.rs has **no 0.23.0 docs** (its build failed; stops at 0.20). Read
   `~/.cargo/registry/src/*/fundsp-0.23.0/src/prelude32.rs` directly.
3. `Shape` is a *trait*: `shape(Tanh(0.9))`, not `Shape::Tanh(..)`.
4. Seeding: `net.ping(false, AttoHash::new(seed))` then `reset()`, in that
   order.
5. `Shared` is f32-only; use it for realtime params instead of rebuilding
   graphs.
6. `lowpass()` (3 inputs) is sweepable; `lowpass_hz()` bakes the frequency.
7. Call `.allocate()` before going realtime.
8. Pulls `glam 0.28` while runt is on 0.33 → duplicate glam if ever added to
   `runt-core`. Keep audio in its own crate.

**cpal 0.18.1** vs documented 0.15: `build_output_stream` takes config **by
value**; `sample_rate` is a bare `u32`; `device.id()` replaces `name()`;
`ErrorKind::Xrun` is the native glitch signal.

**Worklet/wasm:** growing Rust's heap **detaches every `Float32Array` view**
— re-create views after any allocating export (see `refreshViews()`).
`outputs[0][ch]` is planar. `currentTime` does not advance within one
`process()` call.

**Trunk integration (verified):** a `pre_build` hook + two `copy-file` links
places the raw cdylib alongside the wasm-bindgen bundle in `dist/`. Gotchas:
`copy-file` hrefs resolve relative to index.html; copied files keep unhashed
names.

## Glitch methodology + caveat

The processor records `currentFrame` per call and flags any delta ≠ 128 (the
spec guarantees one call per quantum). This measures **render-thread
starvation**; no browser hook exists for device-buffer underruns downstream.
Headless Chrome uses a null sink with no device-clock deadline, so near-zero
glitch counts mean "the worklet keeps up", not "click-free on a loaded
laptop" — see manual steps.

## Manual verification remaining (needs a human + speakers)

1. `cargo run -p spike-native --release -- play` (or `-- wav /tmp/x.wav`).
2. `./build-web.sh && python3 -m http.server 8099 --directory web` →
   `localhost:8099`: confirm `crossOriginIsolated: false`, click start, drag
   cutoff, click trigger; soak under load, watch the glitch counter.
3. Read real `outputLatency` (0 here — null sink); it's the missing term in
   the latency budget.
4. `?coi=1` for path B; `?coi=1&ring=1024` to *hear* the underruns.
5. Confirm on the real GH Pages origin (formality — path A depends on
   nothing).
6. **Mobile untested** — mid-range Android + iOS Safari. iOS is the real
   unknown.

## Layout

```
spikes/audio/                  own cargo workspace — NOT a root member
├─ Cargo.toml  .gitignore  build-web.sh  FINDINGS.md
├─ patch/src/lib.rs      THE PATCH — shared verbatim by native + wasm
├─ native/src/main.rs    cpal host: hash|bench|analyze|play|wav
├─ worklet/src/lib.rs    raw cdylib, extern "C", zero imports
└─ web/  index.html  app.js  worklet-direct.js (path A)
         worklet-ring.js + ring-worker.js (path B)
         coi-serviceworker.js (vendored v0.1.7, MIT)  spike_worklet.wasm (gitignored)
```

`native -- analyze` exists because nobody can listen on this box: it measures
the rendered buffer directly — autocorrelation pitch tracks to ≤0.31%, cutoff
sweep moves the spectral centroid monotonically 126.6→210.9 Hz, live
`set_cutoff_hz` moves it 143.0→223.8 Hz, `trigger()` takes RMS 0→0.141 with
a clean decay.
