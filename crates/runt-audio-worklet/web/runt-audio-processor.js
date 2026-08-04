// The AudioWorkletProcessor half of DESIGN §8. Adapted from
// `spikes/audio/web/worklet-direct.js`, which is where the approach was proven.
//
// The three AudioWorkletGlobalScope landmines and how this file dodges them:
//
//  1. No TextEncoder/TextDecoder  -> the wasm is a raw cdylib with a C ABI, so
//     there is no wasm-bindgen glue that would need them.
//  2. No fetch()                  -> the main thread compiles the module and
//     ships it in `processorOptions`. `WebAssembly.Module` is structured-
//     cloneable, so this needs no SharedArrayBuffer and no COOP/COEP — which is
//     what lets the demo ship on GitHub Pages, a host that cannot set headers.
//  3. Async instantiation would leave the first quanta silent -> instantiating
//     an ALREADY-COMPILED module is synchronous and always permitted, so the
//     processor is live on its very first process() call.
//
// THE ONE RULE: Rust's allocator can grow linear memory, and growing it
// DETACHES every existing ArrayBuffer view. Only two exports allocate —
// `runt_audio_new` and `runt_audio_load_bank`, both called in the constructor —
// and `refreshViews()` runs after them. Nothing in `process()` allocates, which
// is what makes the views safe to hold for the life of the page.

class RuntAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { module, bank, masterGain } = options.processorOptions;

    // Synchronous: `module` is already compiled and imports nothing at all.
    this.wasm = new WebAssembly.Instance(module, {});
    const e = this.wasm.exports;

    this.synth = e.runt_audio_new(sampleRate);
    this.maxFrames = e.runt_audio_max_frames();
    if (typeof masterGain === 'number') {
      e.runt_audio_set_master_gain(this.synth, masterGain);
    }

    // Load the patch bank before any view is cached: decoding it allocates.
    this.patches = 0;
    this.bankError = null;
    if (bank && bank.byteLength) {
      const ptr = e.runt_audio_bank_ptr(this.synth);
      const capacity = e.runt_audio_bank_capacity();
      const n = Math.min(bank.byteLength, capacity);
      new Uint8Array(e.memory.buffer, ptr, capacity).set(new Uint8Array(bank, 0, n));
      this.patches = e.runt_audio_load_bank(this.synth, n);
      if (this.patches === 0) this.bankError = 'bank did not decode';
    } else {
      this.bankError = 'no bank supplied';
    }

    this.refreshViews();

    // --- instrumentation --------------------------------------------------
    this.quanta = 0;
    this.lastFrame = -1;
    // A "glitch" is a discontinuity in the context's frame clock between two
    // consecutive process() calls. process() is specified to run once per render
    // quantum for an active node, so any delta != 128 means the audio thread
    // missed its deadline. This measures render-thread starvation only; no
    // browser hook exists for device-buffer underruns downstream (FINDINGS,
    // "Glitch methodology + caveat").
    this.glitches = 0;
    this.worstGap = 0;
    this.submitted = 0;
    this.dropped = 0;

    // Queue rather than act: every message is applied at a known point in the
    // block (the top), which is what makes the timing reproducible.
    this.pending = [];
    this.port.onmessage = (ev) => {
      const data = ev.data;
      if (data && data.t === 'poll') {
        this.report();
        return;
      }
      if (this.pending.length >= 64) {
        // A page that stopped rendering can back a queue up; drop the oldest
        // rather than fire a hundred notes at once when it comes back.
        this.dropped += this.pending.shift().byteLength / 32;
        return;
      }
      this.pending.push(data);
    };

    this.report('ready');
  }

  refreshViews() {
    const e = this.wasm.exports;
    this.left = new Float32Array(
      e.memory.buffer, e.runt_audio_left_ptr(this.synth), this.maxFrames);
    this.right = new Float32Array(
      e.memory.buffer, e.runt_audio_right_ptr(this.synth), this.maxFrames);
    this.events = new Uint8Array(
      e.memory.buffer, e.runt_audio_events_ptr(this.synth),
      e.runt_audio_events_capacity());
  }

  // Copy each queued blob into the wasm scratch and apply it. No allocation:
  // `set` writes into an existing view and `runt_audio_submit` cannot grow the
  // heap.
  drain() {
    if (this.pending.length === 0) return;
    const e = this.wasm.exports;
    for (const blob of this.pending) {
      const src = new Uint8Array(blob);
      const n = Math.min(src.length, this.events.length);
      if (n < src.length) this.dropped += (src.length - n) / 32;
      this.events.set(n === src.length ? src : src.subarray(0, n));
      this.submitted += e.runt_audio_submit(this.synth, n);
    }
    this.pending.length = 0;
  }

  report(kind) {
    const e = this.wasm.exports;
    this.port.postMessage({
      t: kind || 'stat',
      patches: this.patches,
      bankError: this.bankError,
      sampleRate,
      rms: e.runt_audio_last_rms(this.synth),
      voices: e.runt_audio_active_voices(this.synth),
      played: e.runt_audio_played(this.synth),
      nanGuarded: e.runt_audio_nan_guarded(this.synth),
      quanta: this.quanta,
      submitted: this.submitted,
      dropped: this.dropped,
      glitches: this.glitches,
      worstGap: this.worstGap,
    });
  }

  process(inputs, outputs) {
    // Frame-clock continuity check, before any work.
    if (this.lastFrame >= 0) {
      const gap = currentFrame - this.lastFrame;
      if (gap !== 128) {
        this.glitches++;
        if (gap > this.worstGap) this.worstGap = gap;
      }
    }
    this.lastFrame = currentFrame;

    this.drain();

    const out = outputs[0];
    const frames = out[0].length;
    this.wasm.exports.runt_audio_render(this.synth, frames);
    out[0].set(this.left.subarray(0, frames));
    if (out.length > 1) out[1].set(this.right.subarray(0, frames));

    // Roughly six times a second at 48 kHz. Cheap, and it is the only way a
    // headless harness can see that sound is being produced.
    if (++this.quanta % 64 === 0) this.report();

    return true; // keep the node alive even while silent
  }
}

registerProcessor('runt-audio', RuntAudioProcessor);
