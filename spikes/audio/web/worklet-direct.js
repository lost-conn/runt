// PATH A -- no SharedArrayBuffer. The synth runs INSIDE the worklet.
//
// The three AudioWorkletGlobalScope landmines and how this file dodges them:
//
//  1. No TextEncoder/TextDecoder  -> we use a raw cdylib with a C ABI, so there
//     is no wasm-bindgen glue that would need them.
//  2. No fetch()                  -> the main thread compiles the module and
//     ships it in `processorOptions`. `WebAssembly.Module` is structured-
//     cloneable, so this needs no SharedArrayBuffer and no COOP/COEP.
//  3. Async instantiation would leave the first quanta silent -> instantiating
//     an ALREADY-COMPILED module is synchronous and always permitted, so the
//     processor is live on its very first process() call.

class RuntSynthProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const { module } = options.processorOptions;
    // Synchronous: `module` is already compiled, and the module imports
    // nothing at all (verified: WebAssembly.Module.imports() === []).
    this.wasm = new WebAssembly.Instance(module, {});
    const e = this.wasm.exports;

    this.synth = e.synth_new(sampleRate);
    this.maxFrames = e.synth_max_frames();
    this.refreshViews();

    // --- instrumentation ------------------------------------------------
    this.quanta = 0;
    this.lastFrame = -1;
    // A "glitch" is a discontinuity in the context's frame clock between two
    // consecutive process() calls. process() is specified to run once per
    // render quantum for an active node, so any delta != 128 means the audio
    // thread missed its deadline and the context skipped ahead.
    this.glitches = 0;
    this.worstGap = 0;
    this.pending = [];
    this.reportEvery = 12;

    this.port.onmessage = (ev) => {
      // Runs on the worklet thread's event loop, i.e. BETWEEN render quanta.
      // Queue rather than act, so every message is applied at a known point
      // in the block -- that is what makes the timing reproducible.
      this.pending.push(ev.data);
    };
  }

  // Rust's allocator can grow linear memory, which DETACHES every existing
  // ArrayBuffer view. Anything that may allocate must be followed by this.
  // (synth_render does not allocate; synth_bench and canonical_hash_* do.)
  refreshViews() {
    const e = this.wasm.exports;
    this.left = new Float32Array(
      e.memory.buffer, e.synth_left_ptr(this.synth), this.maxFrames);
    this.right = new Float32Array(
      e.memory.buffer, e.synth_right_ptr(this.synth), this.maxFrames);
  }

  drain() {
    const e = this.wasm.exports;
    for (const msg of this.pending) {
      switch (msg.t) {
        case 'cutoff':   e.synth_set_cutoff(this.synth, msg.v); break;
        case 'drone':    e.synth_set_drone(this.synth, msg.v); break;
        case 'pluckHz':  e.synth_set_pluck_hz(this.synth, msg.v); break;
        case 'trigger':  e.synth_trigger(this.synth); break;
        case 'ping':
          // Acknowledge from inside process(): the round trip therefore
          // includes "wait for the next render quantum", which is exactly the
          // quantity the tick-boundary latency question asks about.
          this.port.postMessage({ t: 'pong', id: msg.id, frame: currentFrame });
          break;
        case 'bench': {
          const t0 = currentTime;
          const acc = e.synth_bench(this.synth, msg.quanta);
          const t1 = currentTime;
          this.refreshViews(); // synth_bench allocates
          this.port.postMessage({ t: 'benchDone', quanta: msg.quanta, acc,
                                  audioClockDelta: t1 - t0 });
          break;
        }
        case 'samples': {
          const ptr = e.canonical_ptr();
          const len = e.canonical_len();
          // Re-view AFTER the call: canonical_render allocates and may have
          // grown (and thus detached) memory.
          const all = new Float32Array(e.memory.buffer, ptr, len);
          this.refreshViews();
          this.port.postMessage({ t: 'samplesDone',
                                  head: Array.from(all.subarray(0, msg.n || 16)) });
          break;
        }
        case 'hash': {
          const lo = e.canonical_hash_lo() >>> 0;
          const hi = e.canonical_hash_hi() >>> 0;
          this.refreshViews(); // canonical_render allocates
          this.port.postMessage({ t: 'hashDone', lo, hi });
          break;
        }
      }
    }
    this.pending.length = 0;
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
    this.wasm.exports.synth_render(this.synth, frames);
    out[0].set(this.left.subarray(0, frames));
    if (out.length > 1) out[1].set(this.right.subarray(0, frames));

    if (++this.quanta % this.reportEvery === 0) {
      this.port.postMessage({
        t: 'stat',
        rms: this.wasm.exports.synth_last_rms(this.synth),
        quanta: this.quanta,
        glitches: this.glitches,
        worstGap: this.worstGap,
        frame: currentFrame,
        time: currentTime,
      });
    }
    return true; // keep alive
  }
}

registerProcessor('runt-synth', RuntSynthProcessor);
