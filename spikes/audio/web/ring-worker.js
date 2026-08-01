// PATH B producer. A plain Worker running the SAME wasm synth, writing
// interleaved stereo frames into a SharedArrayBuffer ring that the worklet
// drains. Only reachable when `crossOriginIsolated === true`.

let wasm = null, synth = null, left = null, right = null, maxFrames = 0;
let ctrl = null, data = null, capacity = 0;
let timer = null;

const W = 0; // ctrl[0] write cursor, in frames, monotonic
const R = 1; // ctrl[1] read cursor,  in frames, monotonic
const PRODUCED = 2;

function refreshViews() {
  const e = wasm.exports;
  left = new Float32Array(e.memory.buffer, e.synth_left_ptr(synth), maxFrames);
  right = new Float32Array(e.memory.buffer, e.synth_right_ptr(synth), maxFrames);
}

function fill() {
  const e = wasm.exports;
  // Keep the ring topped up, but never overwrite unread frames.
  for (;;) {
    const w = Atomics.load(ctrl, W);
    const r = Atomics.load(ctrl, R);
    if (w - r > capacity - 256) break;

    e.synth_render(synth, 128);
    const base = (w % capacity) * 2;
    for (let i = 0; i < 128; i++) {
      const o = base + i * 2;
      // The ring length is a multiple of 128 so a block never wraps mid-way.
      data[o] = left[i];
      data[o + 1] = right[i];
    }
    Atomics.store(ctrl, W, w + 128);
    Atomics.add(ctrl, PRODUCED, 128);
  }
}

self.onmessage = (ev) => {
  const m = ev.data;
  switch (m.t) {
    case 'init': {
      wasm = new WebAssembly.Instance(m.module, {});
      const e = wasm.exports;
      synth = e.synth_new(m.sampleRate);
      maxFrames = e.synth_max_frames();
      refreshViews();
      ctrl = new Int32Array(m.ctrl);
      data = new Float32Array(m.data);
      capacity = m.capacity;
      fill();
      // 4 ms cadence: well inside the ring depth, cheap enough to ignore.
      timer = setInterval(fill, 4);
      self.postMessage({ t: 'ready' });
      break;
    }
    case 'cutoff':  wasm.exports.synth_set_cutoff(synth, m.v); break;
    case 'drone':   wasm.exports.synth_set_drone(synth, m.v); break;
    case 'pluckHz': wasm.exports.synth_set_pluck_hz(synth, m.v); break;
    case 'trigger': wasm.exports.synth_trigger(synth); break;
    case 'stop':    clearInterval(timer); break;
  }
};
