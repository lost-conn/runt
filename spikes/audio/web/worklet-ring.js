// PATH B consumer. Does no DSP at all -- it drains the SharedArrayBuffer ring
// the worker fills. This is the classic "audio thread only copies" shape.

class RuntRingProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.ctrl = new Int32Array(o.ctrl);
    this.data = new Float32Array(o.data);
    this.capacity = o.capacity;

    this.quanta = 0;
    this.underruns = 0;   // ring was empty when audio needed frames
    this.glitches = 0;    // frame-clock discontinuity (same metric as path A)
    this.lastFrame = -1;
    this.rms = 0;

    this.port.onmessage = (ev) => {
      if (ev.data.t === 'ping') {
        this.pendingPing = ev.data.id;
      }
    };
  }

  process(inputs, outputs) {
    if (this.lastFrame >= 0 && currentFrame - this.lastFrame !== 128) {
      this.glitches++;
    }
    this.lastFrame = currentFrame;

    if (this.pendingPing !== undefined) {
      this.port.postMessage({ t: 'pong', id: this.pendingPing, frame: currentFrame });
      this.pendingPing = undefined;
    }

    const out = outputs[0];
    const frames = out[0].length;
    const w = Atomics.load(this.ctrl, 0);
    const r = Atomics.load(this.ctrl, 1);

    if (w - r < frames) {
      this.underruns++;
      out[0].fill(0);
      if (out.length > 1) out[1].fill(0);
    } else {
      const base = (r % this.capacity) * 2;
      let sum = 0;
      const l = out[0], rr = out.length > 1 ? out[1] : null;
      for (let i = 0; i < frames; i++) {
        const o = base + i * 2;
        const a = this.data[o], b = this.data[o + 1];
        l[i] = a;
        if (rr) rr[i] = b;
        sum += a * a;
      }
      this.rms = Math.sqrt(sum / frames);
      Atomics.store(this.ctrl, 1, r + frames);
    }

    if (++this.quanta % 12 === 0) {
      this.port.postMessage({
        t: 'stat',
        rms: this.rms,
        quanta: this.quanta,
        underruns: this.underruns,
        glitches: this.glitches,
        depth: w - r,
        frame: currentFrame,
      });
    }
    return true;
  }
}

registerProcessor('runt-ring', RuntRingProcessor);
