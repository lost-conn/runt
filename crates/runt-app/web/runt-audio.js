// The browser half of runt's audio host (DESIGN §8).
//
// This file owns everything that is *browser* rather than *engine*: creating the
// AudioContext, compiling the worklet's wasm module on the main thread, and
// waiting for the user gesture that autoplay policy demands. The wasm game
// module calls exactly two functions on it — `init` and `submit` — through
// `crates/runt-app/src/audio.rs`.
//
// It is JavaScript rather than web-sys for two reasons. The code would be the
// same shape either way (object literals become `Reflect::set`), and a page that
// does not include this script simply has no audio: `window.runtAudio` is
// undefined, the Rust side logs once and installs the silent backend. That is
// the same opt-in-by-DOM rule the `#runt-status` HUD already follows.
//
// Nothing here knows what a patch is. The bank is an opaque byte array from
// Rust to the synth; the events are opaque 32-byte records.

(() => {
  const DEFAULTS = {
    // Both are `copy-file` links in the page's index.html, so they land beside
    // it with unhashed names (FINDINGS, "Trunk integration").
    wasmUrl: 'runt_audio_worklet.wasm',
    processorUrl: 'runt-audio-processor.js',
    masterGain: 1.0,
  };

  const state = {
    // idle → booting → armed (context suspended, waiting for a gesture)
    //                 → running | failed
    phase: 'idle',
    error: null,
    ctx: null,
    node: null,
    bank: null,
    options: { ...DEFAULTS },
    wasmBytes: 0,
    compileMs: 0,
    submitted: 0,
    droppedBeforeStart: 0,
    // Last report from the processor. The headless harness reads this.
    stats: null,
  };

  async function boot() {
    if (state.phase !== 'idle') return;
    state.phase = 'booting';
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) throw new Error('no AudioContext');
      const ctx = new Ctx();
      if (!ctx.audioWorklet) throw new Error('no AudioWorklet');

      // Fetch and compile in parallel with loading the processor module. The
      // wasm is compiled ONCE here, on the main thread, where compilation is
      // async and unrestricted; the resulting `WebAssembly.Module` is
      // structured-cloneable, which is the whole trick (FINDINGS).
      const t0 = performance.now();
      const [bytes] = await Promise.all([
        fetch(state.options.wasmUrl).then((r) => {
          if (!r.ok) throw new Error(`${state.options.wasmUrl}: HTTP ${r.status}`);
          return r.arrayBuffer();
        }),
        ctx.audioWorklet.addModule(state.options.processorUrl),
      ]);
      const module = await WebAssembly.compile(bytes);
      state.wasmBytes = bytes.byteLength;
      state.compileMs = +(performance.now() - t0).toFixed(2);

      const node = new AudioWorkletNode(ctx, 'runt-audio', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: {
          module,
          bank: state.bank,
          masterGain: state.options.masterGain,
        },
      });
      node.port.onmessage = (ev) => {
        state.stats = ev.data;
      };
      node.connect(ctx.destination);

      state.ctx = ctx;
      state.node = node;
      // A context created outside a gesture starts suspended; `resume()` on the
      // first gesture is what actually starts it. If policy already allows it,
      // the resume below succeeds immediately and nothing is waiting.
      state.phase = ctx.state === 'running' ? 'running' : 'armed';
      resume();
    } catch (e) {
      state.phase = 'failed';
      state.error = String(e);
      console.warn('runt-audio:', e);
    }
  }

  function resume() {
    if (!state.ctx || state.phase === 'failed') return;
    state.ctx.resume().then(
      () => {
        if (state.ctx.state === 'running') state.phase = 'running';
      },
      () => {},
    );
  }

  // Autoplay policy: a context may only start from a user gesture. Capture
  // phase, so a canvas that stops propagation cannot swallow the one event we
  // need, and `once` semantics per type so the listeners cost nothing after the
  // first click. Deliberately silent — no "click to enable sound" banner, no DOM
  // the page did not ask for.
  const GESTURES = ['pointerdown', 'keydown', 'touchend'];
  function armGesture() {
    const onGesture = () => {
      if (state.phase === 'idle') boot();
      else resume();
      if (state.phase === 'running') {
        for (const type of GESTURES) window.removeEventListener(type, onGesture, true);
      }
    };
    for (const type of GESTURES) window.addEventListener(type, onGesture, true);
  }

  window.runtAudio = {
    /// Called once by the wasm module. `bank` is a Uint8Array of postcard bytes.
    init(bank, options) {
      state.bank = bank && bank.byteLength ? bank.slice().buffer : null;
      if (options) {
        for (const key of Object.keys(DEFAULTS)) {
          if (options[key] !== undefined) state.options[key] = options[key];
        }
      }
      armGesture();
      // Build everything now so the first gesture only has to `resume()` — the
      // fetch and compile are ~6 ms but they are not worth spending after a
      // click that the player expects to make a noise.
      boot();
    },

    /// One tick-batch of 32-byte wire records, as a Uint8Array.
    ///
    /// Dropped while the context is not running: a player who has not clicked
    /// yet should get silence, not a backlog fired all at once on their first
    /// click.
    submit(bytes) {
      if (!bytes || !bytes.byteLength) return;
      // Count BEFORE posting: transferring a buffer detaches it, and a detached
      // `Uint8Array` reports `byteLength === 0`.
      const count = bytes.byteLength / 32;
      if (state.phase !== 'running' || !state.node) {
        state.droppedBeforeStart += count;
        return;
      }
      // Transfer rather than copy: the buffer came from `Uint8Array::from` on
      // the Rust side and is nobody else's.
      state.node.port.postMessage(bytes.buffer, [bytes.buffer]);
      state.submitted += count;
    },

    /// Ask the processor to report immediately. Headless verification uses this
    /// rather than waiting for the periodic report.
    poll() {
      if (state.node) state.node.port.postMessage({ t: 'poll' });
    },

    /// Start without waiting for a gesture. Only useful where policy already
    /// allows it (a headless browser run with
    /// `--autoplay-policy=no-user-gesture-required`).
    start() {
      if (state.phase === 'idle') boot();
      else resume();
    },

    /// Everything a harness or a bug report wants. Read-only by convention.
    debug: state,
  };
})();
