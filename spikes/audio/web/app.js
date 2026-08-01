// Main-thread driver for both paths. Everything interesting is mirrored onto
// `window.spike` so a headless browser can assert on it without a speaker.

const $ = (id) => document.getElementById(id);
const WASM_URL = 'spike_worklet.wasm';

const spike = {
  env: {},
  a: { started: false, stat: null, latency: null, bench: null, hash: null, errors: [] },
  b: { started: false, stat: null, latency: null, errors: [] },
};
window.spike = spike;

window.addEventListener('error', (e) => spike.a.errors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => spike.a.errors.push(String(e.reason)));

// ---------------------------------------------------------------- environment

spike.env = {
  crossOriginIsolated: self.crossOriginIsolated === true,
  sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
  audioWorklet: typeof AudioWorklet !== 'undefined',
  coiShimRequested: new URLSearchParams(location.search).get('coi') === '1',
  ua: navigator.userAgent,
};
$('env').textContent = Object.entries(spike.env)
  .map(([k, v]) => `${k.padEnd(22)} : ${v}`).join('\n');

// ------------------------------------------------------------- shared helpers

let wasmModule = null;
async function getModule() {
  if (!wasmModule) {
    const t0 = performance.now();
    const bytes = await fetch(WASM_URL).then((r) => r.arrayBuffer());
    // Compiled ONCE on the main thread (async, no size limit). The resulting
    // WebAssembly.Module is structured-cloneable, so it can be handed to the
    // worklet and to a worker without SharedArrayBuffer.
    wasmModule = await WebAssembly.compile(bytes);
    spike.env.wasmBytes = bytes.byteLength;
    spike.env.wasmCompileMs = +(performance.now() - t0).toFixed(2);
    $('env').textContent += `\nwasmBytes              : ${bytes.byteLength}` +
                            `\nwasmCompileMs          : ${spike.env.wasmCompileMs}`;
  }
  return wasmModule;
}

function stats(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(s.length * p))];
  return {
    n: s.length,
    min: +s[0].toFixed(3),
    median: +q(0.5).toFixed(3),
    p95: +q(0.95).toFixed(3),
    max: +s[s.length - 1].toFixed(3),
    mean: +(s.reduce((a, b) => a + b, 0) / s.length).toFixed(3),
  };
}

/// Send `n` pings spaced `gapMs` apart and time the acknowledgement that the
/// processor emits from inside process(). The round trip therefore contains a
/// full "wait for the next render quantum".
async function measureLatency(port, n = 100, gapMs = 20) {
  const pending = new Map();
  const samples = [];
  const prev = port.onmessage;
  port.onmessage = (ev) => {
    if (ev.data.t === 'pong' && pending.has(ev.data.id)) {
      samples.push(performance.now() - pending.get(ev.data.id));
      pending.delete(ev.data.id);
    } else if (prev) prev(ev);
  };
  for (let i = 0; i < n; i++) {
    pending.set(i, performance.now());
    port.postMessage({ t: 'ping', id: i });
    await new Promise((r) => setTimeout(r, gapMs));
  }
  await new Promise((r) => setTimeout(r, 300));
  port.onmessage = prev;
  return { roundTripMs: stats(samples), lost: n - samples.length };
}

// -------------------------------------------------------------------- path A

let ctxA = null, nodeA = null;

async function startA() {
  const module = await getModule();
  ctxA = new AudioContext({ latencyHint: 'interactive' });
  await ctxA.resume();
  await ctxA.audioWorklet.addModule('worklet-direct.js');

  nodeA = new AudioWorkletNode(ctxA, 'runt-synth', {
    numberOfInputs: 0,
    outputChannelCount: [2],
    processorOptions: { module },   // structured clone, no SAB needed
  });
  nodeA.onprocessorerror = (e) => {
    spike.a.errors.push('processorerror: ' + (e.message || 'AudioWorkletProcessor threw'));
    $('statA').innerHTML = '<span class="bad">processor error — see spike.a.errors</span>';
  };
  nodeA.port.onmessage = (ev) => {
    if (ev.data.t === 'stat') {
      spike.a.stat = ev.data;
      $('statA').textContent =
        `rms      ${ev.data.rms.toFixed(5)}\n` +
        `quanta   ${ev.data.quanta}\n` +
        `glitches ${ev.data.glitches} (frame-clock gaps != 128; worst gap ${ev.data.worstGap})\n` +
        `frame    ${ev.data.frame}   audio clock ${ev.data.time.toFixed(3)} s`;
    } else if (ev.data.t === 'benchDone') {
      spike.a.bench = ev.data;
    } else if (ev.data.t === 'samplesDone') {
      spike.a.samples = ev.data.head;
    } else if (ev.data.t === 'hashDone') {
      // Reassemble the 64-bit FNV-1a the native binary prints.
      const hex = (ev.data.hi >>> 0).toString(16).padStart(8, '0') +
                  (ev.data.lo >>> 0).toString(16).padStart(8, '0');
      spike.a.hash = '0x' + hex;
      $('resultsA').textContent += `\nwasm canonical hash: 0x${hex}`;
    }
  };
  nodeA.connect(ctxA.destination);

  spike.a.started = true;
  // Exposed so a headless browser can drive the patch without a mouse.
  spike.a.send = (m) => nodeA.port.postMessage(m);
  spike.a.measureLatency = (n, gap) => measureLatency(nodeA.port, n, gap);
  spike.env.sampleRate = ctxA.sampleRate;
  spike.env.baseLatencyMs = +(ctxA.baseLatency * 1000).toFixed(3);
  spike.env.outputLatencyMs = +((ctxA.outputLatency || 0) * 1000).toFixed(3);
  $('resultsA').textContent =
    `sampleRate ${ctxA.sampleRate} Hz | baseLatency ${spike.env.baseLatencyMs} ms` +
    ` | outputLatency ${spike.env.outputLatencyMs} ms`;

  for (const id of ['stopA', 'trigger', 'latency', 'bench', 'hash']) $(id).disabled = false;
  $('startA').disabled = true;
}

function stopA() {
  if (nodeA) nodeA.disconnect();
  if (ctxA) ctxA.close();
  ctxA = nodeA = null;
  spike.a.started = false;
  $('startA').disabled = false;
  for (const id of ['stopA', 'trigger', 'latency', 'bench', 'hash']) $(id).disabled = true;
}

$('startA').onclick = () => startA().catch((e) => {
  spike.a.errors.push(String(e));
  $('statA').innerHTML = `<span class="bad">${e}</span>`;
});
$('stopA').onclick = stopA;
$('trigger').onclick = () => nodeA?.port.postMessage({ t: 'trigger' });

const bindSlider = (id, key, fmt = (v) => `${v} Hz`) => {
  $(id).oninput = (e) => {
    const v = +e.target.value;
    $(id + 'V').textContent = fmt(v);
    nodeA?.port.postMessage({ t: key, v });
    workerB?.postMessage({ t: key, v });
  };
};
bindSlider('cutoff', 'cutoff');
bindSlider('drone', 'drone');
bindSlider('pluckHz', 'pluckHz');

$('latency').onclick = async () => {
  $('latency').disabled = true;
  $('resultsA').textContent += '\nmeasuring…';
  const r = await measureLatency(nodeA.port);
  spike.a.latency = r;
  $('resultsA').textContent +=
    `\npostMessage round trip (ms): median ${r.roundTripMs.median}` +
    ` p95 ${r.roundTripMs.p95} max ${r.roundTripMs.max} min ${r.roundTripMs.min}` +
    ` (n=${r.roundTripMs.n}, lost ${r.lost})`;
  $('latency').disabled = false;
};

$('bench').onclick = () => {
  const quanta = 20000;
  const t0 = performance.now();
  const done = (ev) => {
    if (ev.data.t !== 'benchDone') return;
    nodeA.port.removeEventListener('message', done);
    const wall = performance.now() - t0;
    // Wall time here includes the message round trip; the audio-clock delta
    // the processor reports is the honest in-worklet figure. Report both.
    spike.a.bench = {
      quanta,
      mainThreadWallMs: +wall.toFixed(2),
      perQuantumUsWall: +((wall * 1000) / quanta).toFixed(3),
      audioClockDeltaS: ev.data.audioClockDelta,
    };
    $('resultsA').textContent +=
      `\nbench: ${quanta} quanta, ${wall.toFixed(1)} ms wall (incl. round trip)` +
      ` -> ${((wall * 1000) / quanta).toFixed(2)} us/quantum` +
      `  [realtime budget 2666 us/quantum @48k]`;
  };
  nodeA.port.addEventListener('message', done);
  nodeA.port.start?.();
  nodeA.port.postMessage({ t: 'bench', quanta });
};

$('hash').onclick = () => nodeA.port.postMessage({ t: 'hash' });

// -------------------------------------------------------------------- path B

let ctxB = null, nodeB = null, workerB = null;
// Ring depth IS path B's added latency: a sample written now is read
// (writeCursor - readCursor) frames later. Overridable as ?ring=N to find the
// point where the producer can no longer keep the ring fed.
const RING_FRAMES = +(new URLSearchParams(location.search).get('ring')) || 8192;

async function startB() {
  if (!spike.env.sharedArrayBuffer) {
    throw new Error('SharedArrayBuffer unavailable — page is not cross-origin isolated. ' +
                    'Reload with ?coi=1 (service-worker shim) or serve real COOP/COEP headers.');
  }
  const module = await getModule();
  ctxB = new AudioContext({ latencyHint: 'interactive' });
  await ctxB.resume();
  await ctxB.audioWorklet.addModule('worklet-ring.js');

  const ctrl = new SharedArrayBuffer(4 * 4);
  const data = new SharedArrayBuffer(RING_FRAMES * 2 * 4);

  workerB = new Worker('ring-worker.js');
  workerB.postMessage({ t: 'init', module, ctrl, data,
                        capacity: RING_FRAMES, sampleRate: ctxB.sampleRate });

  nodeB = new AudioWorkletNode(ctxB, 'runt-ring', {
    numberOfInputs: 0,
    outputChannelCount: [2],
    processorOptions: { ctrl, data, capacity: RING_FRAMES },
  });
  nodeB.onprocessorerror = (e) => {
    spike.b.errors.push('processorerror: ' + (e.message || 'AudioWorkletProcessor threw'));
  };
  nodeB.port.onmessage = (ev) => {
    if (ev.data.t !== 'stat') return;
    spike.b.stat = ev.data;
    $('statB').textContent =
      `rms       ${ev.data.rms.toFixed(5)}\n` +
      `quanta    ${ev.data.quanta}\n` +
      `underruns ${ev.data.underruns} (ring empty when audio needed frames)\n` +
      `glitches  ${ev.data.glitches}\n` +
      `ring depth ${ev.data.depth} frames (${(ev.data.depth / ctxB.sampleRate * 1000).toFixed(1)} ms)`;
  };
  nodeB.connect(ctxB.destination);

  spike.b.started = true;
  $('startB').disabled = true;
  $('stopB').disabled = false;
  $('latencyB').disabled = false;
  $('resultsB').textContent =
    `ring ${RING_FRAMES} frames = ${(RING_FRAMES / ctxB.sampleRate * 1000).toFixed(0)} ms of buffered audio`;
}

function stopB() {
  workerB?.postMessage({ t: 'stop' });
  workerB?.terminate();
  nodeB?.disconnect();
  ctxB?.close();
  ctxB = nodeB = workerB = null;
  spike.b.started = false;
  $('startB').disabled = false;
  $('stopB').disabled = true;
  $('latencyB').disabled = true;
}

$('startB').onclick = () => startB().catch((e) => {
  spike.b.errors.push(String(e));
  $('statB').innerHTML = `<span class="bad">${e}</span>`;
});
$('stopB').onclick = stopB;
$('latencyB').onclick = async () => {
  $('latencyB').disabled = true;
  const r = await measureLatency(nodeB.port);
  spike.b.latency = r;
  $('resultsB').textContent +=
    `\npostMessage round trip to ring worklet (ms): median ${r.roundTripMs.median}` +
    ` p95 ${r.roundTripMs.p95} max ${r.roundTripMs.max}` +
    `\nNOTE: audible latency additionally includes the ring depth above.`;
  $('latencyB').disabled = false;
};
