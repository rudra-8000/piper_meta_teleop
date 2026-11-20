// Runs web/quest.js against a fake WebXR/WebGL/WebSocket so the real XR code path executes without a headset.
// usage: node xr_harness.js <quest.js> <scenario>   scenario: ar | vr | hang_rate | no_frames | frame_error
const fs = require('fs'), vm = require('vm');
const [, , file, scenario = 'ar'] = process.argv;
const out = { reports: [], binary: [], gl: [], raf: 0, errors: [], texts: {} };

class El { constructor() { this.textContent = ''; this.className = ''; this.checked = false; this.disabled = false; this.onclick = null; this.style = {}; } }
const els = {}; const $ = (id) => els[id] || (els[id] = new El());
$('passthrough').checked = scenario !== 'vr'; $('headless').checked = false;

let rafCb = null; const sockets = [];
class WS {
  constructor(url) { this.url = url; this.readyState = 1; this.bufferedAmount = 0; sockets.push(this); setTimeout(() => this.onopen && this.onopen(), 0); }
  send(d) { if (typeof d === 'string') out.reports.push(JSON.parse(d)); else out.binary.push(Buffer.from(d.buffer, d.byteOffset, d.byteLength).toString('hex')); }
  close() {}
}
const gl = new Proxy({ FRAMEBUFFER: 1, COLOR_BUFFER_BIT: 16384, DEPTH_BUFFER_BIT: 256, SCISSOR_TEST: 3089, makeXRCompatible: async () => {} }, {
  get: (t, k) => (k in t ? t[k] : (...a) => { out.gl.push([String(k), ...a]); }) });
class XRWebGLLayer { constructor() { this.framebuffer = {}; } getViewport() { return { x: 0, y: 0, width: 1000, height: 1000 }; } }

const src = (hand, sq, trig) => ({ handedness: hand, gripSpace: { n: hand + 'g' }, targetRaySpace: { n: hand + 'r' },
  profiles: ['oculus-touch-v3'], gamepad: { buttons: [{ value: trig, pressed: trig > .5 }, { value: sq, pressed: sq > .5 }, { value: 0 }, { value: 0, pressed: false }, { value: 0, pressed: false }, { value: 0, pressed: false }], axes: [0, 0, 0.1, -0.2], hapticActuators: [{ pulse() {} }] } });
const session = {
  inputSources: [src('left', 1.0, 0.25), src('right', 0.0, 1.0)], visibilityState: 'visible', frameRate: 90, supportedFrameRates: [72, 90, 120],
  environmentBlendMode: scenario === 'vr' ? 'opaque' : 'alpha-blend', renderState: {},
  updateRenderState(s) { this.renderState = { ...s }; }, requestReferenceSpace: async (t) => ({ type: t }),
  requestAnimationFrame(cb) { if (scenario !== 'no_frames') rafCb = cb; out.raf++; return 1; }, addEventListener() {}, end: async () => {},
  // the suspected Quest behaviour: this promise never resolves (until frames flow, which would be never if we awaited it)
  updateTargetFrameRate: () => new Promise(() => {}),
};
const frame = { session,
  getPose: (sp) => { if (scenario === 'frame_error') throw new Error('boom'); return { transform: { position: { x: sp.n[0] === 'l' ? -0.2 : 0.2, y: 1.1, z: -0.3 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }, emulatedPosition: false }; },
  getViewerPose: () => ({ views: [{}, {}] }) };

const ctx = vm.createContext({
  window: { addEventListener() {} }, document: { getElementById: $, createElement: () => ({ getContext: () => gl }) },
  navigator: { userAgent: 'harness', xr: { isSessionSupported: async () => true, requestSession: async () => session } },
  location: { search: '', host: 'localhost:8788', protocol: 'http:' }, WebSocket: WS, XRWebGLLayer, performance, URLSearchParams,
  setInterval, setTimeout, clearTimeout, console, DataView, ArrayBuffer, Uint8Array, JSON, Math, Array, Object, Set, Promise, Error });
vm.runInContext(fs.readFileSync(file, 'utf8'), ctx);

(async () => {
  await new Promise((r) => setTimeout(r, 30));
  const started = $('start').onclick();               // the real click handler
  await Promise.race([started, new Promise((r) => setTimeout(r, 300))]);
  for (let i = 0; i < 3 && rafCb; i++) rafCb(i * 11, frame);
  if (scenario === 'no_frames') await new Promise((r) => setTimeout(r, 5300));
  await new Promise((r) => setTimeout(r, 1200));      // let the 1 Hz reporter fire
  out.texts = { hands: $('hands').textContent, log: $('log').textContent, start: $('start').textContent };
  console.log(JSON.stringify(out));
  process.exit(0);
})().catch((e) => { out.errors.push(String(e && e.stack || e)); console.log(JSON.stringify(out)); process.exit(1); });
