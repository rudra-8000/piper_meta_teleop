// Quest 3 WebXR client: streams both controllers' poses to the PC at display rate.
// Binary layout must match piper_teleop/protocol.py.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const wsUrl = params.get('ws') ||
    `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  const FAKE = params.has('fake');           // desktop test: synthetic controllers, no XR

  const HEADER = 12, HAND = 62;
  const buf = new ArrayBuffer(HEADER + 2 * HAND);
  const dv = new DataView(buf);
  let seq = 0;
  let ws = null, wsOpen = false, rtt = null, serverState = {}, session = null, gl = null;
  let headless = false;

  const log = (m) => { $('log').textContent = m; };

  // What this page tells the PC dashboard about itself (errors inside VR are otherwise invisible).
  let serverCalib = null;        // calibration progress from the PC (drawn as three squares in the view)
  function speak(text) {         // best effort: spoken prompts inside the headset
    try {
      $('log').textContent = text;
      if (window.speechSynthesis && window.SpeechSynthesisUtterance) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new window.SpeechSynthesisUtterance(text));
      }
    } catch (e) {}
  }
  let lastReport = { state: 'loading', msg: '' };
  let trackedNow = 'none';
  function report(state, msg, extra) {
    lastReport = Object.assign({}, lastReport, { state, msg: msg || '' }, extra || {});
    if (wsOpen) ws.send(JSON.stringify(Object.assign({ t: 'xr' }, lastReport)));
  }
  function inputList() {
    if (!session) return [];
    return Array.from(session.inputSources).map((s) => ({
      h: s.handedness, p: (s.profiles || []).slice(0, 2).join(','), g: !!s.gamepad, hand: !!s.hand }));
  }
  window.addEventListener('error', (e) => report('error', 'page error: ' + e.message));
  window.addEventListener('unhandledrejection', (e) => report('error', 'page error: ' + (e.reason && e.reason.message || e.reason)));

  // ---------------------------------------------------------------- websocket
  function connect() {
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      wsOpen = true;
      ws.send(JSON.stringify({ t: 'hello', role: 'controller', ua: navigator.userAgent, mode: headless ? 'headless' : 'headset' }));
      $('net').textContent = 'connected';
      $('net').className = 'ok';
      ws.send(JSON.stringify(Object.assign({ t: 'xr' }, lastReport)));   // dashboard sees our state right away
    };
    ws.onclose = () => {
      wsOpen = false;
      $('net').textContent = 'disconnected - retrying';
      $('net').className = 'bad';
      setTimeout(connect, 1000);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return;
      const m = JSON.parse(ev.data);
      if (m.t === 'pong') {
        rtt = performance.now() - m.c;
        $('rtt').textContent = rtt.toFixed(1) + ' ms';
        ws.send(JSON.stringify({ t: 'rtt', ms: rtt }));
      } else if (m.t === 'haptic') {
        pulse(m.hand, m.i, m.ms);
      } else if (m.t === 'speak') {
        speak(m.text);
      } else if (m.t === 'status') {
        serverCalib = m.calib || null;
        serverState = m.arms || {};
        const parts = Object.entries(serverState).map(([k, v]) => `${k}: ${v.state}${v.fine ? ' (fine)' : ''} - ${v.msg}`);
        $('arms').textContent = parts.join('\n');
      }
    };
  }
  setInterval(() => { if (wsOpen) ws.send(JSON.stringify({ t: 'ping', c: performance.now() })); }, 500);

  // ------------------------------------------------------------------- haptics
  function pulse(hand, intensity, ms) {
    if (!session) return;
    for (const src of session.inputSources) {
      if (src.handedness === hand && src.gamepad && src.gamepad.hapticActuators &&
          src.gamepad.hapticActuators[0]) {
        try { src.gamepad.hapticActuators[0].pulse(intensity, ms); } catch (e) {}
      }
    }
  }

  // ---------------------------------------------------------------- packet pack
  function packHand(o, id, flags, p, q, rq, trig, sq, sx, sy) {
    dv.setUint8(o, id); dv.setUint8(o + 1, flags); o += 2;
    for (const v of [p.x, p.y, p.z, q.x, q.y, q.z, q.w, rq.x, rq.y, rq.z, rq.w, trig, sq, sx, sy]) {
      dv.setFloat32(o, v, true); o += 4;
    }
    return o;
  }
  function begin(n) {
    dv.setUint8(0, 1); dv.setUint8(1, n); dv.setUint16(2, seq++ & 0xffff, true);
    dv.setFloat64(4, performance.now(), true);
    return HEADER;
  }
  function send(n) {
    // never queue stale poses: if the socket is backed up, drop this frame
    if (wsOpen && ws.bufferedAmount < 1024) { ws.send(new Uint8Array(buf, 0, HEADER + n * HAND)); sentPackets++; }
  }

  // ------------------------------------------------------------------- XR loop
  let frames = 0, sentPackets = 0, lastFramesSeen = 0, lastSentSeen = 0;
  const handsNow = {};           // 'left'/'right' -> { sq, trig } as read in the current frame
  const errorsSeen = new Set();
  function fail(where, e) {      // report each distinct problem once, to the PC dashboard
    const msg = `${where}: ${e && e.message ? e.message : e}`;
    if (errorsSeen.has(msg)) return;
    errorsSeen.add(msg);
    log(msg);
    report('error', msg);
  }

  // Head-locked status squares (left arm | right arm) drawn with scissored clears: no shaders needed, so a
  // "black screen" can never hide whether the session is alive.
  const COL = { none: [0.45, 0.45, 0.45], idle: [1.0, 0.75, 0.1], engaged: [0.1, 0.9, 0.3], estop: [1.0, 0.1, 0.1], busy: [0.2, 0.5, 1.0] };
  function armColor(name) {
    if (!handsNow[name]) return COL.none;                    // this hand is not tracked
    const st = (serverState[name] || {}).state;
    if (st === 'engaged') return COL.engaged;
    if (st === 'estop') return COL.estop;
    if (st === 'homing' || st === 'parking') return COL.busy;
    return COL.idle;                                         // tracked, arm waiting for GRIP
  }
  function drawHud(vp) {
    gl.enable(gl.SCISSOR_TEST);
    const sz = Math.max(8, Math.round(vp.height * 0.06));
    const y = vp.y + Math.round(vp.height * 0.3);
    const cx = vp.x + Math.round(vp.width / 2);
    for (const [name, x] of [['left', cx - sz - Math.round(sz * 0.4)], ['right', cx + Math.round(sz * 0.4)]]) {
      const c = armColor(name);
      gl.scissor(x, y, sz, sz); gl.clearColor(c[0], c[1], c[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
      if (handsNow[name] && handsNow[name].sq > 0.6) {       // white dot: the browser itself sees GRIP pressed
        const d = Math.round(sz * 0.4);
        gl.scissor(x + Math.round((sz - d) / 2), y + Math.round((sz - d) / 2), d, d);
        gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
    if (serverCalib && serverCalib.active) {                 // calibration progress: three squares above the arm squares
      const y2 = y + Math.round(sz * 1.8), small = Math.round(sz * 0.7);
      for (let i = 0; i < serverCalib.of; i++) {
        const c = i < serverCalib.step ? [0.1, 0.9, 0.3] : i === serverCalib.step ? (serverCalib.phase === 'record' ? [1.0, 0.75, 0.1] : [0.2, 0.5, 1.0]) : [0.45, 0.45, 0.45];
        const x = cx + Math.round((i - (serverCalib.of - 1) / 2) * small * 1.5) - Math.round(small / 2);
        gl.scissor(x, y2, small, small); gl.clearColor(c[0], c[1], c[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
    gl.disable(gl.SCISSOR_TEST);
  }

  function onXRFrame(t, frame) {
    frame.session.requestAnimationFrame(onXRFrame);          // keep the loop alive whatever happens below
    frames++;
    try { renderAndSend(frame, frame.session); } catch (e) { fail('frame loop', e); }
  }

  function renderAndSend(frame, s) {
    const opaque = s.environmentBlendMode === 'opaque';
    for (const k of Object.keys(handsNow)) delete handsNow[k];
    // 1) read both controllers
    let n = 0, o = HEADER;
    const names = [];
    for (const src of s.inputSources) {
      if (!src.gamepad || (src.handedness !== 'left' && src.handedness !== 'right') || !src.gripSpace) continue;
      const gp = frame.getPose(src.gripSpace, refSpace);
      const rp = frame.getPose(src.targetRaySpace, refSpace);
      if (!gp || !rp) continue;
      if (n === 0) o = begin(2);
      const b = src.gamepad.buttons, ax = src.gamepad.axes;
      const val = (i) => (b[i] ? b[i].value : 0);
      const prs = (i) => (b[i] ? b[i].pressed : false);
      const flags = 1 | (gp.emulatedPosition ? 2 : 0) | (prs(4) ? 4 : 0) | (prs(5) ? 8 : 0) | (prs(3) ? 16 : 0);
      const stickX = ax.length > 2 ? ax[2] : 0, stickY = ax.length > 3 ? ax[3] : 0;
      o = packHand(o, src.handedness === 'left' ? 0 : 1, flags,
        gp.transform.position, gp.transform.orientation, rp.transform.orientation,
        val(0), val(1), stickX, stickY);
      handsNow[src.handedness] = { sq: val(1), trig: val(0) };
      n++;
      names.push(src.handedness);
    }
    // 2) draw: passthrough stays fully transparent; VR gets a dark background that doubles as status
    const layer = s.renderState.baseLayer;
    if (gl && layer) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      const states = Object.values(serverState).map((v) => v.state);
      let c = [0, 0, 0, 0];
      if (opaque) c = headless ? [0, 0, 0, 1] : states.includes('estop') ? [0.45, 0, 0, 1]
        : states.includes('engaged') ? [0, 0.16, 0.05, 1] : [0.02, 0.02, 0.07, 1];
      gl.clearColor(c[0], c[1], c[2], c[3]);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!headless && refSpace) {
        const vp = frame.getViewerPose(refSpace);
        if (vp) for (const v of vp.views) drawHud(layer.getViewport(v));
      }
    }
    // 3) send
    if (n > 0) {
      dv.setUint8(1, n);
      send(n);
      $('hands').textContent = names.join('+');
      trackedNow = names.join('+');
    } else {
      $('hands').textContent = 'none tracked';
      trackedNow = 'none';
    }
  }

  let refSpace = null;
  async function startXR() {
    headless = $('headless').checked;
    lastReport.mode = headless ? 'headless' : 'headset';        // the server arms the proximity override for headless sessions
    const wantAR = $('passthrough').checked;
    let mode = 'immersive-vr';
    try { if (wantAR && (await navigator.xr.isSessionSupported('immersive-ar'))) mode = 'immersive-ar'; } catch (e) {}
    report('starting', mode + ' - requesting session');
    try {
      session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor'] });
    } catch (e) {
      log('Could not start XR session: ' + e.message);
      report('error', 'requestSession(' + mode + ') failed: ' + e.message);
      return;
    }
    frames = 0; lastFramesSeen = 0; lastSentSeen = 0; sentPackets = 0;
    setTimeout(() => {       // armed BEFORE any further setup, so a hang anywhere below is still reported
      if (session && frames === 0) fail('watchdog', `a VR session exists but the browser has not called the frame callback in 5 s (visibility=${session.visibilityState}). Black screen with no controller data.`);
    }, 5000);
    try {
      const canvas = document.createElement('canvas');
      gl = canvas.getContext('webgl2', { xrCompatible: true, antialias: false, depth: false, alpha: true });
      if (!gl) throw new Error('could not create a WebGL2 context');
      await gl.makeXRCompatible();
      session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
      try { refSpace = await session.requestReferenceSpace('local-floor'); }
      catch (e) { refSpace = await session.requestReferenceSpace('local'); }
    } catch (e) {
      fail('XR setup', e);
      try { await session.end(); } catch (_) {}
      return;
    }
    session.addEventListener('end', () => { session = null; trackedNow = 'none'; log('XR session ended'); report('ended', 'XR session ended'); });
    session.addEventListener('inputsourceschange', () => report('running', lastReport.msg, { inputs: inputList() }));
    session.addEventListener('visibilitychange', () => report('running', 'visibility: ' + session.visibilityState, { vis: session.visibilityState }));
    // FRAMES FIRST: nothing below may be awaited before the frame loop runs (some browsers only resolve
    // updateTargetFrameRate once frames are flowing - awaiting it here would mean a black screen and no data).
    session.requestAnimationFrame(onXRFrame);
    report('running', `${mode}, ${session.visibilityState}`, { inputs: inputList(), blend: session.environmentBlendMode });
    log(`XR ${mode} started`);
    try {
      const rates = session.supportedFrameRates;
      if (rates && session.updateTargetFrameRate) {
        session.updateTargetFrameRate(Math.max(...Array.from(rates)))
          .then(() => { report('running', `${mode}, ${session.frameRate || '?'} Hz`, { inputs: inputList() }); })
          .catch((e) => fail('updateTargetFrameRate', e));
      }
    } catch (e) { fail('frame rate request', e); }
  }

  // ---------------------------------------------- desktop fake mode (no headset)
  function startFake() {
    log('FAKE mode: synthetic controllers (no XR)');
    report('running', 'FAKE synthetic controllers (desktop test)', { tracked: 'left+right', inputs: [] });
    $('hands').textContent = 'fake';
    const q = { x: 0, y: 0, z: 0, w: 1 };
    const t00 = performance.now() / 1000;
    setInterval(() => {
      const t = performance.now() / 1000;
      const grip = t - t00 > 1.5 ? 1.0 : 0.0;   // release first: the PC requires a fresh grab
      const o = begin(2);
      let off = o;
      for (const [id, ph] of [[0, 0], [1, 1.5]]) {
        const p = { x: (id ? 0.2 : -0.2) + 0.08 * Math.sin(t * 1.2 + ph), y: 1.1 + 0.05 * Math.sin(t * 0.9 + ph), z: -0.4 + 0.06 * Math.cos(t * 1.2 + ph) };
        off = packHand(off, id, 1, p, q, q, 0.5 + 0.5 * Math.sin(t + ph), grip, 0, 0);
      }
      send(2);
    }, 1000 / 90);
  }

  // ----------------------------------------------------------------------- UI
  async function init() {
    connect();
    if (FAKE) { startFake(); return; }
    if (!navigator.xr) {
      report('unavailable', 'WebXR not available (needs Quest Browser on http://localhost or https://)');
      $('xrstate').textContent = 'WebXR not available in this browser (use Quest Browser over http://localhost or https://)';
      $('xrstate').className = 'bad';
      return;
    }
    const vr = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
    const ar = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
    if (!(vr || ar)) report('unavailable', `this browser has WebXR but supports no immersive session (VR: ${vr}, AR: ${ar}) - use Quest Browser on http://localhost:PORT or https://`);
    else report('idle', `WebXR ok (VR: ${vr}, passthrough AR: ${ar}) - press "Enter VR & start streaming"`);
    $('xrstate').textContent = `WebXR ok  (VR: ${vr}, passthrough AR: ${ar})`;
    $('xrstate').className = vr || ar ? 'ok' : 'bad';
    $('passthrough').disabled = !ar;
    $('passthrough').checked = ar;
    $('start').disabled = !(vr || ar);
    $('start').onclick = startXR;
  }
  // once a second: what the headset sees (so "no controller data" on the PC can be explained)
  setInterval(() => {
    if (!session) return;
    const fps = frames - lastFramesSeen, sent = sentPackets - lastSentSeen;
    lastFramesSeen = frames; lastSentSeen = sentPackets;
    report('running', lastReport.msg, { inputs: inputList(), tracked: trackedNow, vis: session.visibilityState, fps, sent, blend: session.environmentBlendMode });
  }, 1000);
  $('estop').onclick = () => { if (wsOpen) ws.send(JSON.stringify({ t: 'estop' })); };
  init();
})();
