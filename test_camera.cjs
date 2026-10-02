const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(process.env.CAMERA_TEST_HTML || `${__dirname}/interface.html`, 'utf8');
const source = html.slice(html.indexOf('async function iniciarCamera('), html.indexOf("document.getElementById('btn-start').addEventListener"));

function camera({ serverMode, denied = false }) {
  const elements = new Map();
  const calls = { play: 0, inference: 0, stop: 0 };
  for (const id of ['video', 'stream-img', 'btn-modo-cam', 'cam-off', 'btn-reset-cam']) {
    elements.set(id, {
      style: {}, classList: { add() {}, remove() {} },
      removeAttribute(name) { delete this[name]; },
      addEventListener() {},
      async play() { calls.play++; },
    });
  }
  elements.get('video').style.display = 'none';
  elements.get('stream-img').style.display = 'block';
  elements.get('stream-img').src = '/stream';
  const track = { stop() { calls.stop++; }, getSettings() { return { deviceId: 'webcam' }; } };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  const context = vm.createContext({
    document: { getElementById: id => elements.get(id) },
    navigator: { mediaDevices: { async getUserMedia() {
      if (denied) throw Object.assign(new Error('Permissão negada'), { name: 'NotAllowedError' });
      return stream;
    } } },
    localStorage: { setItem() {}, removeItem() {} },
    modoServidor: serverMode, camerasConhecidas: [{}], cameraAtualId: 'webcam', streamAtivo: stream,
    statusCamera() {}, selectsCamera: () => [], listarCameras: async () => [],
    sincronizarCanvas() {}, pararEnvioFrames() {},
    iniciarEnvioFrames() { calls.inference++; }, textoErroCamera: e => e.message, alert() {},
  });
  vm.runInContext(source, context);
  return { context, elements, calls, stream };
}

for (const serverMode of [true, false]) {
  test(`webcam fica visível ao abrir pelo seletor/reset (modo servidor: ${serverMode})`, async () => {
    const { context, elements, calls, stream } = camera({ serverMode });
    await context.iniciarCamera();
    assert.equal(context.modoServidor, false);
    assert.equal(elements.get('video').style.display, 'block');
    assert.equal(elements.get('video').srcObject, stream);
    assert.equal(elements.get('stream-img').style.display, 'none');
    assert.equal(elements.get('stream-img').src, undefined);
    assert.equal(elements.get('cam-off').style.display, 'none');
    assert.equal(calls.play, 1);
    assert.equal(calls.inference, 1);
    assert.equal(calls.stop, 1);
  });
}

test('permissão negada mantém a orientação visível sem iniciar inferência', async () => {
  const { context, elements, calls } = camera({ serverMode: true, denied: true });
  await context.iniciarCamera();
  assert.equal(elements.get('cam-off').style.display, 'flex');
  assert.equal(calls.inference, 0);
  assert.equal(calls.play, 0);
});
