// Dependency-free DOM-contract tests. These are not rendered-browser QA.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('inspector/inspector.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
class Node {
  constructor(tag, text = '') { this.tag = tag; this.text = text; this.children = []; this.value = ''; this.listeners = {}; this.checked = false; this.disabled = false; this.classList = {add() {}}; }
  set textContent(text) { this.text = String(text); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.text = ''; this.children = nodes; if (this.tag === 'select') this.value = String(nodes[0]?.value ?? ''); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  async dispatch(name) { await this.listeners[name](); await settle(); }
}
async function app({demo = false, haveHidden = true, failLoad = false} = {}) {
  const ids = ['cards','left','right','metrics','details','notice','word','empty','compatible','state-badge'];
  const nodes = Object.fromEntries(ids.map(id => [id, new Node(['left','right'].includes(id) ? 'select' : 'div')]));
  const records = ['literal','metaphorical','mythical'].map((tag, inspector_id) => ({inspector_id, level: 'token', decoded: 'moon', normalized: 'moon', generated_text: 'The moon rose.', tags: [tag], note: '<script>not executable</script>', prompt: 'Look up.'}));
  const requests = [];
  const hidden = haveHidden ? [{index: 0, norm: 1, dimensions: 2}] : [];
  const context = {document: {getElementById: id => nodes[id], createElement: tag => new Node(tag), createTextNode: text => new Node('text', text)},
    Option: class extends Node { constructor(text, value) { super('option', text); this.value = String(value); } },
    fetch: async path => {requests.push(path); if (failLoad) throw new Error('offline'); return {ok: true, json: async () => path === '/api/annotations' ? {records, demo} : path.startsWith('/api/trace') ? {hidden, attention: [], message: 'Saved test fixture'} : {layers: [{index: 0, cosine: 0, distance: 1.4142, norm_a: 1, norm_b: 1}]}};}
  };
  vm.runInNewContext(source, context);
  await settle(); await settle();
  return {nodes, requests};
}
test('demo explains absence of metrics; filter empties and restores cards', async () => {
  const {nodes, requests} = await app({demo: true, haveHidden: false});
  assert.equal(nodes.cards.children.length, 3);
  assert.match(nodes.notice.textContent, /手書き/);
  assert.match(nodes.metrics.textContent, /まだ数値はありません/);
  assert.equal(nodes.compatible.disabled, true);
  nodes.word.value = 'unknown'; await nodes.word.dispatch('input');
  assert.equal(nodes.cards.children.length, 0);
  assert.equal(nodes.empty.hidden, false);
  assert.equal(nodes.left.disabled, true);
  nodes.word.value = 'MOON'; await nodes.word.dispatch('input');
  assert.equal(nodes.cards.children.length, 3);
  assert.equal(nodes.left.disabled, false);
  assert.equal(requests.some(path => path.startsWith('/api/compare')), false);
  assert.match(nodes.cards.textContent, /<script>not executable<\/script>/);
});
test('metrics require confirmation, switching resets it, same-record compare is blocked', async () => {
  const {nodes, requests} = await app();
  assert.match(nodes.metrics.textContent, /採取条件/);
  assert.equal(requests.some(path => path.startsWith('/api/compare')), false);
  nodes.compatible.checked = true; await nodes.compatible.dispatch('change');
  assert.match(nodes.metrics.textContent, /1.4142/);
  assert.equal(requests.filter(path => path.startsWith('/api/compare')).length, 1);
  nodes.right.value = '2'; await nodes.right.dispatch('change');
  assert.equal(nodes.compatible.checked, false);
  assert.match(nodes.metrics.textContent, /採取条件/);
  nodes.right.value = nodes.left.value; await nodes.right.dispatch('change');
  assert.equal(nodes.compatible.disabled, true);
  assert.match(nodes.metrics.textContent, /別の注釈/);
});
test('missing hidden states leave annotations usable', async () => {
  const {nodes} = await app({haveHidden: false});
  assert.equal(nodes.cards.children.length, 3);
  assert.match(nodes.metrics.textContent, /Hidden states がありません/);
});
test('load failure gives visible error', async () => {
  const {nodes} = await app({failLoad: true});
  assert.match(nodes.notice.textContent, /読み込みに失敗しました: offline/);
});
