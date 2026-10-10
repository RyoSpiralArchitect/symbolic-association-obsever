// Dependency-free DOM-contract tests. These are not rendered-browser QA.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('inspector/inspector.js', 'utf8');
const html = fs.readFileSync('inspector/index.html', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
const sampleRecords = () => ['literal','metaphorical','mythical'].map((tag, inspector_id) => ({inspector_id, level: 'token', decoded: 'moon', normalized: 'moon', generated_text: `The ${tag} moon rose.`, tags: [tag], note: `${tag}: <script>not executable</script>`, prompt: 'Look up.'}));
const descendants = node => [node, ...node.children.flatMap(descendants)];
class Node {
  constructor(tag, text = '') { this.tag = tag; this.text = text; this.children = []; this.value = ''; this.listeners = {}; this.checked = false; this.disabled = false; this.classList = {add() {}}; }
  set textContent(text) { this.text = String(text); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.text = ''; this.children = nodes; if (this.tag === 'select') this.value = String(nodes[0]?.value ?? ''); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  async dispatch(name) { await this.listeners[name](); await settle(); }
}
async function app({demo = false, haveHidden = true, failLoad = false, failTrace = false, records = sampleRecords(), traceResponse} = {}) {
  const ids = ['cards','left','right','selected-contexts','metrics','details','notice','word','empty','compatible','state-badge'];
  const nodes = Object.fromEntries(ids.map(id => [id, new Node(['left','right'].includes(id) ? 'select' : 'div')]));
  const requests = [];
  const hidden = haveHidden ? [{index: 0, norm: 1, dimensions: 2}] : [];
  const context = {document: {getElementById: id => nodes[id], createElement: tag => new Node(tag), createTextNode: text => new Node('text', text)},
    Option: class extends Node { constructor(text, value) { super('option', text); this.value = String(value); } },
    fetch: async path => {
      requests.push(path);
      if (failLoad) throw new Error('offline');
      if (failTrace && path.startsWith('/api/trace')) throw new Error('state unavailable');
      return {ok: true, json: async () => path === '/api/annotations' ? {records, demo} : path.startsWith('/api/trace') ? (traceResponse ? traceResponse(path) : {hidden, attention: [], message: 'Saved test fixture'}) : {layers: [{index: 0, cosine: 0, distance: 1.4142, norm_a: 1, norm_b: 1}]}};
    }
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
test('selected A/B passages, tags and notes are visible without tensors or confirmation', async () => {
  const {nodes, requests} = await app({demo: true, haveHidden: false});
  const panels = nodes['selected-contexts'];
  assert.equal(panels.children.length, 2);
  assert.match(panels.children[0].textContent, /A \/ 選択した文脈.*literal.*The literal moon rose\./);
  assert.match(panels.children[1].textContent, /B \/ 選択した文脈.*metaphorical.*The metaphorical moon rose\./);
  assert.match(panels.children[0].textContent, /literal: <script>not executable<\/script>/);
  assert.match(panels.children[1].textContent, /TEXT FIXTURE · NO TOKEN IDS \/ TENSORS/);
  assert.equal(descendants(panels).some(node => node.tag === 'script'), false);
  assert.equal(nodes.compatible.checked, false);
  assert.equal(requests.some(path => path.startsWith('/api/compare')), false);
  nodes.right.value = '2'; await nodes.right.dispatch('change');
  assert.match(panels.children[1].textContent, /B \/ 選択した文脈.*mythical.*The mythical moon rose\./);
  assert.match(panels.children[0].textContent, /The literal moon rose\./);
  nodes.left.value = '1'; await nodes.left.dispatch('change');
  assert.match(panels.children[0].textContent, /A \/ 選択した文脈.*The metaphorical moon rose\./);
  assert.match(panels.children[1].textContent, /The mythical moon rose\./);
});
test('filtering preserves valid selections, clears stale panels, and resetting restores them', async () => {
  const records = sampleRecords();
  records[2].decoded = records[2].normalized = 'sun';
  const {nodes} = await app({records});
  const panels = nodes['selected-contexts'];
  nodes.left.value = '1'; await nodes.left.dispatch('change');
  nodes.right.value = '2'; await nodes.right.dispatch('change');
  nodes.compatible.checked = true; await nodes.compatible.dispatch('change');
  nodes.word.value = 'moon'; await nodes.word.dispatch('input');
  assert.equal(nodes.compatible.checked, false);
  assert.equal(String(nodes.left.value), '1');
  assert.equal(String(nodes.right.value), '1');
  assert.match(panels.children[0].textContent, /The metaphorical moon rose\./);
  assert.doesNotMatch(panels.textContent, /mythical/);
  nodes.word.value = 'unknown'; await nodes.word.dispatch('input');
  assert.equal(descendants(panels).filter(node => node.tag === 'article').length, 0);
  assert.match(panels.textContent, /比較する文脈がありません/);
  assert.equal(nodes.left.disabled, true);
  assert.equal(nodes.right.disabled, true);
  nodes.word.value = ''; await nodes.word.dispatch('input');
  assert.equal(panels.children.length, 2);
  assert.match(panels.children[0].textContent, /The literal moon rose\./);
  assert.match(panels.children[1].textContent, /The metaphorical moon rose\./);
  assert.equal(nodes.right.disabled, false);
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
  assert.match(nodes['selected-contexts'].textContent, /The literal moon rose\./);
  assert.match(nodes['selected-contexts'].textContent, /The metaphorical moon rose\./);
});
test('trace failures leave selected text usable and independent of comparison errors', async () => {
  const {nodes, requests} = await app({failTrace: true});
  assert.match(nodes.metrics.textContent, /比較できませんstate unavailable/);
  assert.match(nodes['selected-contexts'].textContent, /The literal moon rose\./);
  nodes.right.value = '2'; await nodes.right.dispatch('change');
  assert.match(nodes['selected-contexts'].children[1].textContent, /The mythical moon rose\./);
  assert.equal(nodes.compatible.disabled, true);
  assert.equal(requests.some(path => path.startsWith('/api/compare')), false);
});
test('selected panels update during slow tensor loading; stale responses cannot restore old selections', async () => {
  const pending = new Map();
  const {nodes} = await app({traceResponse: path => new Promise(resolve => pending.set(path, resolve))});
  assert.match(nodes.metrics.textContent, /Loading/);
  assert.match(nodes['selected-contexts'].children[1].textContent, /The metaphorical moon rose\./);
  nodes.right.value = '2'; await nodes.right.dispatch('change');
  assert.match(nodes['selected-contexts'].children[1].textContent, /The mythical moon rose\./);
  nodes.word.value = 'unknown'; await nodes.word.dispatch('input');
  for (const resolve of pending.values()) resolve({hidden: [], attention: [], message: 'Late trace'});
  await settle(); await settle();
  assert.match(nodes['selected-contexts'].textContent, /比較する文脈がありません/);
  assert.equal(nodes.details.children.length, 0);
  assert.match(nodes.metrics.textContent, /注釈を選んでください/);
});
test('missing fields are explicit; segment text remains readable with no generated passage', async () => {
  const records = [{inspector_id: 0, level: 'token'}, {inspector_id: 1, level: 'segment', segment_text: 'A moon\nover water.', note: 'Line one\nLine two'}];
  const {nodes} = await app({records, haveHidden: false});
  const [a, b] = nodes['selected-contexts'].children;
  assert.match(a.textContent, /タグはありません.*本文は記録されていません.*メモはありません/);
  assert.match(b.textContent, /A moon\nover water\./);
  assert.match(b.textContent, /Line one\nLine two/);
  assert.equal(descendants(b).filter(node => node.tag === 'mark').length, 1);
  assert.doesNotMatch(a.textContent + b.textContent, /TEXT FIXTURE/);
});
test('empty and single-record datasets give honest selected-context states', async () => {
  const empty = await app({records: []});
  assert.match(empty.nodes['selected-contexts'].textContent, /比較する文脈がありません/);
  assert.equal(empty.requests.filter(path => path.startsWith('/api/trace')).length, 0);
  const single = await app({records: sampleRecords().slice(0, 1)});
  assert.equal(single.nodes['selected-contexts'].children.length, 2);
  assert.match(single.nodes['selected-contexts'].children[1].textContent, /B \/ 選択した文脈.*The literal moon rose\./);
  assert.match(single.nodes.metrics.textContent, /別の注釈を選んでください/);
  assert.equal(single.nodes.compatible.disabled, true);
});
test('selected-context section precedes the tensor-only section in the document', () => {
  const selected = html.indexOf('id="selected-contexts"');
  const tensors = html.indexOf('<section class="comparison"');
  assert.ok(selected > 0 && selected < tensors);
  assert.ok(html.indexOf('id="left"') < selected && html.indexOf('id="right"') < selected);
});
test('load failure gives visible error', async () => {
  const {nodes} = await app({failLoad: true});
  assert.match(nodes.notice.textContent, /読み込みに失敗しました: offline/);
});
