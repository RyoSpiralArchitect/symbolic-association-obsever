'use strict';
const $ = id => document.getElementById(id);
let dataset = {records: [], demo: false};
let visible = [];
let requestVersion = 0;
const traceCache = new Map();
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function label(record) {
  const target = record.level === 'token' ? record.decoded : record.segment_text;
  return `${record.inspector_id + 1}. ${(record.tags || []).join(' / ') || record.level} · ${target || 'annotation'}`;
}
function textWithTarget(text, target) {
  const paragraph = element('p', undefined, 'passage');
  // Literal substring marking is a reading aid, not reconstructed token alignment.
  const index = target ? text.toLocaleLowerCase().indexOf(target.toLocaleLowerCase()) : -1;
  if (index < 0) paragraph.textContent = text;
  else paragraph.append(document.createTextNode(text.slice(0, index)), element('mark', text.slice(index, index + target.length)), document.createTextNode(text.slice(index + target.length)));
  return paragraph;
}
function renderCards() {
  const query = $('word').value.trim().toLocaleLowerCase();
  visible = dataset.records.filter(record => [record.decoded, record.normalized, record.segment_text].some(value => (value || '').toLocaleLowerCase().includes(query)));
  $('cards').replaceChildren();
  for (const record of visible) {
    const kind = (record.tags || []).find(tag => ['literal','metaphorical','mythical','mythic'].includes(tag)) || '';
    const card = element('article', undefined, `card ${kind}`);
    const head = element('div', undefined, 'card-head');
    for (const tag of record.tags || []) head.append(element('span', tag, 'tag'));
    head.append(element('span', `#${record.inspector_id + 1} / ${record.level}`, 'card-id'));
    card.append(head, textWithTarget(record.generated_text || record.segment_text || '', (record.decoded || record.segment_text || '').trim()));
    card.append(element('p', record.note || 'メモはありません', 'note'));
    const details = element('details');
    details.append(element('summary', 'Prompt / 注釈対象'), element('p', record.prompt || 'Prompt not recorded'), element('p', `対象: ${record.decoded || record.segment_text || 'Not recorded'}`));
    card.append(details);
    card.append(element('p', dataset.demo ? 'TEXT FIXTURE · NO TOKEN IDS / TENSORS' : (record.level === 'token' ? `token id ${record.token_id ?? '?'} · generated index ${record.token_index_in_generation ?? '?'} · full position ${record.seq_pos_in_full ?? '?'}` : `segment ${record.segment_id ?? '?'} · ${(record.segment_token_indices || []).length} annotated tokens`), 'position'));
    $('cards').append(card);
  }
  $('empty').hidden = visible.length > 0;
  const previous = [$('left').value, $('right').value];
  for (const [position, id] of ['left', 'right'].entries()) {
    $(id).replaceChildren(...visible.map(record => new Option(label(record), record.inspector_id)));
    const preferred = visible.find(record => String(record.inspector_id) === previous[position]) || visible[Math.min(position, visible.length - 1)];
    if (preferred) $(id).value = preferred.inspector_id;
    $(id).disabled = visible.length === 0;
  }
  $('compatible').checked = false;
  refreshComparison();
}
function noData(title, message) {
  const box = element('div', undefined, 'no-data');
  box.append(element('h3', title), element('p', message));
  $('metrics').replaceChildren(box);
}
function trace(id) {
  if (!traceCache.has(id)) traceCache.set(id, api(`/api/trace?id=${id}`).catch(error => {traceCache.delete(id); throw error;}));
  return traceCache.get(id);
}
function detail(traceData, side) {
  const box = element('div', undefined, 'trace-detail');
  box.append(element('strong', `${side} · ${traceData.hidden.length} hidden / ${traceData.attention.length} attention slots`), element('p', traceData.message));
  if (traceData.attention.length) {
    const disclosure = element('details', undefined, 'attention');
    disclosure.append(element('summary', 'Attention: head 平均 / 上位5位置'));
    const list = element('ul');
    for (const slot of traceData.attention) list.append(element('li', slot.missing ? `Slot ${slot.slot}: missing` : `Slot ${slot.slot} (${slot.heads} heads): ${slot.top_positions.map(p => `pos ${p.position} → ${p.weight.toFixed(4)}`).join(' · ')}`));
    disclosure.append(list); box.append(disclosure);
  }
  return box;
}
async function refreshComparison() {
  const version = ++requestVersion;
  $('details').replaceChildren();
  $('state-badge').textContent = 'NO TENSORS';
  $('compatible').disabled = true;
  if (!visible.length) { noData('注釈を選んでください', '検索条件を変えると、比較対象を選べます。'); return; }
  noData('Loading…', '保存された注釈を読み込んでいます');
  try {
    const left = $('left').value, right = $('right').value;
    const [a, b] = await Promise.all([trace(left), trace(right)]);
    if (version !== requestVersion) return;
    $('details').append(detail(a, 'A'), detail(b, 'B'));
    const haveHidden = a.hidden.length > 0 && b.hidden.length > 0;
    $('state-badge').textContent = haveHidden ? 'SAVED HIDDEN STATES' : (a.attention.length || b.attention.length ? 'ATTENTION ONLY' : 'NO HIDDEN STATES');
    $('compatible').disabled = !haveHidden || left === right;
    if (dataset.demo) { noData('ここには、まだ数値はありません。', 'この3つの文章は手書きのデモです。実際の JSONL と .pt を開くと、層ごとの cosine similarity / L2 distance が表示されます。'); return; }
    if (!haveHidden) { noData('Hidden states がありません', '注釈は読むことができます。保存済みの hidden states が両方に揃うと比較できます。'); return; }
    if (left === right) { noData('別の注釈を選んでください', 'A と B に異なる記録を選びます。'); return; }
    if (!$('compatible').checked) { noData('採取条件を確認してください', '同じモデル・tokenizer・設定で採取したことを確認すると、数値を表示します。'); return; }
    const comparison = await api(`/api/compare?left=${left}&right=${right}`);
    if (version !== requestVersion) return;
    const table = element('table', undefined, 'metric-table');
    table.append(element('caption', 'Stored hidden index · cosine は −1〜1 / L2 は未正規化の距離'));
    const head = element('thead'); const headRow = element('tr');
    for (const name of ['Index', 'Cosine', 'L2 distance', 'Norm A', 'Norm B']) headRow.append(element('th', name));
    head.append(headRow); table.append(head);
    const body = element('tbody');
    for (const layer of comparison.layers) {
      const row = element('tr');
      for (const value of [layer.index, layer.cosine === null ? 'undefined (zero norm)' : layer.cosine.toFixed(4), layer.distance.toFixed(4), layer.norm_a.toFixed(4), layer.norm_b.toFixed(4)]) row.append(element('td', String(value)));
      body.append(row);
    }
    table.append(body);
    const wrap = element('div', undefined, 'table-wrap'); wrap.append(table);
    $('metrics').replaceChildren(wrap, element('p', '数値は表現空間内の幾何的な差です。高い類似度や小さい距離だけで、意味や象徴性が同じだとは言えません。', 'fine'));
  } catch (error) {
    if (version === requestVersion) noData('比較できません', error.message);
  }
}
$('word').addEventListener('input', renderCards);
for (const id of ['left', 'right']) $(id).addEventListener('change', () => { $('compatible').checked = false; refreshComparison(); });
$('compatible').addEventListener('change', refreshComparison);
api('/api/annotations').then(data => {
  dataset = data;
  $('notice').textContent = dataset.demo ? 'DEMO / 手書きの moon 文脈3つ。モデル未実行・tensor なし。実測値を模した数値は表示しません。' : `${dataset.records.length} annotations / 読み取り専用。元の JSONL と state ファイルは変更しません。`;
  renderCards();
}).catch(error => { $('notice').textContent = `読み込みに失敗しました: ${error.message}`; $('notice').classList.add('error'); });
