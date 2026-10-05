'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const gib = (b) => (b / 1073741824).toFixed(2) + ' GiB';
const mib = (b) => Math.round(b / 1048576) + ' MiB';
const num = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');

async function getJSON(url) {
  const r = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  return r.json();
}

/* ---------- Pestañas ---------- */
const tabButtons = Array.from(document.querySelectorAll('#tabs button'));
function selectTab(name) {
  document.querySelectorAll('.tab').forEach((s) => s.classList.toggle('active', s.id === 'tab-' + name));
  tabButtons.forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  if (name === 'estado') refreshEstado();
  if (name === 'bench') loadBench();
  if (name === 'agentes') loadAgents();
}
tabButtons.forEach((b) => b.addEventListener('click', () => selectTab(b.dataset.tab)));

/* ---------- Chat ---------- */
const CHAT_KEY = 'brasa.chat.v1';
let messages = [];
try { messages = JSON.parse(localStorage.getItem(CHAT_KEY) || '[]'); } catch (_) { messages = []; }
let controller = null;
let streaming = false;

function saveChat() {
  // localStorage puede no estar disponible (modo privado, cuota): no romper la UI.
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(messages)); } catch (_) { /* ignorar */ }
}

function renderThink(text) {
  const d = el('details', 'think');
  d.appendChild(el('summary', null, 'razonamiento'));
  d.appendChild(el('div', null, text));
  return d;
}

function renderChat() {
  const box = $('#messages');
  box.innerHTML = '';
  for (const m of messages) {
    if (m.role === 'error') { box.appendChild(el('div', 'msg error', m.content)); continue; }
    const div = el('div', 'msg ' + m.role);
    if (m.reasoning) div.appendChild(renderThink(m.reasoning));
    div.appendChild(document.createTextNode(m.content || ''));
    if (m.error) div.appendChild(el('div', 'msg-note', 'Error: ' + m.error));
    // El corte por cancelación se muestra aparte y no se reenvía al modelo.
    if (m.cancelled) div.appendChild(el('div', 'msg-note', '[cancelado]'));
    box.appendChild(div);
  }
  box.scrollTop = box.scrollHeight;
}

function params() {
  const p = {
    temperature: parseFloat($('#p-temp').value),
    top_p: parseFloat($('#p-topp').value),
    top_k: parseInt($('#p-topk').value, 10),
    max_tokens: parseInt($('#p-max').value, 10),
    reasoning_effort: $('#p-think').checked ? 'medium' : 'none',
  };
  const seed = parseInt($('#p-seed').value, 10);
  if (!isNaN(seed)) p.seed = seed;
  return p;
}

function handleDelta(chunk, acc) {
  let data;
  try { data = JSON.parse(chunk); } catch (_) { return false; }
  // Error a mitad del stream: el daemon manda `{"error": {...}}`.
  if (data.error) {
    acc.error = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
    return true;
  }
  const choice = data.choices && data.choices[0];
  if (!choice) return false;
  const d = choice.delta || {};
  if (d.reasoning_content) acc.reasoning += d.reasoning_content;
  if (d.content) acc.content += d.content;
  if (choice.finish_reason) acc.finished = true;
  return true;
}

async function send() {
  if (streaming) return;
  const text = $('#input').value.trim();
  if (!text) return;
  $('#input').value = '';
  messages.push({ role: 'user', content: text });
  const acc = { role: 'assistant', content: '', reasoning: '' };
  messages.push(acc);
  renderChat();
  saveChat();

  controller = new AbortController();
  streaming = true;
  setComposer(true);
  const body = Object.assign({ model: $('#model-id').textContent || 'brasa', stream: true, messages: chatMessages() }, params());
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      const t = await res.text();
      throw new Error('HTTP ' + res.status + ' ' + t.slice(0, 300));
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of raw.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          if (handleDelta(payload, acc)) renderChat();
        }
      }
      if (acc.error) throw new Error(acc.error);
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      // No se guarda dentro del contenido: se muestra aparte y no se reenvía al modelo.
      acc.cancelled = true;
    } else {
      messages.push({ role: 'error', content: 'Error: ' + e.message });
    }
  } finally {
    streaming = false;
    controller = null;
    setComposer(false);
    renderChat();
    saveChat();
    refreshEstado();
  }
}

function chatMessages() {
  return messages
    .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
    .map((m) => ({ role: m.role, content: m.content }));
}

function setComposer(busy) {
  $('#send').disabled = busy;
  $('#cancel').disabled = !busy;
  $('#input').disabled = busy;
}

$('#send').addEventListener('click', send);
$('#cancel').addEventListener('click', () => { if (controller) controller.abort(); });
$('#clear').addEventListener('click', () => { messages = []; saveChat(); renderChat(); });
$('#input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

/* ---------- Estado ---------- */
function kvList(node, pairs) {
  node.innerHTML = '';
  for (const [k, v] of pairs) { node.appendChild(el('dt', null, k)); node.appendChild(el('dd', null, v)); }
}

async function refreshEstado() {
  try {
    const [st, me] = await Promise.all([getJSON('/api/status'), getJSON('/api/metrics')]);
    const budget = (st.budget && st.budget.bytes) || 0;
    const total = st.plan.total || 0;
    const pct = budget ? Math.min(100, (total / budget) * 100) : 0;
    const bar = $('#mem-bar');
    bar.style.width = pct.toFixed(1) + '%';
    bar.className = 'bar-fill' + (pct >= 95 ? ' bad' : pct >= 80 ? ' warn' : '');
    $('#mem-legend').textContent = gib(total) + ' / ' + gib(budget) + ' (' + pct.toFixed(0) + '%)';
    kvList($('#mem-list'), [
      ['pesos', gib(st.plan.weights)],
      ['KV cache', gib(st.plan.kv)],
      ['workspace', gib(st.plan.workspace)],
      ['overhead', gib(st.plan.overhead)],
      ['presupuesto', st.budget.source || '—'],
    ]);
    kvList($('#perf-list'), [
      ['TTFT último', num(me.ttft_ms.last, 0) + ' ms'],
      ['TTFT p50', num(me.ttft_ms.p50, 0) + ' ms'],
      ['decode último', num(me.decode_tok_s.last) + ' tok/s'],
      ['decode media', num(me.decode_tok_s.mean) + ' tok/s'],
      ['tokens generados', String(me.generated_tokens)],
      ['prefix cache', String(me.cached_tokens) + ' tok'],
      ['cola', st.queue.pending + ' + ' + st.queue.running],
    ]);
    kvList($('#srv-list'), [
      ['versión', st.version + ' (' + st.commit + ')'],
      ['uptime', Math.round(st.uptime_s) + ' s'],
      ['modelo', st.model.id],
      ['contexto', st.context.ctx + ' (' + st.context.kv + ')'],
      ['pesos sha256', String(st.model.weights_sha256_declarado).slice(0, 16) + '…'],
      ['huella', gib(st.process.footprint)],
    ]);
    kvList($('#sys-list'), [
      ['presión', st.system.pressure],
      ['libre', gib(st.system.free)],
      ['disponible', st.system.available_percent == null ? '—' : st.system.available_percent + '%'],
      ['swap usado', mib(st.system.swap_used)],
      ['comprimida', gib(st.system.compressed)],
    ]);
    $('#estado-note').textContent = '';
  } catch (e) {
    $('#estado-note').textContent = 'No se pudo consultar el servidor: ' + e.message;
  }
}
setInterval(() => { if ($('#tab-estado').classList.contains('active')) refreshEstado(); }, 2000);

/* ---------- Benchmarks ---------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// El SVG se arma como markup y se inserta con innerHTML: el parser de HTML lo crea en el
// namespace correcto sin nombrarlo en el código.
function barChart(caption, rows, unit) {
  const fig = el('figure');
  fig.appendChild(el('figcaption', null, caption));
  if (!rows.length) { fig.appendChild(el('div', 'muted', 'sin datos')); return fig; }
  const W = 520, H = 220, pad = 34, base = H - pad;
  const max = Math.max.apply(null, rows.map((r) => r.value)) || 1;
  const bw = Math.max(6, (W - pad * 2) / rows.length - 6);
  const parts = [];
  rows.forEach((r, i) => {
    const h = ((base - pad) * r.value) / max;
    const x = pad + i * ((W - pad * 2) / rows.length);
    const color = r.valid ? 'var(--accent-2)' : 'var(--bad)';
    parts.push('<rect x="' + x + '" y="' + (base - h) + '" width="' + bw + '" height="' + h + '" fill="' + color + '" rx="2"></rect>');
    parts.push('<text x="' + (x + bw / 2) + '" y="' + (base - h - 4) + '" text-anchor="middle" font-size="10" fill="currentColor">' + r.value.toFixed(unit === 'ms' ? 0 : 1) + '</text>');
    parts.push('<text x="' + (x + bw / 2) + '" y="' + (base + 12) + '" text-anchor="middle" font-size="9" fill="currentColor">' + esc(r.label) + '</text>');
  });
  const wrap = el('div');
  wrap.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '">' + parts.join('') + '</svg>';
  fig.appendChild(wrap.firstChild);
  return fig;
}

async function loadBench() {
  const note = $('#bench-note');
  const tbody = $('#bench-table tbody');
  const charts = $('#bench-charts');
  tbody.innerHTML = '';
  charts.innerHTML = '';
  try {
    const data = await getJSON('/api/bench');
    note.textContent = data.exists ? data.dir + ' · ' + data.reports.length + ' reportes' : 'no existe ' + data.dir;
    for (const r of data.reports) {
      const tr = document.createElement('tr');
      const cells = [
        r.engine == null ? '—' : r.engine, r.label || 'default', r.model == null ? '—' : r.model, String(r.ctx == null ? '—' : r.ctx),
        num(r.ttft_ms, 0), num(r.prefill_tok_s), num(r.decode_tok_s),
        r.peak_footprint_bytes == null ? '—' : (r.peak_footprint_bytes / 1073741824).toFixed(2),
        r.valid ? 'sí' : 'no',
      ];
      cells.forEach((c, i) => {
        const td = el('td', !r.valid && i === 8 ? 'invalid' : null, c);
        tr.appendChild(td);
      });
      if (!r.valid && Array.isArray(r.invalid_reasons) && r.invalid_reasons.length) tr.title = r.invalid_reasons.join('; ');
      tbody.appendChild(tr);
    }
    const etiqueta = (r) => String(r.engine || r.file || '?').slice(0, 7) + '@' + (r.ctx == null ? '?' : r.ctx);
    const rows = data.reports.map((r) => ({
      label: etiqueta(r),
      value: r.decode_tok_s || 0,
      valid: r.valid,
    }));
    const ttft = data.reports.map((r) => ({
      label: etiqueta(r),
      value: r.ttft_ms || 0,
      valid: r.valid,
    }));
    charts.appendChild(barChart('decode tok/s', rows, 'tok/s'));
    charts.appendChild(barChart('TTFT ms', ttft, 'ms'));
    if (data.errors && data.errors.length) note.textContent += ' · ' + data.errors.length + ' con error';
  } catch (e) {
    note.textContent = 'Error: ' + e.message;
  }
}

/* ---------- Plan ---------- */
$('#plan-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = new URLSearchParams({
    ctx: $('#plan-ctx').value,
    chunk: $('#plan-chunk').value,
    kv: $('#plan-kv').value,
    perfil: $('#plan-perfil').value,
  });
  const out = $('#plan-out');
  out.textContent = 'calculando…';
  try {
    const p = await getJSON('/api/plan?' + q.toString());
    if (p.error) { out.textContent = 'Error: ' + p.error; return; }
    out.textContent = p.message + '\n\n' +
      'contexto pedido : ' + p.ctx + '\n' +
      'máximo del modelo: ' + p.model_max_ctx + '\n' +
      'ctx máximo que entra: ' + (p.max_ctx == null ? '—' : p.max_ctx) + '\n' +
      'presupuesto     : ' + p.budget.gib.toFixed(2) + ' GiB (' + p.budget.source + ')\n' +
      'plan            : pesos ' + gib(p.plan.weights) + ' + KV ' + gib(p.plan.kv) +
      ' + workspace ' + gib(p.plan.workspace) + ' + overhead ' + gib(p.plan.overhead) +
      ' = ' + gib(p.plan.total);
  } catch (err) {
    out.textContent = 'Error: ' + err.message;
  }
});

/* ---------- Agentes ---------- */
async function loadAgents() {
  const out = $('#agents-out');
  out.innerHTML = '';
  const ctx = $('#agents-ctx').value || '';
  try {
    const data = await getJSON('/api/agents?ctx=' + encodeURIComponent(ctx));
    for (const key of Object.keys(data.tools)) {
      const card = el('div', 'agent');
      const head = el('header');
      head.appendChild(el('strong', null, key));
      const copy = el('button', null, 'Copiar');
      copy.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(data.tools[key]); copy.textContent = 'Copiado'; }
        catch (_) { copy.textContent = 'No se pudo'; }
        setTimeout(() => { copy.textContent = 'Copiar'; }, 1500);
      });
      head.appendChild(copy);
      card.appendChild(head);
      card.appendChild(el('pre', null, data.tools[key]));
      out.appendChild(card);
    }
  } catch (e) {
    out.appendChild(el('div', 'muted', 'Error: ' + e.message));
  }
}
$('#agents-load').addEventListener('click', loadAgents);

/* ---------- Arranque ---------- */
(async function init() {
  renderChat();
  try {
    const st = await getJSON('/api/status');
    $('#model-id').textContent = st.model.id;
  } catch (_) { /* el servidor puede no responder todavía */ }
  refreshEstado();
})();
