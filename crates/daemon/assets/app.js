'use strict';

/* Brasa GUI. Todo lo que muestra sale de /api/* o de /v1/*; nada se inventa ni se estima acá. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const GiB = 1073741824;
const gib = (b) => (typeof b === 'number' ? (b / GiB).toFixed(2) + ' GiB' : '—');
const mib = (b) => (typeof b === 'number' ? Math.round(b / 1048576) + ' MiB' : '—');
const num = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
const int = (v) => (typeof v === 'number' && isFinite(v) ? Math.round(v).toLocaleString('es') : '—');
const ago = (ms) => {
  const s = Math.max(0, ms / 1000);
  if (s < 60) return Math.floor(s) + ' s';
  if (s < 3600) return Math.floor(s / 60) + ' min';
  return Math.floor(s / 3600) + ' h';
};
const uptime = (s) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? h + ' h ' + m + ' min' : m ? m + ' min' : Math.floor(s) + ' s';
};

async function getJSON(url, opts) {
  const r = await fetch(url, Object.assign({ headers: { Accept: 'application/json' } }, opts));
  if (!r.ok) {
    // El daemon devuelve {"error": "..."} en los 4xx: mostrarlo, no solo "HTTP 400".
    let msg = url + ' respondió HTTP ' + r.status;
    try {
      const j = await r.json();
      if (j && j.error) msg = typeof j.error === 'string' ? j.error : (j.error.message || msg);
    } catch (_) { /* el cuerpo no era JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

function kvList(node, pairs) {
  node.replaceChildren();
  for (const [k, v, dot, cls] of pairs) {
    const dt = el('dt');
    if (dot) { const d = el('span', 'dot'); d.style.background = dot; dt.appendChild(d); }
    dt.appendChild(document.createTextNode(k));
    node.appendChild(dt);
    node.appendChild(el('dd', cls || null, v));
  }
}

/* Cliente que marca la GUI en sus pedidos de chat (cabecera X-Brasa-Client). */
const GUI_CLIENT = 'brasa-gui';
const clientLabel = (c) => (c === GUI_CLIENT ? 'chat de la GUI' : c || '—');

/* Escalas fijas de la tira de calor (0 = ceniza): decode en pasos de 10 tok/s, prefill en pasos
   de 100 tok/s de prompt procesado. */
function heatLevel(tok, step = 10) {
  if (!(tok > 0.05)) return 0;
  if (tok >= step * 5) return 6;
  return 1 + Math.floor(tok / step);
}

/* ---------- Estado compartido ---------- */
const state = { status: null, metrics: null, activity: null, offline: false, view: 'monitor' };

function setOffline(off) {
  state.offline = off;
  $('#offline').hidden = !off;
  if (off) {
    $('#rail-led').dataset.state = 'offline';
    $('#rail-state').textContent = 'Sin conexión';
  }
}

const STATE_NAMES = {
  loading: 'Cargando', loaded: 'Cargado', idle: 'Descargado', paused: 'En pausa', stopped: 'Detenido',
};

function renderRail() {
  const st = state.status;
  if (!st || state.offline) return;
  const active = state.activity ? state.activity.active.length : 0;
  const led = $('#rail-led');
  if (st.state === 'loaded' && active) {
    led.dataset.state = 'busy';
    $('#rail-state').textContent = active === 1 ? 'Generando' : 'Generando · ' + active + ' pedidos';
  } else {
    led.dataset.state = st.state;
    $('#rail-state').textContent = STATE_NAMES[st.state] || st.state;
  }
  $('#rail-model').textContent = st.model.id + ' · ' + st.context.ctx.toLocaleString('es') + ' · ' + st.context.kv;
  $('#version').textContent = 'v' + st.version;
}

async function pollStatus() {
  try {
    const [st, me] = await Promise.all([getJSON('/api/status'), getJSON('/api/metrics')]);
    state.status = st;
    state.metrics = me;
    setOffline(false);
    renderRail();
    if (state.view === 'monitor') { renderMemory(); renderModel(); renderTotals(); }
  } catch (_) {
    setOffline(true);
  }
}

async function pollActivity() {
  try {
    state.activity = await getJSON('/api/activity');
    setOffline(false);
    renderRail();
    if (state.view === 'monitor') { renderHeat(); renderRequests(); }
    if (state.view === 'chat') renderChatStats();
  } catch (_) {
    setOffline(true);
  }
}

/* ---------- Navegación ---------- */
const VIEWS = ['monitor', 'chat', 'modelos', 'benchmarks', 'plan', 'agentes'];
function route() {
  const name = (location.hash || '#monitor').slice(1);
  const view = VIEWS.includes(name) ? name : 'monitor';
  state.view = view;
  for (const v of VIEWS) $('#v-' + v).hidden = v !== view;
  document.querySelectorAll('.nav a').forEach((a) => {
    if (a.dataset.view === view) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  document.title = (view === 'monitor' ? '' : $('#v-' + view + ' h1').textContent + ' · ') + 'brasa';
  if (view === 'monitor') { renderAllMonitor(); }
  if (view === 'chat') { renderChat(); renderChatStats(); }
  if (view === 'modelos') loadModels();
  if (view === 'benchmarks') loadBench();
  if (view === 'agentes') loadAgents();
}
window.addEventListener('hashchange', route);

/* ---------- Monitor ---------- */
function renderAllMonitor() {
  renderHeat(); renderRequests(); renderMemory(); renderModel(); renderTotals();
}

function fillLane(node, n) {
  if (node.childElementCount !== n) {
    node.replaceChildren();
    for (let i = 0; i < n; i++) node.appendChild(document.createElement('i'));
  }
  return node.children;
}

function renderHeat() {
  const a = state.activity;
  if (!a) return;
  const secs = a.seconds;
  const pre = fillLane($('#lane-prefill'), secs.length);
  const dec = fillLane($('#lane-decode'), secs.length);
  let tokens = 0, peak = 0, busy = 0, prompt = 0;
  secs.forEach((s, i) => {
    pre[i].dataset.h = s.prefill_live ? 'live' : String(heatLevel(s.prefill_tokens, 100));
    pre[i].title = s.prefill_live ? 'prefill en curso' : Math.round(s.prefill_tokens) + ' tok de prompt en ese segundo';
    dec[i].dataset.h = String(heatLevel(s.tokens));
    dec[i].title = Math.round(s.tokens) + ' tok generados en ese segundo';
    tokens += s.tokens;
    prompt += s.prefill_tokens;
    peak = Math.max(peak, s.tokens);
    busy += s.prefill;
  });
  const summary = tokens > 0 || busy > 0
    ? int(prompt) + ' tokens de prompt procesados en ' + num(busy, 1) + ' s de prefill; ' + int(tokens) +
      ' generados, con un pico de ' + int(peak) + ' en un segundo.'
    : 'Sin actividad en los últimos ' + a.history_s + ' s.';
  $('#heat-summary').textContent = summary;
}

function tagFor(outcome, phase) {
  if (phase === 'prefill') return ['tag live', 'prefill'];
  if (phase === 'decode') return ['tag live', 'generando'];
  if (outcome === 'ok') return ['tag ok', 'listo'];
  if (outcome === 'cancelled') return ['tag cold', 'cancelado'];
  return ['tag bad', 'error: ' + outcome];
}

function renderRequests() {
  const a = state.activity;
  if (!a) return;
  const tbody = $('#req-table tbody');
  tbody.replaceChildren();
  const now = a.now_unix_ms;
  const row = (cells, cls) => {
    const tr = el('tr', cls);
    for (const [text, c] of cells) {
      const td = el('td', c || null);
      if (text instanceof Node) td.appendChild(text); else td.textContent = text;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  };
  const tag = (o, p) => { const [c, t] = tagFor(o, p); return el('span', c, t); };
  for (const r of a.active) {
    row([
      [tag(null, r.phase)], [clientLabel(r.client), 'client'], [r.endpoint],
      [ago(now - r.started_unix_ms), 'n'], [r.ttft_ms == null ? '—' : int(r.ttft_ms) + ' ms', 'n'],
      ['—', 'n'], ['—', 'n'], ['—', 'n'], ['—', 'n'],
    ]);
  }
  for (const r of a.recent) {
    row([
      [tag(r.outcome)], [clientLabel(r.client), 'client'], [r.endpoint],
      [ago(now - r.ended_unix_ms), 'n'], [r.ttft_ms == null ? '—' : int(r.ttft_ms) + ' ms', 'n'],
      [num(r.decode_tok_s), 'n'], [int(r.input_tokens), 'n'], [int(r.cached_tokens), 'n'], [int(r.output_tokens), 'n'],
    ], r.outcome === 'ok' ? null : 'muted');
  }
  const empty = !a.active.length && !a.recent.length;
  $('#req-table').hidden = empty;
  $('#req-empty').hidden = !empty;
  $('#req-note').textContent = a.active.length
    ? a.active.length + ' en curso · ' + a.recent.length + ' recientes'
    : a.recent.length ? a.recent.length + ' recientes' : '';
  $('#mon-sub').textContent = a.active.length
    ? 'Hay ' + (a.active.length === 1 ? 'un pedido' : a.active.length + ' pedidos') + ' en curso.'
    : 'Sin pedidos en curso.';
}

const SEG_COLORS = ['--seg-weights', '--seg-kv', '--seg-workspace', '--seg-overhead'];
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

function renderMemory() {
  const st = state.status;
  if (!st) return;
  const budget = (st.budget && st.budget.bytes) || 0;
  const plan = st.plan || {};
  const resident = st.state === 'loaded' || st.state === 'paused';
  const foot = st.process && st.process.footprint;
  $('#mem-used').textContent = gib(foot);
  $('#mem-of').textContent = budget ? 'de ' + gib(budget) + ' de presupuesto' : '';
  const keys = ['weights', 'kv', 'workspace', 'overhead'];
  const stack = $('#mem-stack');
  keys.forEach((k) => {
    const seg = stack.querySelector('[data-k="' + k + '"]');
    const pct = resident && budget ? Math.min(100, (plan[k] / budget) * 100) : 0;
    seg.style.width = pct.toFixed(2) + '%';
  });
  stack.classList.toggle('full', resident && budget > 0 && plan.total / budget >= 0.95);
  const mark = $('#mem-mark');
  if (budget && typeof foot === 'number') {
    mark.style.left = Math.min(100, (foot / budget) * 100).toFixed(2) + '%';
    mark.classList.add('on');
  } else mark.classList.remove('on');
  stack.setAttribute('aria-label', 'Plan ' + gib(plan.total) + ', huella ' + gib(foot) + ', presupuesto ' + gib(budget));
  const colors = SEG_COLORS.map(cssVar);
  kvList($('#mem-list'), [
    ['Pesos', gib(plan.weights), colors[0]],
    ['Caché KV', gib(plan.kv), colors[1]],
    ['Workspace', gib(plan.workspace), colors[2]],
    ['Overhead', gib(plan.overhead), colors[3]],
    ['Huella del proceso', gib(foot), cssVar('--ember')],
    ['Residente', gib(st.process && st.process.resident)],
    ['Swap del sistema', mib(st.system && st.system.swap_used)],
  ]);
  const pressure = st.system && st.system.pressure;
  const pr = { normal: 'Presión normal', warning: 'Presión en aviso', critical: 'Presión crítica' }[pressure] || 'Presión desconocida';
  $('#mem-pressure').textContent = (resident ? 'Plan ' + gib(plan.total) + ' · ' : 'Modelo fuera de memoria · ') + pr;
}

function renderModel() {
  const st = state.status;
  if (!st) return;
  const m = st.model;
  kvList($('#model-list'), [
    ['Modelo', m.id],
    ['Estado', STATE_NAMES[st.state] || st.state],
    ['Contexto', st.context.ctx.toLocaleString('es') + ' tokens · KV ' + st.context.kv],
    ['Chunk de prefill', String(st.context.chunk)],
    ['Origen', m.source_repo + '@' + String(m.source_commit).slice(0, 8), null, 'mono-v'],
    ['sha256 declarado', String(m.weights_sha256_declarado).slice(0, 12) + '…', null, 'mono-v'],
    ['Cola', st.queue.pending + ' esperando · ' + st.queue.running + ' en curso'],
    ['En línea hace', uptime(st.uptime_s)],
    ['Versión', st.version + ' · ' + String(st.commit).slice(0, 12), null, 'mono-v'],
  ]);
  const can = {
    loaded: ['idle', 'pause', 'stop'],
    idle: ['load', 'stop'],
    paused: ['resume', 'idle', 'stop'],
    loading: [],
    stopped: [],
  }[st.state] || [];
  document.querySelectorAll('#controls [data-op]').forEach((b) => {
    if (b.getAttribute('aria-busy') === 'true') return;
    b.disabled = !can.includes(b.dataset.op);
  });
  $('#model-state-note').textContent = STATE_NAMES[st.state] || st.state;
}

function renderTotals() {
  const me = state.metrics;
  if (!me) return;
  const reqs = Object.values(me.endpoints).reduce((a, b) => a + b, 0);
  const errs = Object.values(me.errors).reduce((a, b) => a + b, 0);
  const cachePct = me.prompt_tokens ? Math.round((me.cached_tokens / me.prompt_tokens) * 100) : null;
  const node = $('#totals');
  node.replaceChildren();
  const add = (k, v, small, title) => {
    const d = el('div');
    if (title) d.title = title;
    d.appendChild(el('dt', null, k));
    const dd = el('dd', null, v);
    if (small) dd.appendChild(el('small', null, ' ' + small));
    d.appendChild(dd);
    node.appendChild(d);
  };
  add('Pedidos', int(reqs), null, Object.entries(me.endpoints).map(([k, v]) => k + ': ' + v).join('\n'));
  add('Tokens de prompt', int(me.prompt_tokens));
  add('Desde el prefix cache', int(me.cached_tokens), cachePct == null ? null : cachePct + '%');
  add('Tokens generados', int(me.generated_tokens));
  add('TTFT p50', num(me.ttft_ms.p50, 0), 'ms');
  add('Decode, media', num(me.decode_tok_s.mean), 'tok/s');
  add('Cancelados', int(me.cancelled));
  add('Errores', int(errs), null, Object.entries(me.errors).map(([k, v]) => k + ': ' + v).join('\n'));
  $('#tot-note').textContent = me.ttft_ms.samples ? 'Medias sobre los últimos ' + me.ttft_ms.samples + ' pedidos' : '';
}

/* ---------- Model Manager ---------- */
const OP_DONE = {
  load: 'Modelo cargado.', idle: 'Modelo descargado: pesos y KV liberados.',
  pause: 'En pausa: los pedidos esperan en la cola.', resume: 'Reanudado.',
  stop: 'Servidor detenido. Para volver a usarlo, corré brasa serve en la terminal.',
};
let stopArmed = null;
document.querySelectorAll('#controls [data-op]').forEach((b) => {
  b.addEventListener('click', async () => {
    const op = b.dataset.op;
    const msg = $('#control-msg');
    if (op === 'stop' && !stopArmed) {
      b.classList.add('confirm');
      b.lastChild.textContent = '¿Detener? Confirmar';
      stopArmed = setTimeout(() => {
        b.classList.remove('confirm');
        b.lastChild.textContent = 'Detener servidor';
        stopArmed = null;
      }, 4000);
      return;
    }
    if (stopArmed) { clearTimeout(stopArmed); stopArmed = null; }
    document.querySelectorAll('#controls [data-op]').forEach((x) => { x.disabled = true; });
    b.setAttribute('aria-busy', 'true');
    msg.className = 'control-msg';
    msg.textContent = { load: 'Cargando el modelo…', idle: 'Descargando…', pause: 'Pausando…', resume: 'Reanudando…', stop: 'Deteniendo…' }[op];
    try {
      await getJSON('/api/model/' + op, { method: 'POST' });
      msg.textContent = OP_DONE[op];
    } catch (e) {
      msg.className = 'control-msg bad';
      msg.textContent = 'No se pudo: ' + e.message;
    } finally {
      b.removeAttribute('aria-busy');
      if (op === 'stop') {
        b.classList.remove('confirm');
        b.lastChild.textContent = 'Detener servidor';
      }
      pollStatus();
    }
  });
});

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
  d.appendChild(el('summary', null, 'Razonamiento'));
  d.appendChild(el('div', null, text));
  return d;
}

function renderChat() {
  const box = $('#messages');
  box.replaceChildren();
  if (!messages.length) {
    const e = el('div', 'chat-empty');
    e.appendChild(el('strong', null, 'Probá el modelo que está sirviendo este servidor.'));
    e.appendChild(document.createTextNode('Los parámetros de este panel van en cada pedido. La conversación queda solo en este navegador.'));
    box.appendChild(e);
    return;
  }
  messages.forEach((m, i) => {
    if (m.role === 'error') { box.appendChild(el('div', 'msg error', m.content)); return; }
    const live = streaming && i === messages.length - 1 && m.role === 'assistant';
    const div = el('div', 'msg ' + m.role + (live ? ' streaming' : ''));
    if (m.reasoning) div.appendChild(renderThink(m.reasoning));
    if (m.role === 'assistant') div.appendChild(renderMarkdown(m.content || ''));
    else div.appendChild(document.createTextNode(m.content || ''));
    if (m.error) div.appendChild(el('div', 'msg-note bad', 'Error: ' + m.error));
    // El corte por cancelación se muestra aparte y no se reenvía al modelo.
    if (m.cancelled) div.appendChild(el('div', 'msg-note', 'Cancelado por vos; este fragmento no se reenvía.'));
    box.appendChild(div);
  });
  box.scrollTop = box.scrollHeight;
}

/* Markdown mínimo para las respuestas: títulos, listas, separadores, bloques de código, negrita,
   itálica y código en línea. Se arma con nodos de texto: nunca innerHTML con texto del modelo. */
function inlineMd(parent, text) {
  const re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    const t = m[0];
    if (t[0] === '`') parent.appendChild(el('code', null, t.slice(1, -1)));
    else if (t.startsWith('**')) parent.appendChild(el('strong', null, t.slice(2, -2)));
    else parent.appendChild(el('em', null, t.slice(1, -1)));
    last = m.index + t.length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

function renderMarkdown(src) {
  const root = el('div', 'md');
  const lines = src.split('\n');
  let list = null, para = null, code = null;
  const close = () => { list = null; para = null; };
  for (const line of lines) {
    if (code) {
      if (/^```/.test(line)) { code = null; continue; }
      code.textContent += (code.textContent ? '\n' : '') + line;
      continue;
    }
    if (/^```/.test(line)) { close(); const pre = el('pre'); code = el('code'); pre.appendChild(code); root.appendChild(pre); continue; }
    if (!line.trim()) { close(); continue; }
    let m;
    if ((m = /^(#{1,6})\s+(.*)$/.exec(line))) {
      close();
      const h = el('h' + Math.min(5, m[1].length + 2));
      inlineMd(h, m[2]);
      root.appendChild(h);
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { close(); root.appendChild(el('hr')); continue; }
    if ((m = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line))) {
      const ordered = /\d/.test(m[1]);
      if (!list || (list.tagName === 'OL') !== ordered) { para = null; list = el(ordered ? 'ol' : 'ul'); root.appendChild(list); }
      const li = el('li');
      inlineMd(li, m[2]);
      list.appendChild(li);
      continue;
    }
    list = null;
    if (!para) { para = el('p'); root.appendChild(para); } else para.appendChild(el('br'));
    inlineMd(para, line);
  }
  return root;
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

function chatMessages() {
  return messages
    .filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content))
    .map((m) => ({ role: m.role, content: m.content }));
}

function setComposer(busy) {
  $('#send').hidden = busy;
  $('#cancel').hidden = !busy;
  $('#input').disabled = busy;
  $('#clear').disabled = busy;
  $('#composer-state').textContent = busy ? 'Generando…' : '';
  if (!busy) $('#input').focus();
}

async function send() {
  if (streaming) return;
  const text = $('#input').value.trim();
  if (!text) return;
  $('#input').value = '';
  messages.push({ role: 'user', content: text });
  const acc = { role: 'assistant', content: '', reasoning: '' };
  messages.push(acc);
  controller = new AbortController();
  streaming = true;
  setComposer(true);
  renderChat();
  saveChat();
  const model = state.status ? state.status.model.id : 'brasa';
  const body = Object.assign({ model, stream: true, messages: chatMessages() }, params());
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', 'X-Brasa-Client': GUI_CLIENT },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      const t = await res.text();
      let m = 'HTTP ' + res.status;
      try { const j = JSON.parse(t); m = (j.error && (j.error.message || j.error)) || m; } catch (_) { m += ' ' + t.slice(0, 300); }
      throw new Error(m);
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
    } else if (!acc.error) {
      // Si el error vino a mitad del stream, ya se muestra en la nota del mensaje: no duplicar.
      messages.push({ role: 'error', content: 'Error: ' + e.message });
    }
  } finally {
    streaming = false;
    controller = null;
    setComposer(false);
    renderChat();
    saveChat();
    pollActivity();
  }
}

$('#composer').addEventListener('submit', (e) => { e.preventDefault(); send(); });
$('#cancel').addEventListener('click', () => { if (controller) controller.abort(); });
$('#clear').addEventListener('click', () => { messages = []; saveChat(); renderChat(); });
$('#input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});

function renderChatStats() {
  const a = state.activity;
  if (!a) return;
  const r = a.recent.find((x) => x.client === GUI_CLIENT);
  const live = a.active.find((x) => x.client === GUI_CLIENT);
  if (live) {
    kvList($('#chat-stats'), [
      ['Estado', live.phase === 'prefill' ? 'Prefill' : 'Generando'],
      ['TTFT', live.ttft_ms == null ? '—' : int(live.ttft_ms) + ' ms'],
      ['Hace', ago(a.now_unix_ms - live.started_unix_ms)],
    ]);
  } else if (r) {
    kvList($('#chat-stats'), [
      ['Resultado', { ok: 'Listo', cancelled: 'Cancelado' }[r.outcome] || 'Error: ' + r.outcome],
      ['TTFT', r.ttft_ms == null ? '—' : int(r.ttft_ms) + ' ms'],
      ['Decode', r.decode_tok_s == null ? '—' : num(r.decode_tok_s) + ' tok/s'],
      ['Prompt', int(r.input_tokens) + ' tok'],
      ['Del prefix cache', int(r.cached_tokens) + ' tok'],
      ['Salida', int(r.output_tokens) + ' tok'],
    ]);
  } else {
    kvList($('#chat-stats'), [['Todavía no mandaste nada desde este chat.', '']]);
  }
  const cells = fillLane($('#mini-heat'), 60);
  const secs = a.seconds.slice(-60);
  secs.forEach((s, i) => { cells[i].dataset.h = String(heatLevel(s.tokens)); });
}

/* ---------- Modelos ---------- */
async function loadModels() {
  const tbody = $('#inst-table tbody');
  const cbody = $('#cat-table tbody');
  try {
    const [m, st] = await Promise.all([getJSON('/api/models'), getJSON('/api/status')]);
    state.status = st;
    $('#models-sub').textContent = 'Carpeta ' + m.dir + ' · el plan usa el contexto ' + m.ctx.toLocaleString('es') + ' y KV ' + m.kv + ' de este servidor.';
    tbody.replaceChildren();
    for (const x of m.installed) {
      const tr = el('tr');
      const name = el('td', null, x.name);
      const stTd = el('td');
      if (x.loaded) {
        const live = st.state === 'loaded' || st.state === 'paused';
        stTd.appendChild(el('span', live ? 'tag ok' : 'tag cold', live ? 'En memoria' : STATE_NAMES[st.state] || st.state));
      } else if (!x.ok) {
        stTd.appendChild(el('span', 'tag bad', 'Encabezado inválido'));
        stTd.title = x.error || '';
      } else {
        stTd.appendChild(el('span', 'tag cold', 'En disco'));
      }
      const plan = x.plan == null ? '—' : x.plan.fits ? gib(x.plan.total_bytes) : 'No entra';
      const foot = x.loaded ? gib(st.process.footprint) : '—';
      const res = x.loaded ? gib(st.process.resident) : '—';
      [name, stTd, el('td', 'n', gib(x.bytes)), el('td', 'n', plan), el('td', 'n', foot), el('td', 'n', res)].forEach((td) => tr.appendChild(td));
      if (!x.loaded) tr.className = 'muted';
      tbody.appendChild(tr);
    }
    $('#inst-note').textContent = m.installed.length ? m.installed.length + (m.installed.length === 1 ? ' modelo' : ' modelos') : 'No hay modelos .brasa en la carpeta.';
    cbody.replaceChildren();
    for (const c of m.catalog) {
      const tr = el('tr');
      const cmd = el('td');
      const code = el('code', null, 'brasa pull ' + c.name);
      const b = el('button', 'btn ghost');
      b.type = 'button';
      b.setAttribute('aria-label', 'Copiar el comando');
      b.innerHTML = '<svg class="ic"><use href="#i-copy"/></svg>';
      b.addEventListener('click', () => copy(b, 'brasa pull ' + c.name, null));
      cmd.append(code, ' ', b);
      [el('td', null, c.name), el('td', null, c.family), el('td', null, c.quant || '—'), el('td', null, c.license || '—'),
        el('td', 'n', int(c.max_context)), el('td', 'n', gib(c.download_bytes)), cmd].forEach((td) => tr.appendChild(td));
      cbody.appendChild(tr);
    }
    $('#cat-table').hidden = !m.catalog.length;
    $('#cat-empty').hidden = m.catalog.length > 0;
  } catch (e) {
    $('#models-sub').textContent = 'No se pudo leer la lista de modelos: ' + e.message;
  }
}

async function copy(btn, text, label) {
  const old = btn.innerHTML;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copiado';
  } catch (_) {
    btn.textContent = 'No se pudo copiar';
  }
  setTimeout(() => { if (label) btn.textContent = label; else btn.innerHTML = old; }, 1500);
}

/* ---------- Benchmarks ---------- */
function hbarChart(title, reports, key, unit, digits) {
  const panel = el('section', 'panel');
  const head = el('div', 'panel-head');
  head.appendChild(el('h2', null, title));
  head.appendChild(el('p', 'panel-note', 'mediana del último reporte de cada engine y variante · una escala por gráfico'));
  panel.appendChild(head);
  // Un reporte por engine, variante y contexto: el más reciente (la tabla de abajo tiene todos).
  const latest = new Map();
  reports.filter((r) => typeof r[key] === 'number').forEach((r) => {
    const id = [r.engine, r.label || 'default', r.ctx].join('|');
    const prev = latest.get(id);
    if (!prev || String(r.timestamp || '') > String(prev.timestamp || '')) latest.set(id, r);
  });
  const rows = [...latest.values()];
  if (!rows.length) { panel.appendChild(el('p', 'empty-inline', 'Ningún reporte trae este dato.')); return panel; }
  const max = Math.max.apply(null, rows.map((r) => r[key]));
  const grid = el('div', 'hbars');
  const byCtx = new Map();
  rows.forEach((r) => {
    const k = r.ctx == null ? '?' : r.ctx;
    if (!byCtx.has(k)) byCtx.set(k, []);
    byCtx.get(k).push(r);
  });
  [...byCtx.keys()].sort((a, b) => a - b).forEach((ctx) => {
    grid.appendChild(el('div', 'grp', 'Contexto ' + (typeof ctx === 'number' ? ctx.toLocaleString('es') : ctx)));
    byCtx.get(ctx).forEach((r) => {
      const name = String(r.engine || '?') + (r.label && r.label !== 'default' ? ' · ' + r.label : '');
      const lbl = el('div', 'lbl', name);
      lbl.title = name + (r.file ? ' — ' + r.file : '');
      const track = el('div', 'track');
      const fill = el('div', 'fill' + (!r.valid ? ' invalid' : r.engine === 'brasa' ? ' own' : ''));
      fill.style.width = ((r[key] / max) * 100).toFixed(1) + '%';
      track.appendChild(fill);
      grid.append(lbl, track, el('div', 'val', r[key].toFixed(digits) + (r.valid ? '' : ' *')));
    });
  });
  panel.appendChild(grid);
  const keyRow = el('div', 'chart-key');
  const k = (cls, t) => { const s = el('span'); const i = el('i', 'fill ' + cls); s.append(i, t); return s; };
  keyRow.append(k('own', 'brasa'), k('', 'baseline'), k('invalid', 'no válido (*)'));
  keyRow.querySelectorAll('i').forEach((i, n) => {
    i.style.background = n === 0 ? 'var(--ink)' : n === 1 ? 'var(--seg-kv)' : '';
  });
  keyRow.lastChild.querySelector('i').style.background = 'repeating-linear-gradient(135deg, var(--bad) 0 3px, transparent 3px 6px)';
  panel.appendChild(keyRow);
  panel.dataset.unit = unit;
  return panel;
}

async function loadBench() {
  const note = $('#bench-note');
  const tbody = $('#bench-table tbody');
  const charts = $('#bench-charts');
  tbody.replaceChildren();
  charts.replaceChildren();
  try {
    const data = await getJSON('/api/bench');
    if (!data.exists) { note.textContent = 'No existe ' + data.dir + '.'; return; }
    note.textContent = data.reports.length + ' reportes en ' + data.dir + (data.errors && data.errors.length ? ' · ' + data.errors.length + ' con error' : '');
    for (const r of data.reports) {
      const tr = el('tr', r.valid ? null : 'muted');
      const cells = [
        [r.engine == null ? '—' : r.engine], [r.label || 'default'], [r.model == null ? '—' : r.model],
        [r.ctx == null ? '—' : r.ctx.toLocaleString('es'), 'n'],
        [num(r.ttft_ms, 0), 'n'], [num(r.prefill_tok_s), 'n'], [num(r.decode_tok_s), 'n'],
        [r.peak_footprint_bytes == null ? '—' : (r.peak_footprint_bytes / GiB).toFixed(2), 'n'],
      ];
      cells.forEach(([t, c]) => tr.appendChild(el('td', c || null, t)));
      const v = el('td');
      v.appendChild(el('span', r.valid ? 'tag ok' : 'tag bad', r.valid ? 'sí' : 'no'));
      if (!r.valid && Array.isArray(r.invalid_reasons) && r.invalid_reasons.length) v.title = r.invalid_reasons.join('; ');
      tr.appendChild(v);
      tbody.appendChild(tr);
    }
    charts.appendChild(hbarChart('Decode tok/s', data.reports, 'decode_tok_s', 'tok/s', 1));
    charts.appendChild(hbarChart('Prefill tok/s', data.reports, 'prefill_tok_s', 'tok/s', 0));
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
  out.replaceChildren(el('p', 'empty-inline', 'Calculando…'));
  try {
    const p = await getJSON('/api/plan?' + q.toString());
    out.replaceChildren();
    const v = el('p', 'verdict');
    v.appendChild(el('span', p.fits ? 'tag ok' : 'tag bad', p.fits ? 'Entra' : 'No entra'));
    v.appendChild(document.createTextNode(p.ctx.toLocaleString('es') + ' tokens · KV ' + p.kv + ' · ' + gib(p.plan.total) + ' de ' + p.budget.gib.toFixed(2) + ' GiB'));
    out.appendChild(v);
    const stack = el('div', 'stack');
    const keys = ['weights', 'kv', 'workspace', 'overhead'];
    keys.forEach((k) => {
      const s = el('span', 'seg');
      s.dataset.k = k;
      s.style.width = Math.min(100, (p.plan[k] / p.budget.bytes) * 100).toFixed(2) + '%';
      stack.appendChild(s);
    });
    if (!p.fits) stack.classList.add('full');
    out.appendChild(stack);
    out.appendChild(el('p', 'plan-msg', p.message));
    const cols = el('div', 'plan-cols');
    const a = el('dl', 'kv');
    const colors = SEG_COLORS.map(cssVar);
    kvList(a, [
      ['Pesos', gib(p.plan.weights), colors[0]], ['Caché KV', gib(p.plan.kv), colors[1]],
      ['Workspace', gib(p.plan.workspace), colors[2]], ['Overhead', gib(p.plan.overhead), colors[3]],
      ['Total', gib(p.plan.total)],
    ]);
    const b = el('dl', 'kv');
    kvList(b, [
      ['Contexto pedido', p.ctx.toLocaleString('es')],
      ['Máximo del modelo', int(p.model_max_ctx)],
      ['Máximo que entra', p.max_ctx == null ? '—' : int(p.max_ctx)],
      ['Presupuesto', p.budget.gib.toFixed(2) + ' GiB'],
      ['Fuente', p.budget.source],
    ]);
    cols.append(a, b);
    out.appendChild(cols);
  } catch (err) {
    out.replaceChildren(el('p', 'msg-note bad', 'Error: ' + err.message));
  }
});

/* ---------- Agentes ---------- */
async function loadAgents() {
  const out = $('#agents-out');
  // Si el campo quedó vacío, no mandar `ctx=`: el daemon responde 400 con vacío.
  const ctx = $('#agents-ctx').value.trim();
  const url = ctx ? '/api/agents?ctx=' + encodeURIComponent(ctx) : '/api/agents';
  try {
    const data = await getJSON(url);
    out.replaceChildren();
    for (const key of Object.keys(data.tools)) {
      const card = el('section', 'panel agent');
      const head = el('div', 'panel-head');
      head.appendChild(el('h2', null, key));
      const b = el('button', 'btn', 'Copiar');
      b.type = 'button';
      b.addEventListener('click', () => copy(b, data.tools[key], 'Copiar'));
      head.appendChild(b);
      card.appendChild(head);
      card.appendChild(el('pre', null, data.tools[key]));
      out.appendChild(card);
    }
  } catch (e) {
    out.replaceChildren(el('p', 'msg-note bad', 'Error: ' + e.message));
  }
}
$('#agents-form').addEventListener('submit', (e) => { e.preventDefault(); loadAgents(); });

/* ---------- Arranque ---------- */
(async function init() {
  await pollStatus();
  const ac = $('#agents-ctx');
  if (ac && state.status && state.status.context) ac.value = state.status.context.ctx;
  await pollActivity();
  route();
  setInterval(() => { if (!document.hidden) pollActivity(); }, 1000);
  setInterval(() => { if (!document.hidden) pollStatus(); }, 2000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { pollStatus(); pollActivity(); } });
})();
