// static/js/scenario.js

let SCEN_FIELDS = [];

function initScenarioForm() {
  const specEl = document.getElementById('field-spec');
  if (!specEl) return;
  SCEN_FIELDS = JSON.parse(specEl.textContent);

  let prefill = {};
  try {
    prefill = JSON.parse(sessionStorage.getItem('scenario_inputs') || '{}');
  } catch (e) { prefill = {}; }

  const originalContainer = document.getElementById('original-fields');
  const modifiedContainer = document.getElementById('modified-fields');

  SCEN_FIELDS.forEach(field => {
    originalContainer.appendChild(buildFieldElement(field, 'orig_', prefill[field.name]));
    modifiedContainer.appendChild(buildFieldElement(field, 'mod_', prefill[field.name]));
  });

  document.getElementById('btn-run-scenario').addEventListener('click', onRunScenario);
}

function readValues(prefix) {
  const values = {};
  SCEN_FIELDS.forEach(field => {
    const el = document.getElementById(prefix + field.name);
    if (!el || el.value === '') return;
    values[field.name] = field.type === 'number' ? parseFloat(el.value) : el.value;
  });
  return values;
}

async function onRunScenario() {
  const errBox = document.getElementById('scenario-errors');
  errBox.textContent = '';
  const original = readValues('orig_');
  const modified = readValues('mod_');

  try {
    const res = await fetch('/api/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ original, modified }),
    });
    const data = await res.json();
    if (!res.ok) { errBox.textContent = data.error || 'Something went wrong.'; return; }
    renderScenario(data);
  } catch (err) {
    errBox.textContent = 'Network or server error: ' + err;
  }
}

function renderScenario(data) {
  document.getElementById('scenario-result').classList.remove('hidden');
  document.getElementById('sc-before-score').textContent = data.before.risk_score;
  document.getElementById('sc-before-prob').textContent = formatPercent(data.before.probability_of_default);
  document.getElementById('sc-before-rec').textContent = data.before.recommendation;

  document.getElementById('sc-after-score').textContent = data.after.risk_score;
  document.getElementById('sc-after-prob').textContent = formatPercent(data.after.probability_of_default);
  document.getElementById('sc-after-rec').textContent = data.after.recommendation;

  const delta = data.change.risk_score_delta;
  const changeEl = document.getElementById('sc-change');
  changeEl.textContent = (delta > 0 ? '+' : '') + delta;
  changeEl.style.color = delta > 0 ? 'var(--good)' : (delta < 0 ? 'var(--bad)' : 'var(--muted)');

  const list = document.getElementById('sc-changed-fields');
  list.innerHTML = data.changed_fields.map(c =>
    `<li>${c.field}: ${c.before ?? '—'} → ${c.after ?? '—'}</li>`
  ).join('') || '<li>No fields changed.</li>';

  document.getElementById('scenario-result').scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', initScenarioForm);
