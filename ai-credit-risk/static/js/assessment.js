// static/js/assessment.js

let ASSESS_FIELDS = [];
let LAST_INPUTS = {};

function initAssessmentForm() {
  const specEl = document.getElementById('field-spec');
  if (!specEl) return;
  ASSESS_FIELDS = JSON.parse(specEl.textContent);

  const container = document.getElementById('form-fields');
  ASSESS_FIELDS.forEach(field => {
    container.appendChild(buildFieldElement(field, 'field_'));
  });

  document.getElementById('assessment-form').addEventListener('submit', onSubmitAssessment);
  document.getElementById('btn-explain').addEventListener('click', onViewExplanation);
  document.getElementById('btn-scenario').addEventListener('click', onRunScenario);
  document.getElementById('btn-reset').addEventListener('click', onNewAssessment);
}

function readFormValues() {
  const values = {};
  ASSESS_FIELDS.forEach(field => {
    const el = document.getElementById('field_' + field.name);
    if (!el || el.value === '') return;
    values[field.name] = field.type === 'number' ? parseFloat(el.value) : el.value;
  });
  return values;
}

async function onSubmitAssessment(e) {
  e.preventDefault();
  const errBox = document.getElementById('form-errors');
  errBox.textContent = '';
  LAST_INPUTS = readFormValues();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(LAST_INPUTS),
    });
    const data = await res.json();
    if (!res.ok) {
      errBox.textContent = data.error || 'Something went wrong.';
      return;
    }
    renderResult(data);
  } catch (err) {
    errBox.textContent = 'Network or server error: ' + err;
  }
}

function renderResult(data) {
  document.getElementById('result-panel').classList.remove('hidden');
  document.getElementById('res-score').textContent = `${data.risk_score} / 1000`;
  document.getElementById('res-prob').textContent = formatPercent(data.probability_of_default);
  document.getElementById('res-category').textContent = data.risk_category;
  document.getElementById('res-recommendation').textContent = data.recommendation;

  const reasons = document.getElementById('res-reasons');
  reasons.innerHTML = '';
  (data.decision_reasons || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    reasons.appendChild(li);
  });

  const anomalyBadge = document.getElementById('res-anomaly');
  anomalyBadge.textContent = data.anomaly.label;
  anomalyBadge.className = 'badge ' + (data.anomaly.is_anomalous ? 'status-warn' : 'status-good');

  document.getElementById('explanation-box').classList.add('hidden');
  document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth' });
}

async function onViewExplanation() {
  const box = document.getElementById('explanation-box');
  box.classList.remove('hidden');
  box.innerHTML = 'Loading explanation…';
  try {
    const res = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(LAST_INPUTS),
    });
    const data = await res.json();
    if (!res.ok) { box.textContent = data.error || 'Could not load explanation.'; return; }
    const exp = data.explanation;
    const fmt = items => items.map(i => `<li>${i.explanation}</li>`).join('') || '<li>None identified.</li>';
    box.innerHTML = `
      <h4>Risk-increasing factors</h4>
      <ul class="reason-list">${fmt(exp.risk_increasing_factors)}</ul>
      <h4>Risk-reducing factors</h4>
      <ul class="reason-list">${fmt(exp.risk_reducing_factors)}</ul>
      <p class="muted small">${exp.disclaimer}</p>
    `;
  } catch (err) {
    box.textContent = 'Network or server error: ' + err;
  }
}

function onRunScenario() {
  sessionStorage.setItem('scenario_inputs', JSON.stringify(LAST_INPUTS));
  window.location.href = '/scenario';
}

function onNewAssessment() {
  ASSESS_FIELDS.forEach(field => {
    const el = document.getElementById('field_' + field.name);
    if (el) el.value = '';
  });
  document.getElementById('result-panel').classList.add('hidden');
  document.getElementById('form-errors').textContent = '';
}

document.addEventListener('DOMContentLoaded', initAssessmentForm);
