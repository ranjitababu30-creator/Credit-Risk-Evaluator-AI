// static/js/app.js
// Shared helpers used across pages.

function buildFieldElement(field, prefix, value) {
  const wrapper = document.createElement('div');
  wrapper.className = 'field';

  const label = document.createElement('label');
  label.textContent = field.label;
  label.setAttribute('for', prefix + field.name);
  wrapper.appendChild(label);

  let input;
  if (field.type === 'select') {
    input = document.createElement('select');
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '-- select --';
    input.appendChild(blank);
    (field.options || []).forEach(opt => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      input.appendChild(o);
    });
  } else {
    input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
  }
  input.id = prefix + field.name;
  input.name = field.name;
  if (value !== undefined && value !== null) input.value = value;
  wrapper.appendChild(input);
  return wrapper;
}

function collectFormValues(containerEl, fields) {
  const values = {};
  fields.forEach(field => {
    const el = document.getElementById(containerEl.id + '_' + field.name) ||
               containerEl.querySelector(`[name="${field.name}"]`);
    if (!el) return;
    if (el.value === '') return;
    values[field.name] = field.type === 'number' ? parseFloat(el.value) : el.value;
  });
  return values;
}

function formatPercent(x) {
  return (x * 100).toFixed(1) + '%';
}
