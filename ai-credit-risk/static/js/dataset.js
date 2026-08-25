function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[character]));
}

function resultTable(items, showAnomaly) {
  if (!items.length) return '<p class="muted">No rows in this group.</p>';
  const anomalyHeader = showAnomaly ? '<th>Anomaly score</th>' : '';
  const rows = items.map((item) => `<tr><td>${item.row_number}</td><td>${item.risk_score}</td><td>${item.risk_category}</td><td>${item.recommendation}</td><td>${item.probability_of_default}</td>${showAnomaly ? `<td>${item.anomaly_score}</td>` : ''}</tr>`).join('');
  return `<div class="table-scroll"><table class="data-table"><thead><tr><th>Source row</th><th>Risk score</th><th>Category</th><th>Decision</th><th>Default probability</th>${anomalyHeader}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

document.getElementById('dataset-form')?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const status = document.getElementById('dataset-status');
  const file = document.getElementById('dataset-file').files[0];
  if (!file) return;
  status.textContent = 'Processing dataset...';
  status.className = 'status-banner status-neutral';
  const formData = new FormData();
  formData.append('dataset', file);
  try {
    const response = await fetch('/api/batch-predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Dataset processing failed.');
    status.textContent = `${data.processed_rows} of ${data.total_rows} rows processed.`;
    document.getElementById('dataset-summary').classList.remove('hidden');
    document.getElementById('dataset-panels').classList.remove('hidden');
    document.getElementById('dataset-title').textContent = `Results: ${escapeHtml(data.filename)}`;
    const counts = [['Approved', data.groups.APPROVE.length], ['Manual review', data.groups.REVIEW.length], ['Cannot be approved', data.groups.REJECT.length], ['Anomalous', data.groups.ANOMALOUS.length]];
    document.getElementById('dataset-counts').innerHTML = counts.map(([label, count]) => `<div class="result-block"><div class="result-label">${label}</div><div class="result-value">${count}</div></div>`).join('');
    document.getElementById('approve-results').innerHTML = resultTable(data.groups.APPROVE, false);
    document.getElementById('review-results').innerHTML = resultTable(data.groups.REVIEW, false);
    document.getElementById('reject-results').innerHTML = resultTable(data.groups.REJECT, false);
    document.getElementById('anomaly-results').innerHTML = resultTable(data.groups.ANOMALOUS, true);
    const errorsPanel = document.getElementById('batch-errors-panel');
    errorsPanel.classList.toggle('hidden', !data.errors.length);
    document.getElementById('batch-errors').innerHTML = data.errors.map((item) => `<p>Row ${item.row_number}: ${escapeHtml(item.error)}</p>`).join('');
  } catch (error) {
    status.textContent = error.message;
    status.className = 'status-banner status-bad';
  }
});