const DATA_PATHS = {
  recommendations: "../reports/business_recommendations.csv",
  features: "../data/processed/churn_features.csv",
  sweeps: {
    "Logistic Regression": "../reports/lr_threshold_sweep.csv",
    "Random Forest": "../reports/rf_threshold_sweep.csv",
    "XGBoost": "../reports/xgb_threshold_sweep.csv",
  },
  segmentScores: {
    "Logistic Regression": "../reports/lr_customer_scores.csv",
    "Random Forest": "../reports/rf_customer_scores.csv",
    "XGBoost": "../reports/xgb_customer_scores.csv",
  },
  importances: {
    "Logistic Regression": "../reports/lr_feature_importance.csv",
    "Random Forest": "../reports/rf_feature_importance.csv",
    "XGBoost": "../reports/xgb_feature_importance.csv",
  },
  shap: {
    "Logistic Regression": {
      summary: "../models/lr_shap_summary.png",
      force: "../models/lr_shap_force.png",
    },
    "Random Forest": {
      summary: "../models/rf_shap_summary.png",
      force: "../models/rf_shap_force.png",
    },
    "XGBoost": {
      summary: "../models/xgb_shap_summary.png",
      force: "../models/xgb_shap_force.png",
    },
  },
};

const state = {
  recommendations: [],
  features: [],
  sweeps: {},
  segmentScores: {},
  importances: {},
};

function parseCSV(csvText) {
  const rows = [];
  const lines = csvText.trim().split(/\r?\n/);
  if (!lines.length) return rows;

  const headers = splitCSVLine(lines[0]);
  for (let i = 1; i < lines.length; i += 1) {
    if (!lines[i].trim()) continue;
    const values = splitCSVLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

function splitCSVLine(line) {
  const out = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out.map((v) => v.trim());
}

async function loadCSV(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return parseCSV(await response.text());
}

function toNumber(value) {
  if (typeof value === "number") return value;
  const cleaned = String(value).replace(/[$,%\s,]/g, "");
  const n = Number(cleaned);
  return Number.isNaN(n) ? 0 : n;
}

function fmtInt(value) {
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function fmtMoney(value, digits = 0) {
  return Number(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function fmtPct(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function findMetric(metricName) {
  const row = state.recommendations.find((r) => r.Metric === metricName);
  return row ? row.Value : "N/A";
}

function metricCard(label, value, delta = "") {
  return `<article class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div>${delta ? `<div class="metric-delta">${delta}</div>` : ""}</article>`;
}

function setupTabs() {
  const buttons = Array.from(document.querySelectorAll(".tab-button"));
  const sections = Array.from(document.querySelectorAll(".tab-section"));

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tab;
      buttons.forEach((b) => b.classList.remove("active"));
      sections.forEach((s) => s.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(target).classList.add("active");
    });
  });
}

function renderOverview() {
  const totalCustomers = state.features.length;
  const churners = state.features.filter((r) => r.Churn === "Yes").length;
  const churnRate = totalCustomers ? churners / totalCustomers : 0;
  const clv = toNumber(findMetric("CLV (Average)"));
  const revenueAtRisk = churners * clv;
  const expectedSegmentRoi = findMetric("Expected Net ROI (Segmented)");

  document.getElementById("overviewMetrics").innerHTML = [
    metricCard("Total Customers", fmtInt(totalCustomers)),
    metricCard("Overall Churn Rate", fmtPct(churnRate)),
    metricCard("Revenue at Risk", fmtMoney(revenueAtRisk, 0)),
    metricCard("Expected ROI (Optimal)", expectedSegmentRoi),
  ].join("");

  const tableBody = document.querySelector("#recommendationTable tbody");
  tableBody.innerHTML = state.recommendations
    .map((r) => `<tr><td>${r.Metric}</td><td>${r.Value}</td></tr>`)
    .join("");

  const bestModel = findMetric("Recommended Model");
  document.getElementById("overviewInsights").innerHTML = `
    <h3>Primary Recommendation</h3>
    <p>Deploy the <strong>${bestModel}</strong> model. It provides the strongest overall performance on this dataset and best business ROI when paired with threshold optimization.</p>
    <p><strong>Absolute business impact:</strong> ${fmtInt(churners)} customers currently churned, representing approximately ${fmtMoney(revenueAtRisk, 0)} at risk based on average CLV.</p>
  `;
}

function nearestThresholdRow(rows, threshold) {
  return rows.reduce((best, current) => {
    const bestDiff = Math.abs(toNumber(best.Threshold) - threshold);
    const currDiff = Math.abs(toNumber(current.Threshold) - threshold);
    return currDiff < bestDiff ? current : best;
  }, rows[0]);
}

function renderDecisionControls() {
  const select = document.getElementById("decisionModel");
  select.innerHTML = Object.keys(DATA_PATHS.sweeps)
    .map((model) => `<option value="${model}">${model}</option>`)
    .join("");

  const thresholdInput = document.getElementById("decisionThreshold");
  const thresholdValue = document.getElementById("decisionThresholdValue");

  const update = () => {
    thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
    renderDecision();
  };

  select.addEventListener("change", renderDecision);
  thresholdInput.addEventListener("input", update);
}

function renderDecision() {
  const model = document.getElementById("decisionModel").value;
  const rows = state.sweeps[model];
  if (!rows || !rows.length) return;

  const threshold = Number(document.getElementById("decisionThreshold").value);
  const selected = nearestThresholdRow(rows, threshold);
  const optimal = rows.reduce((best, r) => (toNumber(r.Net_ROI) > toNumber(best.Net_ROI) ? r : best), rows[0]);

  const delta = `${fmtMoney(toNumber(selected.Net_ROI) - toNumber(optimal.Net_ROI), 0)} vs optimal`;

  document.getElementById("decisionMetrics").innerHTML = [
    metricCard("Customers Contacted", fmtInt(toNumber(selected.Customers_Contacted))),
    metricCard("Retention Cost", fmtMoney(toNumber(selected.Retention_Cost), 0)),
    metricCard("Revenue Saved", fmtMoney(toNumber(selected.Revenue_Saved), 0)),
    metricCard("Net ROI", fmtMoney(toNumber(selected.Net_ROI), 0), delta),
  ].join("");

  document.getElementById("decisionConfusion").innerHTML = [
    metricCard("True Positives", fmtInt(toNumber(selected.TP))),
    metricCard("False Positives", fmtInt(toNumber(selected.FP))),
    metricCard("False Negatives", fmtInt(toNumber(selected.FN))),
    metricCard("True Negatives", fmtInt(toNumber(selected.TN))),
  ].join("");

  document.querySelector("#decisionTable tbody").innerHTML = rows
    .map((r) => `
      <tr>
        <td>${Number(r.Threshold).toFixed(2)}</td>
        <td>${fmtInt(toNumber(r.Customers_Contacted))}</td>
        <td>${fmtInt(toNumber(r.TP))}</td>
        <td>${fmtInt(toNumber(r.FP))}</td>
        <td>${fmtInt(toNumber(r.FN))}</td>
        <td>${fmtPct(toNumber(r.Precision), 1)}</td>
        <td>${fmtPct(toNumber(r.Recall), 1)}</td>
        <td>${fmtMoney(toNumber(r.Net_ROI), 0)}</td>
      </tr>
    `)
    .join("");
}

function renderSegmentSummary() {
  const churned = state.features.filter((r) => r.Churn === "Yes");
  const monthlyCharges = state.features.map((r) => toNumber(r.MonthlyCharges));
  const median = monthlyCharges.slice().sort((a, b) => a - b)[Math.floor(monthlyCharges.length / 2)];

  const withSegment = state.features.map((r) => ({
    ...r,
    value_segment: toNumber(r.MonthlyCharges) >= median ? "High-Value" : "Low-Value",
  }));

  const high = withSegment.filter((r) => r.value_segment === "High-Value");
  const low = withSegment.filter((r) => r.value_segment === "Low-Value");

  const avg = (arr, field) => (arr.length ? arr.reduce((acc, v) => acc + toNumber(v[field]), 0) / arr.length : 0);
  const churnRate = (arr) => (arr.length ? arr.filter((r) => r.Churn === "Yes").length / arr.length : 0);

  const hvThreshold = toNumber(findMetric("High-Value Threshold"));
  const lvThreshold = toNumber(findMetric("Low-Value Threshold"));

  document.getElementById("segmentSummary").innerHTML = `
    <article class="panel">
      <h3 class="segment-card-title">High-Value Customers</h3>
      <div class="metric-grid small">
        ${metricCard("Count", fmtInt(high.length))}
        ${metricCard("Churn Rate", fmtPct(churnRate(high), 1))}
        ${metricCard("Avg Monthly Charge", fmtMoney(avg(high, "MonthlyCharges"), 2))}
        ${metricCard("CLV (24 months)", fmtMoney(avg(high, "MonthlyCharges") * 24, 2), `Threshold ${hvThreshold.toFixed(2)}`)}
      </div>
    </article>
    <article class="panel">
      <h3 class="segment-card-title">Low-Value Customers</h3>
      <div class="metric-grid small">
        ${metricCard("Count", fmtInt(low.length))}
        ${metricCard("Churn Rate", fmtPct(churnRate(low), 1))}
        ${metricCard("Avg Monthly Charge", fmtMoney(avg(low, "MonthlyCharges"), 2))}
        ${metricCard("CLV (24 months)", fmtMoney(avg(low, "MonthlyCharges") * 24, 2), `Threshold ${lvThreshold.toFixed(2)}`)}
      </div>
    </article>
  `;

  const totalChurners = churned.length;
  document.getElementById("segmentsNote").textContent = `${fmtInt(totalChurners)} churners identified in historical data. Table below shows model-scored customers above threshold.`;
}

function renderSegmentsControls() {
  const modelSelect = document.getElementById("segmentModel");
  modelSelect.innerHTML = Object.keys(DATA_PATHS.segmentScores)
    .map((model) => `<option value="${model}">${model}</option>`)
    .join("");

  const thresholdInput = document.getElementById("segmentThreshold");
  const thresholdValue = document.getElementById("segmentThresholdValue");
  const topNInput = document.getElementById("segmentTopN");

  const update = () => {
    thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
    renderSegmentsTable();
  };

  modelSelect.addEventListener("change", renderSegmentsTable);
  thresholdInput.addEventListener("input", update);
  topNInput.addEventListener("input", renderSegmentsTable);
}

function renderSegmentsTable() {
  const model = document.getElementById("segmentModel").value;
  const threshold = Number(document.getElementById("segmentThreshold").value);
  const topN = Number(document.getElementById("segmentTopN").value);
  const rows = state.segmentScores[model] || [];

  const filtered = rows
    .filter((r) => toNumber(r.churn_probability) >= threshold)
    .sort((a, b) => toNumber(b.churn_probability) - toNumber(a.churn_probability))
    .slice(0, topN);

  document.querySelector("#segmentsTable tbody").innerHTML = filtered
    .map((r) => {
      const segClass = r.value_segment === "High-Value" ? "high" : "low";
      const churnClass = r.Churn === "Yes" ? "churn-yes" : "churn-no";
      return `
        <tr>
          <td>${r.customerID}</td>
          <td><span class="badge ${segClass}">${r.value_segment}</span></td>
          <td>${fmtMoney(toNumber(r.MonthlyCharges), 2)}</td>
          <td>${r.Contract}</td>
          <td>${fmtInt(toNumber(r.tenure))}</td>
          <td>${fmtPct(toNumber(r.churn_probability), 1)}</td>
          <td><span class="badge ${churnClass}">${r.Churn}</span></td>
        </tr>
      `;
    })
    .join("");

  document.getElementById("segmentsNote").textContent = `${fmtInt(filtered.length)} customers flagged for retention outreach at threshold ${threshold.toFixed(2)} using ${model}.`;
}

function renderExplainabilityControls() {
  const modelSelect = document.getElementById("expModel");
  modelSelect.innerHTML = Object.keys(DATA_PATHS.importances)
    .map((model) => `<option value="${model}">${model}</option>`)
    .join("");

  modelSelect.addEventListener("change", renderExplainability);
}

function renderExplainability() {
  const model = document.getElementById("expModel").value;
  const rows = state.importances[model] || [];
  const topRows = rows.slice(0, 12);

  document.querySelector("#importanceTable tbody").innerHTML = topRows
    .map((r) => {
      const importance = model === "Logistic Regression" ? toNumber(r.Coefficient) : toNumber(r.Importance);
      const dir = model === "Logistic Regression" ? r.Direction : "N/A";
      return `<tr><td>${r.Feature}</td><td>${importance.toFixed(4)}</td><td>${dir}</td></tr>`;
    })
    .join("");

  const summaryImg = document.getElementById("shapSummary");
  const forceImg = document.getElementById("shapForce");
  summaryImg.src = DATA_PATHS.shap[model].summary;
  forceImg.src = DATA_PATHS.shap[model].force;
}

async function init() {
  const status = document.getElementById("dataStatus");
  try {
    setupTabs();

    const [recommendations, features] = await Promise.all([
      loadCSV(DATA_PATHS.recommendations),
      loadCSV(DATA_PATHS.features),
    ]);

    state.recommendations = recommendations;
    state.features = features;

    const sweepEntries = Object.entries(DATA_PATHS.sweeps);
    const scoreEntries = Object.entries(DATA_PATHS.segmentScores);
    const impEntries = Object.entries(DATA_PATHS.importances);

    for (const [model, path] of sweepEntries) {
      state.sweeps[model] = await loadCSV(path);
    }

    for (const [model, path] of scoreEntries) {
      state.segmentScores[model] = await loadCSV(path);
    }

    for (const [model, path] of impEntries) {
      state.importances[model] = await loadCSV(path);
    }

    renderOverview();
    renderDecisionControls();
    renderDecision();
    renderSegmentSummary();
    renderSegmentsControls();
    renderSegmentsTable();
    renderExplainabilityControls();
    renderExplainability();

    status.textContent = "Data loaded from project CSV artifacts";
  } catch (error) {
    status.textContent = "Data load failed";
    status.style.background = "#fee2e2";
    status.style.color = "#991b1b";
    console.error(error);
  }
}

init();
