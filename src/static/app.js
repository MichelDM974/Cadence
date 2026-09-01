const charts = {};
let currentSource = null; // { type: "store", storeId } | { type: "upload", file, dateCol, valueCol }
let lastForecastRows = null;
let lastAnalysisData = null;
let lastForecastData = null;

const el = (id) => document.getElementById(id);

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderLineChart(canvasId, labels, datasets) {
  if (charts[canvasId]) charts[canvasId].destroy();
  const text = cssVar("--muted");
  const grid = cssVar("--chart-grid");
  charts[canvasId] = new Chart(el(canvasId), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: datasets.length > 1, labels: { color: text } } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, color: text }, grid: { color: grid } },
        y: { ticks: { color: text }, grid: { color: grid } },
      },
    },
  });
}

function simpleDataset(label, data, color) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: "transparent",
    pointRadius: 0,
    borderWidth: 1.5,
    tension: 0.15,
  };
}

async function loadStores() {
  const res = await fetch("/api/stores");
  const stores = await res.json();
  const select = el("store-select");
  select.innerHTML = stores
    .map((s) => `<option value="${s.id}">${s.id} (${s.n_days} jours)</option>`)
    .join("");
}

function setupSourceToggle() {
  document.querySelectorAll('input[name="source"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const isStore = e.target.value === "store";
      el("store-source").hidden = !isStore;
      el("upload-source").hidden = isStore;
      updateAnalyzeButtonState();
    });
  });
}

function updateAnalyzeButtonState() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source === "store") {
    el("analyze-btn").disabled = !el("store-select").value;
  } else {
    el("analyze-btn").disabled = !(el("date-col").value && el("value-col").value);
  }
}

let locationDebounce = null;

function setupLocationInput() {
  const input = el("location-input");
  const list = el("location-suggestions");

  input.addEventListener("input", (e) => {
    el("exog-toggles").hidden = !e.target.value.trim();

    clearTimeout(locationDebounce);
    const query = e.target.value.trim();
    if (query.length < 3) {
      list.hidden = true;
      list.innerHTML = "";
      return;
    }
    locationDebounce = setTimeout(() => fetchCitySuggestions(query), 250);
  });

  input.addEventListener("blur", () => {
    setTimeout(() => (list.hidden = true), 150); // laisse le clic sur une suggestion se déclencher
  });

  input.addEventListener("focus", () => {
    if (list.children.length) list.hidden = false;
  });
}

async function fetchCitySuggestions(query) {
  const list = el("location-suggestions");
  try {
    const res = await fetch(
      `https://api-adresse.data.gouv.fr/search/?q=${encodeURIComponent(query)}&type=municipality&limit=8`
    );
    if (!res.ok) return;
    const data = await res.json();

    const seen = new Set();
    const items = data.features
      .map((f) => ({ city: f.properties.city, postcode: f.properties.postcode }))
      .filter((item) => {
        const key = `${item.city}-${item.postcode}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

    if (!items.length) {
      list.hidden = true;
      list.innerHTML = "";
      return;
    }

    list.innerHTML = items
      .map(
        (item) =>
          `<li data-value="${item.city} ${item.postcode}">${item.city}<span class="postcode">${item.postcode}</span></li>`
      )
      .join("");
    list.hidden = false;

    list.querySelectorAll("li").forEach((li) => {
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        el("location-input").value = li.dataset.value;
        el("exog-toggles").hidden = false;
        list.hidden = true;
      });
    });
  } catch {
    list.hidden = true;
  }
}

function setupFileInput() {
  el("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const firstLine = text.split(/\r?\n/, 1)[0];
    const columns = firstLine.split(",").map((c) => c.trim());

    const fill = (id) =>
      (el(id).innerHTML = columns.map((c) => `<option value="${c}">${c}</option>`).join(""));
    fill("date-col");
    fill("value-col");
    if (columns.length > 1) el("value-col").value = columns[1];

    el("column-selects").hidden = false;
    updateAnalyzeButtonState();
  });
}

function currentSourceFromUI() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source === "store") {
    return { type: "store", storeId: el("store-select").value };
  }
  return {
    type: "upload",
    file: el("file-input").files[0],
    dateCol: el("date-col").value,
    valueCol: el("value-col").value,
    location: el("location-input").value.trim(),
    useFerie: el("toggle-ferie").checked,
    useVacances: el("toggle-vacances").checked,
    useMeteo: el("toggle-meteo").checked,
  };
}

async function fetchAnalysis(source) {
  if (source.type === "store") {
    const res = await fetch(`/api/stores/${source.storeId}/analysis`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur d'analyse");
    return res.json();
  }
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  if (source.location) {
    form.append("location", source.location);
    form.append("use_ferie", source.useFerie);
    form.append("use_vacances", source.useVacances);
    form.append("use_meteo", source.useMeteo);
  }
  const res = await fetch("/api/upload/analysis", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Erreur d'analyse");
  return res.json();
}

async function fetchForecast(source, horizon) {
  if (source.type === "store") {
    const res = await fetch(`/api/stores/${source.storeId}/forecast?horizon=${horizon}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de prévision");
    return res.json();
  }
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  form.append("horizon", horizon);
  if (source.location) {
    form.append("location", source.location);
    form.append("use_ferie", source.useFerie);
    form.append("use_vacances", source.useVacances);
    form.append("use_meteo", source.useMeteo);
  }
  const res = await fetch("/api/upload/forecast", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Erreur de prévision");
  return res.json();
}

function renderAnalysis(data) {
  lastAnalysisData = data;
  el("results").hidden = false;
  el("period-label").textContent = `${data.dates[0]} → ${data.dates[data.dates.length - 1]} (${data.dates.length} jours)`;

  const accent = cssVar("--accent");
  const accent2 = cssVar("--accent-2");
  renderLineChart("observed-chart", data.dates, [simpleDataset("Observé", data.observed, accent)]);
  renderLineChart("trend-chart", data.dates, [simpleDataset("Tendance", data.trend, accent)]);
  renderLineChart("seasonal-chart", data.dates, [simpleDataset("Saisonnier", data.seasonal, accent)]);
  renderLineChart("event-chart", data.dates, [simpleDataset("Événementiel", data.event, accent)]);
  renderLineChart("resid-chart", data.dates, [simpleDataset("Résidu", data.resid, accent)]);

  el("metric-regularity").textContent = data.regularity_index.toFixed(2);
  el("metric-anomalies").textContent = data.anomalies.length;

  const effectLabels = { ferie: "Effet jour férié", vacances: "Effet vacances scolaires" };
  el("effects-metrics").innerHTML = Object.entries(data.effects)
    .map(([key, value]) => {
      const label = effectLabels[key] || key;
      const display = value === 0 ? "n/a" : `${value > 0 ? "+" : ""}${value}`;
      return `<div class="metric"><span class="metric-value">${display}</span><span class="metric-label">${label}</span></div>`;
    })
    .join("");

  const weatherSection = el("weather-section");
  if (data.weather) {
    weatherSection.hidden = false;
    el("weather-corr").textContent =
      `Corrélation résidu / température : ${data.weather.corr_temp} — ` +
      `résidu / précipitations : ${data.weather.corr_precipitation}`;
    renderLineChart("weather-chart", data.dates, [
      simpleDataset("Température max (°C)", data.weather.temp_max, accent),
      simpleDataset("Précipitations (mm)", data.weather.precipitation, accent2),
    ]);
  } else {
    weatherSection.hidden = true;
  }

  const anomaliesSection = el("anomalies-section");
  const tbody = document.querySelector("#anomalies-table tbody");
  if (data.anomalies.length) {
    anomaliesSection.hidden = false;
    tbody.innerHTML = data.anomalies
      .map((a) => `<tr><td>${a.date}</td><td>${a.resid > 0 ? "+" : ""}${a.resid}</td></tr>`)
      .join("");
  } else {
    anomaliesSection.hidden = true;
  }
}

function renderForecast(data) {
  lastForecastData = data;
  renderLineChart("forecast-chart", data.dates, [
    { ...simpleDataset("IC haut", data.ic_haut, "transparent"), fill: false },
    {
      ...simpleDataset("IC bas", data.ic_bas, "transparent"),
      fill: "-1",
      backgroundColor: cssVar("--chart-band"),
    },
    simpleDataset("Prévision", data.prevision, cssVar("--accent")),
  ]);

  lastForecastRows = data.dates.map((date, i) => ({
    date,
    prevision: data.prevision[i],
    ic_bas: data.ic_bas[i],
    ic_haut: data.ic_haut[i],
  }));
  el("download-btn").hidden = false;
}

function downloadForecastCsv() {
  if (!lastForecastRows) return;
  const header = "date,prevision,ic_bas,ic_haut";
  const lines = lastForecastRows.map((r) => `${r.date},${r.prevision},${r.ic_bas},${r.ic_haut}`);
  const csv = [header, ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "prevision.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function onAnalyze() {
  const status = el("source-status");
  try {
    status.textContent = "Analyse en cours...";
    currentSource = currentSourceFromUI();
    const data = await fetchAnalysis(currentSource);
    renderAnalysis(data);
    status.textContent = "";
  } catch (err) {
    status.textContent = err.message;
  }
}

async function onForecast() {
  const status = el("forecast-status");
  const horizon = parseInt(el("horizon-input").value, 10);
  try {
    status.textContent = "Calcul en cours...";
    const data = await fetchForecast(currentSource, horizon);
    renderForecast(data);
    status.textContent = "";
  } catch (err) {
    status.textContent = err.message;
  }
}

function rerenderCharts() {
  if (lastAnalysisData) renderAnalysis(lastAnalysisData);
  if (lastForecastData) renderForecast(lastForecastData);
}

function updateThemeButtons(value) {
  document.querySelectorAll("#theme-switch button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.themeValue === value);
  });
}

function applyTheme(value) {
  if (value === "system") {
    delete document.documentElement.dataset.theme;
    localStorage.removeItem("cadence-theme");
  } else {
    document.documentElement.dataset.theme = value;
    localStorage.setItem("cadence-theme", value);
  }
  updateThemeButtons(value);
  rerenderCharts();
}

function setupThemeSwitch() {
  const saved = localStorage.getItem("cadence-theme") || "system";
  updateThemeButtons(saved);

  document.querySelectorAll("#theme-switch button").forEach((btn) => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeValue));
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!localStorage.getItem("cadence-theme")) rerenderCharts();
  });
}

function init() {
  setupThemeSwitch();
  setupSourceToggle();
  setupFileInput();
  setupLocationInput();
  el("store-select").addEventListener("change", updateAnalyzeButtonState);
  el("analyze-btn").addEventListener("click", onAnalyze);
  el("forecast-btn").addEventListener("click", onForecast);
  el("download-btn").addEventListener("click", downloadForecastCsv);
  loadStores().then(updateAnalyzeButtonState);
}

init();
