const charts = {};
let currentSource = null; // { type: "store", storeId } | { type: "upload", file, dateCol, valueCol }
let lastForecastRows = null;
let lastAnalysisData = null;
let lastForecastData = null;
let lastForecastModel = null;
let lastBacktestData = null;

const el = (id) => document.getElementById(id);

const MODEL_LABELS = { naive: "Naïve", sarimax: "SARIMAX", prophet: "Prophet" };

const EXOG_LABELS = { ferie: "Jours fériés", vacances: "Vacances scolaires", meteo: "Météo" };
const activeLocationExog = new Set(); // ferie / vacances / meteo — générées par géolocalisation
const activeFileExog = new Set(); // noms de colonnes du fichier importé, utilisées comme régresseurs
let externalExogFile = null; // { file, dateCol, valueCols } | null — un seul fichier CSV externe à la fois
let vacancesAvailable = true; // "vacances scolaires" n'a de source de données que pour la France
let fileExogColumns = []; // colonnes disponibles du fichier importé, hors date/valeur/entité/catégorie

function setVacancesAvailability(available) {
  vacancesAvailable = available;
  if (!available) activeLocationExog.delete("vacances");
  renderExogTags();
  updateExogAddChoices();
}

function remainingLocationKeys() {
  return Object.keys(EXOG_LABELS).filter((key) => (key !== "vacances" || vacancesAvailable) && !activeLocationExog.has(key));
}

function remainingFileExogColumns() {
  const excluded = new Set(
    [el("date-col")?.value, el("value-col")?.value, el("store-col")?.value, el("category-col")?.value].filter(Boolean)
  );
  // une colonne devenue entité/catégorie ne peut plus rester une variable exogène active
  [...activeFileExog].forEach((col) => excluded.has(col) && activeFileExog.delete(col));
  return fileExogColumns.filter((c) => !excluded.has(c) && !activeFileExog.has(c));
}

function renderExogTags() {
  const tags = [
    ...[...activeLocationExog].map((key) => ({
      key: `loc:${key}`,
      label: EXOG_LABELS[key],
      remove: () => activeLocationExog.delete(key),
    })),
    ...[...activeFileExog].map((col) => ({ key: `col:${col}`, label: col, remove: () => activeFileExog.delete(col) })),
  ];
  if (externalExogFile) {
    tags.push({
      key: "ext:file",
      label: `${externalExogFile.file.name} (${externalExogFile.valueCols.join(", ")})`,
      remove: () => {
        externalExogFile = null;
        el("exog-file-input").value = "";
        el("exog-file-selects").hidden = true;
      },
    });
  }

  el("exog-active").innerHTML = tags
    .map(
      (t) =>
        `<span class="exog-tag" data-key="${escapeHtml(t.key)}">${escapeHtml(t.label)} ` +
        `<button type="button" data-remove="${escapeHtml(t.key)}" aria-label="Retirer ${escapeHtml(t.label)}">×</button></span>`
    )
    .join("");

  const removers = Object.fromEntries(tags.map((t) => [t.key, t.remove]));
  el("exog-active")
    .querySelectorAll("button[data-remove]")
    .forEach((btn) => {
      btn.addEventListener("click", () => {
        removers[btn.dataset.remove]?.();
        renderExogTags();
        updateExogAddChoices();
      });
    });

  el("location-row").hidden = activeLocationExog.size === 0;
}

function updateExogAddChoices() {
  const remainingLocation = remainingLocationKeys();
  const remainingCols = remainingFileExogColumns();
  const externalFileAvailable = externalExogFile === null;

  el("exog-location-select").innerHTML = remainingLocation
    .map((key) => `<option value="${key}">${EXOG_LABELS[key]}</option>`)
    .join("");
  el("exog-col-select").innerHTML = remainingCols
    .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
    .join("");

  document.querySelector('[data-exog-type="location"]').hidden = remainingLocation.length === 0;
  document.querySelector('[data-exog-type="external"]').hidden = remainingCols.length === 0 && !externalFileAvailable;
  document.querySelector('[data-exog-subtype="col"]').hidden = remainingCols.length === 0;
  document.querySelector('[data-exog-subtype="file"]').hidden = !externalFileAvailable;

  const anyAvailable = remainingLocation.length > 0 || remainingCols.length > 0 || externalFileAvailable;
  el("exog-add-open-btn").hidden = !anyAvailable;
}

function resetExogAddFlow() {
  el("exog-location-picker").hidden = true;
  el("exog-external-subtype-choice").hidden = true;
  el("exog-col-picker").hidden = true;
  el("exog-file-picker").hidden = true;
  document.querySelectorAll("[data-exog-type], [data-exog-subtype]").forEach((btn) => btn.classList.remove("active"));
}

function closeExogAddFlow() {
  resetExogAddFlow();
  el("exog-add-flow").hidden = true;
}

function setupExogManager() {
  el("exog-add-open-btn").addEventListener("click", () => {
    resetExogAddFlow();
    el("exog-add-flow").hidden = false;
  });
  el("exog-add-cancel-btn").addEventListener("click", closeExogAddFlow);

  // Les choix "type" et "sous-type" restent visibles en permanence tant que
  // le panneau est ouvert (comme des onglets) : on peut changer d'avis et
  // passer d'un type à l'autre sans jamais perdre les autres options.
  document.querySelectorAll("[data-exog-type]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-exog-type]").forEach((b) => b.classList.toggle("active", b === btn));
      el("exog-location-picker").hidden = btn.dataset.exogType !== "location";
      el("exog-external-subtype-choice").hidden = btn.dataset.exogType !== "external";
      if (btn.dataset.exogType !== "external") {
        el("exog-col-picker").hidden = true;
        el("exog-file-picker").hidden = true;
        document.querySelectorAll("[data-exog-subtype]").forEach((b) => b.classList.remove("active"));
      }
    });
  });
  document.querySelectorAll("[data-exog-subtype]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-exog-subtype]").forEach((b) => b.classList.toggle("active", b === btn));
      el("exog-col-picker").hidden = btn.dataset.exogSubtype !== "col";
      el("exog-file-picker").hidden = btn.dataset.exogSubtype !== "file";
    });
  });

  el("exog-location-confirm-btn").addEventListener("click", () => {
    const key = el("exog-location-select").value;
    if (!key) return;
    const wasEmpty = activeLocationExog.size === 0;
    activeLocationExog.add(key);
    renderExogTags();
    updateExogAddChoices();
    closeExogAddFlow();
    if (wasEmpty) el("location-input").focus();
  });

  el("exog-col-confirm-btn").addEventListener("click", () => {
    const col = el("exog-col-select").value;
    if (!col) return;
    activeFileExog.add(col);
    renderExogTags();
    updateExogAddChoices();
    closeExogAddFlow();
  });

  el("exog-file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    const selectsDiv = el("exog-file-selects");
    if (!file) {
      selectsDiv.hidden = true;
      return;
    }
    const text = await file.text();
    const firstLine = text.split(/\r?\n/, 1)[0];
    const delimiter = detectDelimiter(firstLine);
    const columns = firstLine.split(delimiter).map(stripQuotes);

    const fill = (id) =>
      (el(id).innerHTML = columns.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join(""));
    fill("exog-date-col");
    fill("exog-value-cols");
    if (columns.length > 1) el("exog-value-cols").options[1].selected = true;

    selectsDiv.hidden = false;
  });

  el("exog-file-confirm-btn").addEventListener("click", () => {
    const file = el("exog-file-input").files[0];
    const dateCol = el("exog-date-col").value;
    const valueCols = [...el("exog-value-cols").selectedOptions].map((o) => o.value);
    if (!file || !dateCol || !valueCols.length) return;
    externalExogFile = { file, dateCol, valueCols };
    renderExogTags();
    updateExogAddChoices();
    closeExogAddFlow();
  });

  renderExogTags();
  updateExogAddChoices();
}

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
      plugins: {
        legend: {
          display: datasets.length > 1,
          labels: { color: text, usePointStyle: true, pointStyle: "line" },
        },
      },
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

async function loadPoints() {
  const res = await fetch("/api/points");
  const points = await res.json();
  const select = el("point-select");
  select.innerHTML = points.length
    ? points.map((p) => `<option value="${p.id}">${escapeHtml(p.name)} (${p.n_days} jours)</option>`).join("")
    : `<option value="">Aucun point de vente enregistré</option>`;
  updateAnalyzeButtonState();
}

function setupSourceToggle() {
  document.querySelectorAll('input[name="source"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      const source = e.target.value;
      el("store-source").hidden = source !== "store";
      el("upload-source").hidden = source !== "upload";
      el("point-source").hidden = source !== "point";
      updateAnalyzeButtonState();
    });
  });
}

function updateAnalyzeButtonState() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source === "store") {
    el("analyze-btn").disabled = !el("store-select").value;
  } else if (source === "point") {
    el("analyze-btn").disabled = !el("point-select").value;
  } else {
    const hasColumns = el("date-col").value && el("value-col").value;
    const storeCol = el("store-col").value;
    const hasStore = !storeCol || el("store-value-select").value;
    const categoryCol = el("category-col").value;
    const hasCategory = !categoryCol || el("category-value-select").value;
    el("analyze-btn").disabled = !(hasColumns && hasStore && hasCategory);
  }
  updateBulkRowVisibility();
}

function updateBulkRowVisibility() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source !== "upload") return;
  const storeCol = el("store-col").value;
  const categoryCol = el("category-col").value;
  const storeFree = storeCol && !el("store-value-select").value;
  const categoryFree = categoryCol && !el("category-value-select").value;
  el("bulk-forecast-row").hidden = !(storeFree || categoryFree);
  updateBulkRowCount();
}

let bulkCountDebounce = null;
let bulkCountRequestId = 0;

function updateBulkRowCount() {
  clearTimeout(bulkCountDebounce);
  bulkCountDebounce = setTimeout(fetchBulkRowCount, 250);
}

async function fetchBulkRowCount() {
  const countEl = el("bulk-row-count");
  if (el("bulk-forecast-row").hidden) {
    countEl.textContent = "";
    return;
  }

  const source = currentSourceFromUI();
  const horizon = parseInt(el("bulk-horizon-input").value, 10);
  if (!source.file || !(source.storeCol || source.categoryCol) || !horizon) {
    countEl.textContent = "";
    return;
  }

  const requestId = ++bulkCountRequestId;
  countEl.textContent = "(calcul en cours...)";
  try {
    const form = new FormData();
    form.append("file", source.file);
    appendStoreFields(form, source);
    const res = await fetch("/api/upload/forecast/bulk/count", { method: "POST", body: form });
    if (!res.ok) throw new Error();
    const { n_entities } = await res.json();
    if (requestId !== bulkCountRequestId) return; // une saisie plus récente a déjà démarré une requête
    const nRows = n_entities * horizon;
    countEl.textContent = `(${n_entities} entités × ${horizon} jours = ${nRows.toLocaleString("fr-FR")} lignes)`;
  } catch {
    if (requestId === bulkCountRequestId) countEl.textContent = "";
  }
}

let locationDebounce = null;

function setupLocationInput() {
  const input = el("location-input");
  const list = el("location-suggestions");

  input.addEventListener("input", (e) => {
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
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=8&language=fr&format=json`
    );
    if (!res.ok) return;
    const data = await res.json();
    const items = data.results || [];

    if (!items.length) {
      list.hidden = true;
      list.innerHTML = "";
      return;
    }

    list.innerHTML = items
      .map((item) => {
        const detail = [item.admin1, item.country].filter(Boolean).join(", ");
        const value = [item.name, item.admin1, item.country].filter(Boolean).join(", ");
        return (
          `<li data-value="${escapeHtml(value)}" data-country="${escapeHtml(item.country_code || "")}">` +
          `${escapeHtml(item.name)}<span class="detail">${escapeHtml(detail)}</span></li>`
        );
      })
      .join("");
    list.hidden = false;

    list.querySelectorAll("li").forEach((li) => {
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();
        el("location-input").value = li.dataset.value;
        setVacancesAvailability(li.dataset.country === "FR");
        list.hidden = true;
      });
    });
  } catch {
    list.hidden = true;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function stripQuotes(field) {
  const trimmed = field.trim();
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1).replace(/""/g, '"'); // dé-échappe les guillemets doublés (RFC 4180)
  }
  return trimmed;
}

function detectDelimiter(headerLine) {
  const candidates = [",", ";", "\t"];
  let best = ",";
  let bestCount = 0;
  for (const d of candidates) {
    const count = headerLine.split(d).length - 1;
    if (count > bestCount) {
      bestCount = count;
      best = d;
    }
  }
  return best;
}

function setupFileInput() {
  el("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const firstLine = text.split(/\r?\n/, 1)[0];
    const delimiter = detectDelimiter(firstLine);
    const columns = firstLine.split(delimiter).map(stripQuotes);

    const fill = (id) =>
      (el(id).innerHTML = columns.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join(""));
    fill("date-col");
    fill("value-col");
    if (columns.length > 1) el("value-col").value = columns[1];

    const optionalFill = (id, placeholder) =>
      (el(id).innerHTML =
        `<option value="">${placeholder}</option>` +
        columns.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join(""));
    optionalFill("store-col", "— Pas de distinction (toutes les lignes ensemble) —");
    optionalFill("category-col", "— Aucune 2e dimension —");
    el("store-picker-row").hidden = true;
    el("category-picker-row").hidden = true;
    el("bulk-forecast-row").hidden = true;
    el("store-value-select").innerHTML = "";
    el("category-value-select").innerHTML = "";
    el("store-picker-status").textContent = "";
    el("category-picker-status").textContent = "";

    fileExogColumns = columns;
    activeFileExog.clear(); // colonnes de l'ancien fichier, plus valides
    renderExogTags();
    updateExogAddChoices();

    el("column-selects").hidden = false;
    el("exog-manager").hidden = false;
    el("save-point-row").hidden = false;
    updateAnalyzeButtonState();
    detectColumns();
  });

  el("date-col").addEventListener("change", () => {
    detectColumns();
    renderExogTags();
    updateExogAddChoices();
  });
  el("value-col").addEventListener("change", () => {
    detectColumns();
    renderExogTags();
    updateExogAddChoices();
  });
  el("store-col").addEventListener("change", () => onEntityColChange("store"));
  el("category-col").addEventListener("change", () => onEntityColChange("category"));
  el("store-value-select").addEventListener("change", updateAnalyzeButtonState);
  el("category-value-select").addEventListener("change", updateAnalyzeButtonState);
}

async function detectColumns() {
  const status = el("columns-detect-status");
  const file = el("file-input").files[0];
  const dateCol = el("date-col").value;
  const valueCol = el("value-col").value;
  if (!file || !dateCol || !valueCol) {
    status.textContent = "";
    return;
  }

  status.textContent = "Détection des colonnes...";
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("date_col", dateCol);
    form.append("value_col", valueCol);
    const res = await fetch("/api/upload/columns/detect", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de détection");
    const { candidates } = await res.json();

    if (!candidates.length) {
      status.textContent = "";
      return;
    }
    const summary = candidates.map((c) => `${escapeHtml(c.column)} (${c.n_unique} valeurs)`).join(", ");
    status.textContent =
      candidates.length > 1
        ? `Ce fichier semble contenir plusieurs dimensions : ${summary}. Utilise les sélecteurs ci-dessous pour isoler une entité/catégorie précise.`
        : `Colonne candidate pour distinguer des entités : ${summary}.`;
  } catch (err) {
    status.textContent = "";
  }
}

async function onEntityColChange(kind) {
  const col = el(`${kind}-col`).value;
  const pickerRow = el(`${kind}-picker-row`);
  const valueSelect = el(`${kind}-value-select`);
  const status = el(`${kind}-picker-status`);

  pickerRow.hidden = !col;
  valueSelect.innerHTML = "";
  renderExogTags(); // cette colonne ne doit plus être proposée comme variable exogène
  updateExogAddChoices();

  if (!col) {
    status.textContent = "";
    updateAnalyzeButtonState();
    return;
  }

  status.textContent = "Chargement des valeurs...";
  try {
    const form = new FormData();
    form.append("file", el("file-input").files[0]);
    form.append("store_col", col);
    const res = await fetch("/api/upload/store-ids", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de lecture des valeurs");
    const { store_ids } = await res.json();
    valueSelect.innerHTML =
      `<option value="">— Toutes (prévision groupée) —</option>` +
      store_ids.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
    status.textContent = `${store_ids.length} valeurs détectées.`;
  } catch (err) {
    status.textContent = err.message;
  }
  updateAnalyzeButtonState();
}

function currentSourceFromUI() {
  const source = document.querySelector('input[name="source"]:checked').value;
  if (source === "store") {
    return { type: "store", storeId: el("store-select").value };
  }
  if (source === "point") {
    return { type: "point", pointId: el("point-select").value };
  }
  return {
    type: "upload",
    file: el("file-input").files[0],
    dateCol: el("date-col").value,
    valueCol: el("value-col").value,
    storeCol: el("store-col").value,
    storeValue: el("store-value-select").value,
    categoryCol: el("category-col").value,
    categoryValue: el("category-value-select").value,
    extraExogCols: [...activeFileExog],
    exogFile: externalExogFile,
    location: el("location-input").value.trim(),
    useFerie: activeLocationExog.has("ferie"),
    useVacances: activeLocationExog.has("vacances"),
    useMeteo: activeLocationExog.has("meteo"),
  };
}

function appendStoreFields(form, source) {
  if (source.storeCol) {
    form.append("store_col", source.storeCol);
    form.append("store_value", source.storeValue || "");
  }
  if (source.categoryCol) {
    form.append("category_col", source.categoryCol);
    form.append("category_value", source.categoryValue || "");
  }
}

// N'est utile que pour /forecast, /backtest et le bulk forecast (le seul
// endroit où ces variables servent de régresseurs) — pas pour /analysis ni
// l'enregistrement d'un point de vente.
function appendExogFileFields(form, source) {
  if (source.extraExogCols && source.extraExogCols.length) {
    form.append("extra_exog_cols", source.extraExogCols.join(","));
  }
  if (source.exogFile) {
    form.append("exog_file", source.exogFile.file);
    form.append("exog_date_col", source.exogFile.dateCol);
    form.append("exog_value_cols", source.exogFile.valueCols.join(","));
  }
}

async function fetchAnalysis(source) {
  if (source.type === "store") {
    const res = await fetch(`/api/stores/${source.storeId}/analysis`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur d'analyse");
    return res.json();
  }
  if (source.type === "point") {
    const res = await fetch(`/api/points/${source.pointId}/analysis`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur d'analyse");
    return res.json();
  }
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  appendStoreFields(form, source);
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

async function fetchForecast(source, horizon, model) {
  if (source.type === "store") {
    const res = await fetch(`/api/stores/${source.storeId}/forecast?horizon=${horizon}&model=${model}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de prévision");
    return res.json();
  }
  if (source.type === "point") {
    const res = await fetch(`/api/points/${source.pointId}/forecast?horizon=${horizon}&model=${model}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de prévision");
    return res.json();
  }
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  form.append("horizon", horizon);
  form.append("model", model);
  appendStoreFields(form, source);
  appendExogFileFields(form, source);
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

async function fetchBacktest(source, horizon) {
  if (source.type === "store") {
    const res = await fetch(`/api/stores/${source.storeId}/backtest?horizon=${horizon}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de comparaison");
    return res.json();
  }
  if (source.type === "point") {
    const res = await fetch(`/api/points/${source.pointId}/backtest?horizon=${horizon}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de comparaison");
    return res.json();
  }
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  form.append("horizon", horizon);
  appendStoreFields(form, source);
  appendExogFileFields(form, source);
  if (source.location) {
    form.append("location", source.location);
    form.append("use_ferie", source.useFerie);
    form.append("use_vacances", source.useVacances);
    form.append("use_meteo", source.useMeteo);
  }
  const res = await fetch("/api/upload/backtest", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Erreur de comparaison");
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

function renderForecast(data, model) {
  lastForecastData = data;
  const datasets = [];
  if (el("download-ci-toggle").checked) {
    datasets.push({ ...simpleDataset("IC haut", data.ic_haut, "transparent"), fill: false });
    datasets.push({
      ...simpleDataset("IC bas", data.ic_bas, "transparent"),
      fill: "-1",
      backgroundColor: cssVar("--chart-band"),
    });
  }
  datasets.push(simpleDataset(`Prévision (${MODEL_LABELS[model] || model})`, data.prevision, cssVar("--accent")));
  renderLineChart("forecast-chart", data.dates, datasets);

  lastForecastRows = data.dates.map((date, i) => ({
    date,
    prevision: data.prevision[i],
    ic_bas: data.ic_bas[i],
    ic_haut: data.ic_haut[i],
  }));
  el("download-btn").hidden = false;
}

function renderBacktest(data) {
  lastBacktestData = data;
  el("backtest-result").hidden = false;

  const unit = data.unit_label || "unités";
  const compared = Object.keys(data.improvement_pct); // ex. ["sarimax", "prophet"]
  const best = compared.reduce((a, b) => (data.mae[b] < data.mae[a] ? b : a));
  const bestPct = data.improvement_pct[best];
  const better = bestPct >= 0;
  const pct = Math.abs(bestPct).toFixed(1);
  const bestLabel = MODEL_LABELS[best] || best;

  el("backtest-summary").innerHTML = better
    ? `Sur les ${data.dates.length} derniers jours connus, <strong>${bestLabel}</strong> est le modèle le plus précis pour cette activité — ` +
      `<strong>${pct}% plus précis</strong> qu'une estimation naïve ("même jour la semaine dernière").`
    : `Sur les ${data.dates.length} derniers jours connus, aucun modèle ne bat l'estimation naïve pour cette activité ` +
      `(meilleur : ${bestLabel}, <strong>${pct}% moins bien</strong> que la naïve).`;

  el("backtest-metrics").innerHTML = Object.keys(data.mae)
    .map((name) => {
      const vsNaive = data.improvement_pct[name];
      const suffix = vsNaive === undefined ? "" : ` (${vsNaive >= 0 ? "+" : ""}${vsNaive}% vs naïve)`;
      return (
        `<div class="metric"><span class="metric-value">${data.mae[name]}</span>` +
        `<span class="metric-label">${MODEL_LABELS[name] || name} — MAE en ${unit}${suffix}</span></div>`
      );
    })
    .join("");

  const colors = { naive: cssVar("--accent-2"), sarimax: cssVar("--accent"), prophet: cssVar("--accent-3") };
  const datasets = [simpleDataset("Réel", data.actual, cssVar("--text"))];
  for (const [name, values] of Object.entries(data.predictions)) {
    datasets.push(simpleDataset(MODEL_LABELS[name] || name, values, colors[name] || cssVar("--muted")));
  }
  renderLineChart("backtest-chart", data.dates, datasets);
}

function downloadForecastCsv() {
  if (!lastForecastRows) return;
  const includeCi = el("download-ci-toggle").checked;
  const header = includeCi ? "date,prevision,ic_bas,ic_haut" : "date,prevision";
  const lines = lastForecastRows.map((r) =>
    includeCi ? `${r.date},${r.prevision},${r.ic_bas},${r.ic_haut}` : `${r.date},${r.prevision}`
  );
  const csv = [header, ...lines].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "prevision.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function resetForecastAndBacktest() {
  lastForecastData = null;
  lastForecastModel = null;
  lastForecastRows = null;
  el("download-btn").hidden = true;
  if (charts["forecast-chart"]) {
    charts["forecast-chart"].destroy();
    delete charts["forecast-chart"];
  }
  el("forecast-status").textContent = "";

  lastBacktestData = null;
  el("backtest-result").hidden = true;
  if (charts["backtest-chart"]) {
    charts["backtest-chart"].destroy();
    delete charts["backtest-chart"];
  }
  el("backtest-status").textContent = "";
}

async function onAnalyze() {
  const status = el("source-status");
  resetForecastAndBacktest(); // une nouvelle source rend l'ancienne prévision/comparaison obsolète
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
  const model = el("model-select").value;
  try {
    status.textContent = "Calcul en cours...";
    const data = await fetchForecast(currentSource, horizon, model);
    lastForecastModel = model;
    renderForecast(data, model);
    status.textContent = "";
  } catch (err) {
    status.textContent = err.message;
  }
}

async function onBacktest() {
  const status = el("backtest-status");
  const horizon = parseInt(el("horizon-input").value, 10);
  lastBacktestData = null;
  el("backtest-result").hidden = true; // efface l'ancien résultat avant d'en calculer un nouveau
  try {
    status.textContent = "Comparaison en cours...";
    const data = await fetchBacktest(currentSource, horizon);
    renderBacktest(data);
    status.textContent = "";
  } catch (err) {
    status.textContent = err.message;
  }
}

async function onSavePoint() {
  const status = el("save-point-status");
  const name = el("point-name-input").value.trim();
  if (!name) {
    status.textContent = "Indique un nom pour ce point de vente.";
    return;
  }

  const source = currentSourceFromUI();
  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  form.append("name", name);
  appendStoreFields(form, source);
  if (source.location) {
    form.append("location", source.location);
    form.append("use_ferie", source.useFerie);
    form.append("use_vacances", source.useVacances);
    form.append("use_meteo", source.useMeteo);
  }

  try {
    status.textContent = "Enregistrement en cours...";
    const res = await fetch("/api/points", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur d'enregistrement");
    await loadPoints();
    status.textContent = `"${name}" enregistré.`;
  } catch (err) {
    status.textContent = err.message;
  }
}

function setBulkProgress(fraction) {
  const bar = el("bulk-progress-bar");
  const fill = el("bulk-progress-fill");
  bar.hidden = fraction === null;
  fill.style.width = `${Math.round((fraction || 0) * 100)}%`;
}

function pollBulkProgress(jobId) {
  const status = el("bulk-forecast-status");
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(`/api/upload/forecast/bulk/progress/${jobId}`);
        if (!res.ok) throw new Error((await res.json()).detail || "Erreur de suivi");
        const progress = await res.json();

        if (progress.status === "error") {
          reject(new Error(progress.error || "Erreur de génération"));
          return;
        }

        setBulkProgress(progress.total ? progress.processed / progress.total : 0);
        status.textContent = `Génération en cours : ${progress.processed}/${progress.total} entités...`;

        if (progress.status === "done") {
          resolve(progress);
          return;
        }
        setTimeout(tick, 1000);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

async function onBulkForecast() {
  const status = el("bulk-forecast-status");
  const source = currentSourceFromUI();
  if (!source.storeCol && !source.categoryCol) return;

  const form = new FormData();
  form.append("file", source.file);
  form.append("date_col", source.dateCol);
  form.append("value_col", source.valueCol);
  appendStoreFields(form, source);
  appendExogFileFields(form, source);
  form.append("horizon", el("bulk-horizon-input").value);
  form.append("model", el("bulk-model-select").value);
  form.append("include_ci", el("bulk-ci-toggle").checked);
  if (source.location) {
    form.append("location", source.location);
    form.append("use_ferie", source.useFerie);
    form.append("use_vacances", source.useVacances);
    form.append("use_meteo", source.useMeteo);
  }

  el("bulk-forecast-btn").disabled = true;
  setBulkProgress(0);
  try {
    status.textContent = "Démarrage de la génération...";
    const startRes = await fetch("/api/upload/forecast/bulk/start", { method: "POST", body: form });
    if (!startRes.ok) throw new Error((await startRes.json()).detail || "Erreur de démarrage");
    const { job_id } = await startRes.json();

    const progress = await pollBulkProgress(job_id);

    const res = await fetch(`/api/upload/forecast/bulk/download/${job_id}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de téléchargement");

    const skipped = progress.skipped || 0;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "previsions_toutes_entites.csv";
    a.click();
    URL.revokeObjectURL(url);

    status.textContent =
      `Terminé : ${progress.total - skipped} entités prévues` +
      (skipped > 0 ? `, ${skipped} ignorés (historique trop court).` : ".");
  } catch (err) {
    status.textContent = err.message;
  } finally {
    el("bulk-forecast-btn").disabled = false;
    setBulkProgress(null);
  }
}

async function onDeletePoint() {
  const select = el("point-select");
  const pointId = select.value;
  if (!pointId) return;
  const label = select.options[select.selectedIndex]?.textContent || "ce point de vente";
  if (!confirm(`Supprimer ${label} ? Cette action est irréversible.`)) return;

  await fetch(`/api/points/${pointId}`, { method: "DELETE" });
  await loadPoints();
}

function rerenderCharts() {
  if (lastAnalysisData) renderAnalysis(lastAnalysisData);
  if (lastForecastData) renderForecast(lastForecastData, lastForecastModel);
  if (lastBacktestData) renderBacktest(lastBacktestData);
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
  setupExogManager();
  el("store-select").addEventListener("change", updateAnalyzeButtonState);
  el("point-select").addEventListener("change", updateAnalyzeButtonState);
  el("analyze-btn").addEventListener("click", onAnalyze);
  el("forecast-btn").addEventListener("click", onForecast);
  el("download-btn").addEventListener("click", downloadForecastCsv);
  el("download-ci-toggle").addEventListener("change", () => {
    if (lastForecastData) renderForecast(lastForecastData, lastForecastModel);
  });
  el("backtest-btn").addEventListener("click", onBacktest);
  el("save-point-btn").addEventListener("click", onSavePoint);
  el("bulk-forecast-btn").addEventListener("click", onBulkForecast);
  el("bulk-horizon-input").addEventListener("input", updateBulkRowCount);
  el("delete-point-btn").addEventListener("click", onDeletePoint);
  loadStores().then(updateAnalyzeButtonState);
  loadPoints();
}

init();
