// Page Kaggle : fichiers d'une compétition -> correspondance des colonnes ->
// backtest local ou soumission -> historique des essais. el(), escapeHtml(),
// loadPresets()... viennent de model-params.js ; setupSettingsMenu de menu.js.

const MODEL_LABELS = {
  ensemble: "LightGBM + SARIMAX", lightgbm: "LightGBM", sarimax: "SARIMAX", prophet: "Prophet",
};
const METRIC_LABELS = { rmsle: "RMSLE", rmse: "RMSE", mae: "MAE", smape: "SMAPE" };
const MODEL_HINTS = {
  ensemble:
    "Moyenne à parts égales de LightGBM et SARIMAX : la plus précise à ce jour (Store Sales, Recruit), " +
    "mais aussi lente que SARIMAX, qu'elle ajuste en plus.",
  lightgbm:
    "Un seul modèle entraîné sur toutes les séries à la fois : le plus rapide sur des milliers de séries.",
  sarimax: "Un ajustement par série, en parallèle : compter plusieurs minutes sur des milliers de séries.",
  prophet: "Un ajustement par série, en parallèle : le plus lent des trois sur des milliers de séries.",
};

let inspected = null; // réponse de /api/kaggle/inspect pour les fichiers actuels
let currentRuns = [];

// --- Compétition, mémorisations par navigateur --------------------------

function competitionName() {
  return el("competition-input").value.trim();
}

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // stockage indisponible (navigation privée...) : simple confort perdu
  }
}

function metricKey() {
  return `cadence-kaggle-metric:${competitionName()}`;
}

function selectedMetric() {
  return el("metric-select").value;
}

async function loadCompetitions() {
  const res = await fetch("/api/kaggle/competitions");
  if (!res.ok) return;
  const names = await res.json();
  el("competition-list").innerHTML = names.map((n) => `<option value="${escapeHtml(n)}"></option>`).join("");
  if (!competitionName()) {
    el("competition-input").value = storageGet("cadence-kaggle-last") || names[0] || "";
  }
}

function onCompetitionChange() {
  const name = competitionName();
  storageSet("cadence-kaggle-last", name);
  el("metric-select").value = storageGet(metricKey()) || "rmsle";
  loadRuns();
}

// --- Lecture des colonnes ------------------------------------------------

function optionsHtml(columns, selected, emptyLabel) {
  const opts = emptyLabel ? [`<option value="">${emptyLabel}</option>`] : [];
  for (const c of columns) {
    opts.push(`<option value="${escapeHtml(c)}"${c === selected ? " selected" : ""}>${escapeHtml(c)}</option>`);
  }
  return opts.join("");
}

async function onFilesChange() {
  const train = el("train-file").files[0];
  const test = el("test-file").files[0];
  inspected = null;
  el("mapping-panel").hidden = true;
  el("run-panel").hidden = true;
  if (!train || !test) return;

  const status = el("inspect-status");
  status.className = "status";
  status.textContent = "Lecture des colonnes...";
  const form = new FormData();
  form.append("train_file", train);
  form.append("test_file", test);
  try {
    const res = await fetch("/api/kaggle/inspect", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Fichiers illisibles");
    inspected = await res.json();
  } catch (err) {
    status.className = "status status--error";
    status.textContent = err.message;
    return;
  }

  const { train: tr, test: te } = inspected;
  status.className = "status status--ok";
  status.textContent =
    `Train : ${tr.n_rows.toLocaleString("fr-FR")} lignes, ${tr.columns.length} colonnes — ` +
    `test : ${te.n_rows.toLocaleString("fr-FR")} lignes, ${te.columns.length} colonnes.`;
  fillMapping(inspected.suggestion);
  el("mapping-panel").hidden = false;
  el("run-panel").hidden = false;
}

function fillMapping(s) {
  const trainCols = inspected.train.columns;
  const testCols = inspected.test.columns;
  el("date-col").innerHTML = optionsHtml(trainCols, s.date_col);
  el("target-col").innerHTML = optionsHtml(trainCols, s.target_col);
  el("entity-col-1").innerHTML = optionsHtml(trainCols, s.entity_cols[0]);
  el("entity-col-2").innerHTML = optionsHtml(trainCols, s.entity_cols[1], "— Aucune —");
  el("id-col").innerHTML = optionsHtml(testCols, s.id_col);
  el("test-date-col").innerHTML = optionsHtml(testCols, testCols.includes(s.date_col) ? s.date_col : null);
  el("composite-toggle").checked = Boolean(s.composite_sep);
  el("composite-sep").value = s.composite_sep || "_";
  el("submission-target").value = s.target_col || "";
  // Proposées mais décochées : sur Store Sales, onpromotion gagnait 7 % au
  // backtest récent mais coûtait 0.464 -> 0.532 au score officiel (05/10/2026).
  renderExogChoices(new Set());
  updateCompositeFields();
}

// Candidates : colonnes présentes dans les deux fichiers, hors clés et cible
// (une exogène doit être connue sur les dates du test pour servir).
function renderExogChoices(checked) {
  const used = new Set([el("date-col").value, el("target-col").value, el("entity-col-1").value, el("entity-col-2").value, el("id-col").value]);
  const testCols = new Set(inspected.test.columns);
  const candidates = inspected.train.columns.filter((c) => testCols.has(c) && !used.has(c));
  el("exog-list").innerHTML = candidates
    .map(
      (c) =>
        `<label class="kaggle-check"><input type="checkbox" value="${escapeHtml(c)}"${checked.has(c) ? " checked" : ""} /> ${escapeHtml(c)}</label>`
    )
    .join("");
  el("exog-empty").hidden = candidates.length > 0;
}

function checkedExog() {
  return [...el("exog-list").querySelectorAll("input:checked")].map((i) => i.value);
}

function updateCompositeFields() {
  const composite = el("composite-toggle").checked;
  el("composite-sep-field").hidden = !composite;
  el("test-date-field").hidden = composite;
  el("entity-col-2").disabled = composite;
  if (composite) el("entity-col-2").value = "";
}

// --- Modèle et configuration ---------------------------------------------

// "Défaut" n'envoie aucun réglage : le backend garde alors ses réglages
// automatiques (régularisation LightGBM selon les historiques), qu'un jeu
// de paramètres explicite court-circuiterait.
function refreshConfigSelect() {
  const model = el("model-select").value;
  // L'ensemble n'a pas de configurations enregistrées : réglages par défaut de chaque membre.
  const presets = (loadPresets()[model] || []).filter((p) => !isDefaultPreset(p.id));
  el("model-config").innerHTML = [
    `<option value="">Défaut Cadence (réglages automatiques)</option>`,
    ...presets.map((p) => `<option value="${p.id}">${escapeHtml(p.name)}</option>`),
  ].join("");
  el("model-hint").textContent = MODEL_HINTS[model];
}

function selectedModelParams() {
  const model = el("model-select").value;
  const preset = (loadPresets()[model] || []).find((p) => p.id === el("model-config").value);
  if (!preset) return null;
  const p = preset.params;
  if (model === "sarimax") {
    return { sarimax: { order: [p.p, p.d, p.q], seasonal_order: [p.P, p.D, p.Q, p.s] } };
  }
  return { [model]: p };
}

// --- Lancement d'un essai ------------------------------------------------

function setProgress(fraction) {
  el("run-progress").hidden = fraction === null;
  el("run-progress-fill").style.width = `${Math.round((fraction || 0) * 100)}%`;
}

function pollJob(jobId) {
  const status = el("run-status");
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(`/api/kaggle/progress/${jobId}`);
        if (!res.ok) throw new Error((await res.json()).detail || "Erreur de suivi");
        const p = await res.json();
        if (p.status === "error") return reject(new Error(p.error || "Erreur"));
        if (p.status === "done") return resolve(p.result);
        setProgress(p.total ? p.processed / p.total : 0);
        const step = p.step ? `${p.step} — ` : "";
        status.textContent = p.total
          ? `En cours : ${step}${p.processed}/${p.total} séries...`
          : "Préparation des données...";
        setTimeout(tick, 1000);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

function buildRunForm(mode) {
  const form = new FormData();
  form.append("train_file", el("train-file").files[0]);
  form.append("test_file", el("test-file").files[0]);
  form.append("competition", competitionName());
  form.append("mode", mode);
  form.append("model", el("model-select").value);
  const params = selectedModelParams();
  if (params) form.append("model_params", JSON.stringify(params));
  form.append("date_col", el("date-col").value);
  form.append("target_col", el("target-col").value);
  form.append("entity_cols", [el("entity-col-1").value, el("entity-col-2").value].filter(Boolean).join(","));
  form.append("id_col", el("id-col").value);
  if (el("composite-toggle").checked) {
    form.append("composite_sep", el("composite-sep").value);
  } else {
    form.append("test_date_col", el("test-date-col").value);
  }
  const exog = checkedExog();
  if (exog.length) form.append("exog_cols", exog.join(","));
  form.append("submission_target", el("submission-target").value.trim());
  form.append("zero_dead_series", el("dead-toggle").checked);
  return form;
}

async function downloadSubmission(jobId) {
  const res = await fetch(`/api/kaggle/download/${jobId}`);
  if (!res.ok) throw new Error((await res.json()).detail || "Erreur de téléchargement");
  const match = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") || "");
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = match ? match[1] : "submission.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function runJob(mode) {
  const status = el("run-status");
  status.className = "status";
  if (!competitionName()) {
    status.className = "status status--error";
    status.textContent = "Donne un nom à la compétition (étape 1) : il regroupe ses essais.";
    el("competition-input").focus();
    return;
  }

  el("backtest-btn").disabled = el("submission-btn").disabled = true;
  el("run-result").hidden = true;
  setProgress(0);
  try {
    status.textContent = "Envoi des fichiers...";
    const res = await fetch("/api/kaggle/start", { method: "POST", body: buildRunForm(mode) });
    if (!res.ok) throw new Error((await res.json()).detail || "Erreur de démarrage");
    const { job_id } = await res.json();
    const result = await pollJob(job_id);

    if (mode === "submission") await downloadSubmission(job_id);
    renderResult(result);
    status.className = "status status--ok";
    status.textContent = mode === "submission" ? `Essai #${result.run_id} : fichier de soumission téléchargé.` : `Essai #${result.run_id} enregistré.`;
    await loadCompetitions();
    await loadRuns();
  } catch (err) {
    status.className = "status status--error";
    status.textContent = err.message;
  } finally {
    el("backtest-btn").disabled = el("submission-btn").disabled = false;
    setProgress(null);
  }
}

function gainPct(model, naive) {
  if (!naive) return null;
  return ((naive - model) / naive) * 100;
}

function formatScore(value) {
  return value === null || value === undefined ? "–" : Number(value).toFixed(5);
}

function renderResult(r) {
  const fallback = r.n_fallback
    ? ` ${r.n_fallback.toLocaleString("fr-FR")} ligne(s) sans prévision du modèle (historique trop court) : moyenne récente de la série.`
    : "";
  const gap = r.horizon > r.scored_days ? ` (dont ${r.horizon - r.scored_days} jour(s) d'écart non notés)` : "";

  if (r.mode === "submission") {
    el("run-summary").innerHTML =
      `${r.n_rows.toLocaleString("fr-FR")} prévisions pour ${r.n_entities.toLocaleString("fr-FR")} séries, ` +
      `horizon ${r.horizon} jours${gap}.${fallback} À soumettre sur Kaggle, puis reporter le score dans l'essai <strong>#${r.run_id}</strong> ci-dessous.`;
    el("run-metrics").innerHTML = "";
    el("run-result").hidden = false;
    return;
  }

  const metric = selectedMetric();
  const s = r.scores;
  const gain = gainPct(s.model[metric], s.naive[metric]);
  el("run-summary").innerHTML =
    `Backtest sur les ${r.scored_days} derniers jours connus${gap}, ${r.n_entities.toLocaleString("fr-FR")} séries : ` +
    `<strong>${METRIC_LABELS[metric]} ${formatScore(s.model[metric])}</strong>` +
    (gain === null ? "" : `, ${gain >= 0 ? `${gain.toFixed(1)} % meilleur` : `${(-gain).toFixed(1)} % moins bon`} que le naïf saisonnier`) +
    `.${fallback}`;
  el("run-metrics").innerHTML = Object.keys(METRIC_LABELS)
    .map((m) => {
      const g = gainPct(s.model[m], s.naive[m]);
      const active = m === metric ? " metric--active" : "";
      return (
        `<div class="metric${active}"><span class="metric-value">${formatScore(s.model[m])}</span>` +
        `<span class="metric-label">${METRIC_LABELS[m]} — naïf ${formatScore(s.naive[m])}` +
        (g === null ? "" : ` (${g >= 0 ? "−" : "+"}${Math.abs(g).toFixed(1)} %)`) +
        `</span></div>`
      );
    })
    .join("");
  el("run-result").hidden = false;
}

// --- Historique des essais -----------------------------------------------

function formatDateTime(raw) {
  const [d, t] = raw.split(" ");
  const [y, m, day] = d.split("-");
  return `${day}/${m}/${y}${t ? " " + t.slice(0, 5) : ""}`;
}

async function loadRuns() {
  const name = competitionName();
  el("runs-panel").hidden = !name;
  if (!name) return;
  el("runs-competition").textContent = name;
  const res = await fetch(`/api/kaggle/runs?competition=${encodeURIComponent(name)}`);
  currentRuns = res.ok ? await res.json() : [];
  renderRuns();
}

function bestId(runs, valueOf) {
  let best = null;
  for (const r of runs) {
    const v = valueOf(r);
    if (v === null || v === undefined) continue;
    if (best === null || v < valueOf(best)) best = r;
  }
  return best ? best.id : null;
}

function renderRuns() {
  const metric = selectedMetric();
  el("runs-metric-head").textContent = `Local (${METRIC_LABELS[metric]})`;
  const tbody = el("runs-table").querySelector("tbody");
  if (!currentRuns.length) {
    tbody.innerHTML = `<tr><td colspan="11" class="hint">Aucun essai pour cette compétition.</td></tr>`;
    return;
  }
  const localOf = (r) => (r.scores ? r.scores.model[metric] : null);
  const bestLocal = bestId(currentRuns, localOf);
  const bestPublic = bestId(currentRuns, (r) => r.kaggle_public);
  const bestPrivate = bestId(currentRuns, (r) => r.kaggle_private);

  tbody.innerHTML = currentRuns
    .map((r) => {
      const gain = r.scores ? gainPct(r.scores.model[metric], r.scores.naive[metric]) : null;
      const exog = (r.config.exog_cols || []).join(", ") || "–";
      const preset = r.config.model_params ? " (config perso)" : "";
      const dead = r.config.options && r.config.options.zero_dead_days ? " · éteintes→0" : "";
      const scoreInput = (field, bestRunId) =>
        `<input type="text" inputmode="decimal" class="run-input run-score${r.id === bestRunId ? " is-best" : ""}" ` +
        `data-id="${r.id}" data-field="${field}" value="${r[field] ?? ""}" placeholder="–" />`;
      return (
        `<tr>` +
        `<td>${r.id}</td>` +
        `<td>${formatDateTime(r.created_at)}</td>` +
        `<td><span class="run-mode run-mode--${r.mode}">${r.mode === "backtest" ? "Backtest" : "Soumission"}</span></td>` +
        `<td>${MODEL_LABELS[r.model] || escapeHtml(r.model)}${preset}${dead}</td>` +
        `<td>${escapeHtml(exog)}</td>` +
        `<td class="${r.id === bestLocal ? "is-best" : ""}">${formatScore(localOf(r))}</td>` +
        `<td>${gain === null ? "–" : `${gain >= 0 ? "−" : "+"}${Math.abs(gain).toFixed(1)} %`}</td>` +
        `<td>${scoreInput("kaggle_public", bestPublic)}</td>` +
        `<td>${scoreInput("kaggle_private", bestPrivate)}</td>` +
        `<td><input type="text" class="run-input run-note" data-id="${r.id}" data-field="note" value="${escapeHtml(r.note || "")}" placeholder="–" /></td>` +
        `<td><button type="button" class="run-delete" data-id="${r.id}" title="Supprimer l'essai" aria-label="Supprimer l'essai #${r.id}">✕</button></td>` +
        `</tr>`
      );
    })
    .join("");
}

async function saveRun(runId) {
  const row = (field) => el("runs-table").querySelector(`input[data-id="${runId}"][data-field="${field}"]`).value;
  const form = new FormData();
  form.append("kaggle_public", row("kaggle_public"));
  form.append("kaggle_private", row("kaggle_private"));
  form.append("note", row("note"));
  const status = el("runs-status");
  const res = await fetch(`/api/kaggle/runs/${runId}`, { method: "PATCH", body: form });
  if (!res.ok) {
    status.className = "status status--error";
    status.textContent = (await res.json()).detail || "Enregistrement impossible";
    return;
  }
  const updated = await res.json();
  currentRuns = currentRuns.map((r) => (r.id === updated.id ? updated : r));
  renderRuns();
  status.className = "status status--ok";
  status.textContent = `Essai #${runId} mis à jour.`;
}

async function deleteRun(runId) {
  if (!window.confirm(`Supprimer l'essai #${runId} ?`)) return;
  const res = await fetch(`/api/kaggle/runs/${runId}`, { method: "DELETE" });
  if (res.ok) await loadRuns();
}

// --- Branchements --------------------------------------------------------

setupSettingsMenu();
ensureDefaultPresets();
refreshConfigSelect();

el("competition-input").addEventListener("change", onCompetitionChange);
el("train-file").addEventListener("change", onFilesChange);
el("test-file").addEventListener("change", onFilesChange);
el("composite-toggle").addEventListener("change", updateCompositeFields);
for (const id of ["date-col", "target-col", "entity-col-1", "entity-col-2", "id-col"]) {
  el(id).addEventListener("change", () => renderExogChoices(new Set(checkedExog())));
}
el("target-col").addEventListener("change", () => {
  el("submission-target").value = el("target-col").value;
});
el("model-select").addEventListener("change", refreshConfigSelect);
el("metric-select").addEventListener("change", () => {
  storageSet(metricKey(), selectedMetric());
  renderRuns();
});
el("backtest-btn").addEventListener("click", () => runJob("backtest"));
el("submission-btn").addEventListener("click", () => runJob("submission"));
el("runs-table").addEventListener("change", (e) => {
  if (e.target.classList.contains("run-input")) saveRun(e.target.dataset.id);
});
el("runs-table").addEventListener("click", (e) => {
  const btn = e.target.closest(".run-delete");
  if (btn) deleteRun(btn.dataset.id);
});

loadCompetitions().then(onCompetitionChange);
