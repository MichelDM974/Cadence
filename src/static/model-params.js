// Partagé entre index.html (pour construire le payload envoyé au serveur à
// chaque prévision/backtest/génération en masse) et models.html (formulaire
// d'édition) — chargé avant les deux, avant app.js/models.js.

const el = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Mêmes valeurs par défaut que le backend (src/api.py forecast_sarimax/
// forecast_prophet, src/lightgbm_forecast.py LGB_PARAMS/NUM_BOOST_ROUND) —
// à garder synchronisées si ces défauts changent côté Python.
const DEFAULT_MODEL_PARAMS = {
  sarimax: { p: 1, d: 1, q: 1, P: 1, D: 1, Q: 1, s: 7 },
  prophet: { interval_width: 0.8, changepoint_prior_scale: 0.05, seasonality_prior_scale: 10, seasonality_mode: "multiplicative" },
  lightgbm: { learning_rate: 0.05, num_leaves: 63, min_data_in_leaf: 50, num_boost_round: 300 },
};

const MODEL_PARAM_FIELDS = {
  sarimax: [
    ["mp-sarimax-p", "p", "int"],
    ["mp-sarimax-d", "d", "int"],
    ["mp-sarimax-q", "q", "int"],
    ["mp-sarimax-bigp", "P", "int"],
    ["mp-sarimax-bigd", "D", "int"],
    ["mp-sarimax-bigq", "Q", "int"],
    ["mp-sarimax-s", "s", "int"],
  ],
  prophet: [
    ["mp-prophet-interval", "interval_width", "float"],
    ["mp-prophet-cps", "changepoint_prior_scale", "float"],
    ["mp-prophet-sps", "seasonality_prior_scale", "float"],
    ["mp-prophet-mode", "seasonality_mode", "string"],
  ],
  lightgbm: [
    ["mp-lgb-lr", "learning_rate", "float"],
    ["mp-lgb-leaves", "num_leaves", "int"],
    ["mp-lgb-mindata", "min_data_in_leaf", "int"],
    ["mp-lgb-rounds", "num_boost_round", "int"],
  ],
};

function loadModelParams() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem("cadence-model-params") || "{}");
  } catch {
    saved = {};
  }
  return {
    sarimax: { ...DEFAULT_MODEL_PARAMS.sarimax, ...saved.sarimax },
    prophet: { ...DEFAULT_MODEL_PARAMS.prophet, ...saved.prophet },
    lightgbm: { ...DEFAULT_MODEL_PARAMS.lightgbm, ...saved.lightgbm },
  };
}

function saveModelParams(params) {
  localStorage.setItem("cadence-model-params", JSON.stringify(params));
}

// Configurations nommées sauvegardées par l'utilisateur (indépendantes des
// valeurs actives ci-dessus) — un tableau de {id, name, params} par modèle.
const PRESETS_STORAGE_KEY = "cadence-model-presets";

function loadPresets() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(PRESETS_STORAGE_KEY) || "{}");
  } catch {
    saved = {};
  }
  return {
    sarimax: Array.isArray(saved.sarimax) ? saved.sarimax : [],
    prophet: Array.isArray(saved.prophet) ? saved.prophet : [],
    lightgbm: Array.isArray(saved.lightgbm) ? saved.lightgbm : [],
  };
}

function savePresets(presets) {
  localStorage.setItem(PRESETS_STORAGE_KEY, JSON.stringify(presets));
}

// Ne crée rien si un préréglage du même nom (insensible à la casse/espaces)
// existe déjà pour ce modèle — renvoie ce préréglage existant à la place, et
// `created: false` pour que l'appelant affiche un message plutôt qu'un doublon.
function addPreset(modelName, name, params) {
  const presets = loadPresets();
  const trimmed = name.trim();
  const existing = presets[modelName].find((p) => p.name.trim().toLowerCase() === trimmed.toLowerCase());
  if (existing) return { presets, created: false, preset: existing };

  const preset = { id: crypto.randomUUID(), name: trimmed, params };
  presets[modelName].push(preset);
  savePresets(presets);
  return { presets, created: true, preset };
}

// Id réservé et fixe (pas d'UUID) pour le préréglage "Défaut" pré-installé
// ci-dessous : verrouillé en écriture (ni renommable, ni supprimable) pour
// toujours garder un point de départ connu dans les menus de configuration.
const DEFAULT_PRESET_ID = "default";

function isDefaultPreset(id) {
  return id === DEFAULT_PRESET_ID;
}

function renamePreset(modelName, id, name) {
  if (isDefaultPreset(id)) return loadPresets();
  const presets = loadPresets();
  const preset = presets[modelName].find((p) => p.id === id);
  if (preset) preset.name = name;
  savePresets(presets);
  return presets;
}

function deletePreset(modelName, id) {
  if (isDefaultPreset(id)) return loadPresets();
  const presets = loadPresets();
  presets[modelName] = presets[modelName].filter((p) => p.id !== id);
  savePresets(presets);
  return presets;
}

// Pré-installe une configuration "Défaut" par modèle au tout premier
// chargement (rien à faire pour avoir un point de départ dans les menus de
// l'étape 4 / génération en masse). Idempotent : sur les navigateurs qui
// avaient déjà l'ancien préréglage "Défaut Cadence" (id aléatoire, non
// verrouillé), le migre en place vers l'id/nom réservés ci-dessus.
function ensureDefaultPresets() {
  const presets = loadPresets();
  let changed = false;
  for (const modelName of Object.keys(DEFAULT_MODEL_PARAMS)) {
    const list = presets[modelName];
    if (list.some((p) => p.id === DEFAULT_PRESET_ID)) continue;
    const legacy = list.find((p) => p.name === "Défaut Cadence");
    if (legacy) {
      legacy.id = DEFAULT_PRESET_ID;
      legacy.name = "Défaut";
    } else {
      list.unshift({ id: DEFAULT_PRESET_ID, name: "Défaut", params: { ...DEFAULT_MODEL_PARAMS[modelName] } });
    }
    changed = true;
  }
  if (changed) savePresets(presets);
}

// <option> d'un menu de sélection de configuration pour `modelName` — utilisé
// à l'étape 4 et dans la génération en masse pour appliquer un préréglage
// sans passer par la page Modèles.
function presetSelectOptionsHtml(modelName) {
  const options = [`<option value="">Réglages actuels</option>`];
  for (const preset of loadPresets()[modelName]) {
    options.push(`<option value="${preset.id}">${escapeHtml(preset.name)}</option>`);
  }
  return options.join("");
}

function populateModelParamsInputs(params) {
  for (const [modelName, fields] of Object.entries(MODEL_PARAM_FIELDS)) {
    for (const [id, key] of fields) {
      el(id).value = params[modelName][key];
    }
  }
}

function readModelParamsFromInputs() {
  const params = {};
  for (const [modelName, fields] of Object.entries(MODEL_PARAM_FIELDS)) {
    params[modelName] = {};
    for (const [id, key, type] of fields) {
      const raw = el(id).value;
      params[modelName][key] = type === "int" ? parseInt(raw, 10) : type === "float" ? parseFloat(raw) : raw;
    }
  }
  return params;
}

// Payload envoyé au serveur : reconstruit order/seasonal_order pour SARIMAX
// à partir des 7 champs séparés (p,d,q,P,D,Q,s) affichés dans le formulaire.
// Onglets repliables générique (ex. "Ruptures de rythme" sur index.html,
// "Configurations" par modèle sur models.html) : un bouton `.collapsible-tab`
// avec `aria-controls` pointant vers le panneau à révéler.
function setupCollapsibleTabs() {
  document.querySelectorAll(".collapsible-tab").forEach((tab) => {
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    tab.addEventListener("click", () => {
      const expanded = panel.hidden; // état après bascule
      panel.hidden = !expanded;
      tab.setAttribute("aria-expanded", String(expanded));
      tab.querySelector(".collapsible-tab-icon").textContent = expanded ? "▾" : "▸";
    });
  });
}

function modelParamsPayload() {
  const p = loadModelParams();
  return {
    sarimax: { order: [p.sarimax.p, p.sarimax.d, p.sarimax.q], seasonal_order: [p.sarimax.P, p.sarimax.D, p.sarimax.Q, p.sarimax.s] },
    prophet: p.prophet,
    lightgbm: p.lightgbm,
  };
}
