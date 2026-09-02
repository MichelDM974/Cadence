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
  prophet: { interval_width: 0.8, changepoint_prior_scale: 0.05, seasonality_prior_scale: 10, seasonality_mode: "additive" },
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

function addPreset(modelName, name, params) {
  const presets = loadPresets();
  presets[modelName].push({ id: crypto.randomUUID(), name, params });
  savePresets(presets);
  return presets;
}

function renamePreset(modelName, id, name) {
  const presets = loadPresets();
  const preset = presets[modelName].find((p) => p.id === id);
  if (preset) preset.name = name;
  savePresets(presets);
  return presets;
}

function deletePreset(modelName, id) {
  const presets = loadPresets();
  presets[modelName] = presets[modelName].filter((p) => p.id !== id);
  savePresets(presets);
  return presets;
}

// Pré-installe une configuration "Défaut Cadence" par modèle au tout premier
// chargement (rien à faire pour avoir un point de départ dans les menus de
// l'étape 4 / génération en masse) — ne revient jamais une fois ce premier
// chargement passé, y compris si l'utilisateur supprime ensuite ce préréglage.
function ensureDefaultPresets() {
  if (localStorage.getItem("cadence-model-presets-seeded")) return;
  const presets = loadPresets();
  for (const modelName of Object.keys(DEFAULT_MODEL_PARAMS)) {
    if (presets[modelName].length === 0) {
      presets[modelName].push({ id: crypto.randomUUID(), name: "Défaut Cadence", params: { ...DEFAULT_MODEL_PARAMS[modelName] } });
    }
  }
  savePresets(presets);
  localStorage.setItem("cadence-model-presets-seeded", "true");
}
ensureDefaultPresets();

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
function modelParamsPayload() {
  const p = loadModelParams();
  return {
    sarimax: { order: [p.sarimax.p, p.sarimax.d, p.sarimax.q], seasonal_order: [p.sarimax.P, p.sarimax.D, p.sarimax.Q, p.sarimax.s] },
    prophet: p.prophet,
    lightgbm: p.lightgbm,
  };
}
