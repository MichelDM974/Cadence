// Partagé entre index.html (pour construire le payload envoyé au serveur à
// chaque prévision/backtest/génération en masse) et models.html (formulaire
// d'édition) — chargé avant les deux, avant app.js/models.js.

const el = (id) => document.getElementById(id);

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
