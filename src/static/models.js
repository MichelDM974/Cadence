// Page dédiée aux hyperparamètres des modèles (models.html). Les fonctions
// utilisées ici (el, loadModelParams, populateModelParamsInputs, ...)
// viennent de model-params.js, chargé avant ce fichier.

function setupModelParamsForm() {
  populateModelParamsInputs(loadModelParams());

  for (const fields of Object.values(MODEL_PARAM_FIELDS)) {
    for (const [id] of fields) {
      el(id).addEventListener("change", () => saveModelParams(readModelParamsFromInputs()));
    }
  }

  document.querySelectorAll(".model-params-reset-btn").forEach((resetBtn) => {
    resetBtn.addEventListener("click", () => {
      const modelName = resetBtn.dataset.model;
      const current = loadModelParams();
      current[modelName] = { ...DEFAULT_MODEL_PARAMS[modelName] };
      saveModelParams(current);
      populateModelParamsInputs(current);
    });
  });
}

setupModelParamsForm();
