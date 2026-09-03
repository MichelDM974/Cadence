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

function renderPresetList(modelName) {
  const manager = document.querySelector(`.preset-manager[data-model="${modelName}"]`);
  const list = manager.querySelector(".preset-list");
  const presets = loadPresets()[modelName];

  list.innerHTML = presets.length
    ? presets
        .map(
          (preset) => `
        <li class="preset-item" data-id="${preset.id}">
          <span class="preset-name">${escapeHtml(preset.name)}</span>
          <span class="preset-actions">
            <button type="button" class="preset-apply-btn" title="Charger cette configuration">Charger</button>
            ${
              isDefaultPreset(preset.id)
                ? ""
                : `<button type="button" class="preset-rename-btn" title="Renommer">✎</button>
            <button type="button" class="preset-delete-btn" title="Supprimer">✕</button>`
            }
          </span>
        </li>`
        )
        .join("")
    : `<li class="hint preset-empty">Aucune configuration enregistrée pour l'instant.</li>`;
}

function setupPresetManagers() {
  document.querySelectorAll(".preset-manager").forEach((manager) => {
    const modelName = manager.dataset.model;
    renderPresetList(modelName);

    const input = manager.querySelector(".preset-name-input");
    const status = manager.querySelector(".preset-save-status");

    input.addEventListener("input", () => {
      status.textContent = "";
    });

    manager.querySelector(".preset-save-btn").addEventListener("click", () => {
      const name = input.value.trim();
      if (!name) return;
      const { created } = addPreset(modelName, name, loadModelParams()[modelName]);
      if (!created) {
        status.textContent = "Cette configuration existe déjà.";
        status.className = "status preset-save-status status--error";
        return;
      }
      status.textContent = "";
      status.className = "status preset-save-status";
      input.value = "";
      renderPresetList(modelName);
    });

    manager.querySelector(".preset-list").addEventListener("click", (e) => {
      const item = e.target.closest(".preset-item");
      if (!item) return;
      const id = item.dataset.id;

      if (e.target.classList.contains("preset-apply-btn")) {
        const preset = loadPresets()[modelName].find((p) => p.id === id);
        if (!preset) return;
        const current = loadModelParams();
        current[modelName] = { ...preset.params };
        saveModelParams(current);
        populateModelParamsInputs(current);
      } else if (e.target.classList.contains("preset-delete-btn")) {
        deletePreset(modelName, id);
        renderPresetList(modelName);
      } else if (e.target.classList.contains("preset-rename-btn")) {
        const nameSpan = item.querySelector(".preset-name");
        const currentName = nameSpan.textContent;
        const editInput = document.createElement("input");
        editInput.type = "text";
        editInput.className = "preset-rename-input";
        editInput.maxLength = 60;
        editInput.value = currentName;
        nameSpan.replaceWith(editInput);
        editInput.focus();
        editInput.select();

        const commit = () => {
          const newName = editInput.value.trim() || currentName;
          renamePreset(modelName, id, newName);
          renderPresetList(modelName);
        };
        editInput.addEventListener("blur", commit);
        editInput.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") editInput.blur();
          if (ev.key === "Escape") {
            editInput.removeEventListener("blur", commit);
            renderPresetList(modelName);
          }
        });
      }
    });
  });
}

// Petit bouton ⚙ sur l'encadré de formule : révèle le lexique des symboles,
// masqué par défaut pour ne pas surcharger l'encadré.
function setupFormulaLegendToggles() {
  document.querySelectorAll(".model-formula-info-btn").forEach((btn) => {
    const legend = document.getElementById(btn.getAttribute("aria-controls"));
    btn.addEventListener("click", () => {
      const expanded = legend.hidden; // état après bascule
      legend.hidden = !expanded;
      btn.setAttribute("aria-expanded", String(expanded));
      btn.classList.toggle("active", expanded);
    });
  });
}

// Améliore les équations principales avec un vrai rendu LaTeX (KaTeX, chargé
// en CDN dans models.html) quand la lib a pu charger. Le HTML de secours
// (sub/sup) déjà présent dans la page reste affiché tel quel sinon — ex.
// le navigateur intégré à VS Code bloque ce script CDN (comme window.prompt()).
function setupFormulaRendering() {
  if (typeof katex === "undefined") return;
  document.querySelectorAll(".model-formula-eq[data-latex]").forEach((el) => {
    try {
      katex.render(el.dataset.latex, el, { displayMode: true, throwOnError: false });
    } catch {
      // le HTML de secours reste affiché
    }
  });
}

ensureDefaultPresets();
setupThemeSwitch();
setupThemeOnlyMenu();
setupModelParamsForm();
setupPresetManagers();
setupCollapsibleTabs();
setupFormulaLegendToggles();
setupFormulaRendering();
