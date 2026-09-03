// Menu "Mon compte" (⚙) — ouverture/fermeture, partagé entre index.html
// (dropdown avec "Modèles") et models.html (popup vide pour l'instant, prêt
// à accueillir un futur réglage). Dépend de el(), défini dans
// model-params.js, chargé avant ce fichier.

function setupPopupToggle(triggerEl, scopeSelector, isOpen, open, close) {
  triggerEl.addEventListener("click", (e) => {
    e.stopPropagation();
    if (isOpen()) close();
    else open();
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(scopeSelector)) close();
  });
}

// index.html : le bouton ⚙ ouvre une liste déroulante ("Modèles").
function setupAccountMenu() {
  const dropdown = el("account-dropdown");
  setupPopupToggle(
    el("account-menu-btn"),
    ".account-menu",
    () => !dropdown.hidden,
    () => {
      dropdown.hidden = false;
    },
    () => {
      dropdown.hidden = true;
    }
  );
}

// models.html : pas de sous-menu, le bouton ⚙ ouvre directement le popup.
function setupSettingsMenu() {
  const popup = el("settings-popup");
  setupPopupToggle(
    el("account-menu-btn"),
    ".account-menu",
    () => !popup.hidden,
    () => {
      popup.hidden = false;
    },
    () => {
      popup.hidden = true;
    }
  );
}
