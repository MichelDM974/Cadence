// Menu "Mon compte" (⚙) — ouverture/fermeture, partagé par index.html
// (dropdown), models.html et theorie.html (popup). Dépend de el(), fourni
// par model-params.js sur index/models et par theorie.js sur theorie.
// L'ordre de chargement varie (menu.js passe avant theorie.js) : sans
// importance, el() n'étant résolu qu'au moment de l'appel, toujours après
// que la page a fini de charger ses scripts.

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
