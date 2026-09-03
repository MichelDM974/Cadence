// Bascule de thème (clair/sombre/système) et menu "Mon compte" (⚙) —
// partagés entre index.html (dropdown Thème/Modèles) et models.html (le
// bouton ⚙ ouvre directement le sélecteur de thème, pas de sous-menu).
// Dépend de el(), défini dans model-params.js, chargé avant ce fichier.

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
  // theme.js est partagé avec models.html, qui n'a pas de graphiques : plutôt que
  // d'appeler directement une fonction globale qui peut ne pas exister sur cette
  // page, on notifie qui veut bien écouter (app.js s'y abonne dans son init()).
  document.dispatchEvent(new CustomEvent("cadence:theme-changed"));
}

function setupThemeSwitch() {
  const saved = localStorage.getItem("cadence-theme") || "system";
  updateThemeButtons(saved);

  document.querySelectorAll("#theme-switch button").forEach((btn) => {
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeValue));
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!localStorage.getItem("cadence-theme")) document.dispatchEvent(new CustomEvent("cadence:theme-changed"));
  });
}

// Ouvre/ferme un panneau au clic sur son déclencheur, le referme au clic en
// dehors de `scopeSelector` — partagé par le menu "Mon compte" (index.html,
// avec un sous-menu imbriqué) et le popup de thème seul (models.html).
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

// index.html : le bouton ⚙ ouvre une liste déroulante (Thème / Modèles) ;
// cliquer sur "Thème" remplace son libellé, dans la même ligne, par les 3
// ronds clair/sombre/système (pas de popup séparé).
function setupAccountMenu() {
  const dropdown = el("account-dropdown");
  const appearanceBtn = el("account-appearance-btn");
  const themeSwitch = el("theme-switch");

  function closeAll() {
    dropdown.hidden = true;
    themeSwitch.hidden = true;
    appearanceBtn.hidden = false;
  }

  setupPopupToggle(
    el("account-menu-btn"),
    ".account-menu",
    () => !dropdown.hidden,
    () => {
      dropdown.hidden = false;
    },
    closeAll
  );

  appearanceBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    appearanceBtn.hidden = true;
    themeSwitch.hidden = false;
  });
}

// models.html : pas de sous-menu, le bouton ⚙ ouvre directement le popup de thème.
function setupThemeOnlyMenu() {
  const popup = el("appearance-popup");
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
