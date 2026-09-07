// Chargé en premier sur toute page protégée (index.html, models.html,
// theorie.html) : redirige vers login.html (ou setup.html si aucun compte
// n'existe encore) si la session n'est pas valide. Sinon affiche
// l'identifiant et branche la déconnexion sur les éléments #account-username
// / #logout-btn (présents dans le dropdown ⚙ de chaque page).
(async function () {
  const resp = await fetch("/api/me");
  const body = await resp.json();

  if (!body.logged_in) {
    window.location.replace(body.needs_setup ? "setup.html" : "login.html");
    return;
  }

  // Pas d'attente de DOMContentLoaded : le script est chargé en fin de <body>,
  // donc le dropdown ⚙ est déjà dans le DOM, et surtout l'événement s'est déjà
  // déclenché pendant l'await ci-dessus — un listener posé ici ne partirait
  // jamais (le nom d'utilisateur restait vide et la déconnexion inerte).
  const usernameEl = document.getElementById("account-username");
  if (usernameEl) usernameEl.textContent = body.username;

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await fetch("/api/logout", { method: "POST" });
      window.location.replace("login.html");
    });
  }
})();
