// login.html / setup.html : redirige déjà-connecté -> index, pas-de-compte ->
// setup, compte-déjà-créé -> login (setup n'est utilisable qu'une fois), et
// branche les deux formulaires. Indépendant de model-params.js/menu.js (ces
// pages n'ont pas de menu applicatif) — définit son propre el().

const el = (id) => document.getElementById(id);

async function checkAuthPageAccess() {
  const resp = await fetch("/api/me");
  const body = await resp.json();
  const onLoginPage = !!el("login-form");
  const onSetupPage = !!el("setup-form");

  if (body.logged_in) {
    window.location.replace("index.html");
  } else if (body.needs_setup && onLoginPage) {
    window.location.replace("setup.html");
  } else if (!body.needs_setup && onSetupPage) {
    window.location.replace("login.html");
  }
}

function wireAuthForm(form, url, statusEl) {
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    statusEl.textContent = "";
    statusEl.className = "status";
    const resp = await fetch(url, { method: "POST", body: new FormData(form) });
    if (resp.ok) {
      window.location.replace("index.html");
      return;
    }
    const body = await resp.json().catch(() => ({}));
    statusEl.textContent = body.detail || "Une erreur est survenue.";
    statusEl.className = "status status--error";
  });
}

checkAuthPageAccess();
wireAuthForm(el("login-form"), "/api/login", el("login-status"));
wireAuthForm(el("setup-form"), "/api/setup", el("setup-status"));
