// Aides communes aux web apps de « Mes Projets » (style commun, modèle CryptX) :
// police des graphiques, bulles d'aide, explications des panneaux derrière un « ? ».
// Tout dans une fonction : les autres scripts déclarent déjà el(), cssVar()… au
// niveau global, une redéclaration ici casserait la page.
(function () {
  // Même police que la page (Chart.js n'est pas chargé sur toutes les pages).
  if (window.Chart) Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

  // Bulles d'aide : remplacent l'infobulle native des attributs title (lente, sans style),
  // pour tout élément, même créé après coup. Le title est mis de côté pendant le survol,
  // sinon le navigateur afficherait aussi la sienne.
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.appendChild(tip);
  let cur = null;

  function hide() {
    if (!cur) return;
    if (!cur.hasAttribute("title")) cur.setAttribute("title", cur.dataset.tip);
    delete cur.dataset.tip;
    cur = null;
    tip.hidden = true;
  }
  function show(t) {
    const text = t.getAttribute("title");
    if (!text) return;
    cur = t;
    t.dataset.tip = text;
    t.removeAttribute("title");
    tip.textContent = text;
    tip.hidden = false;
    // Sous l'élément, centrée ; au-dessus s'il n'y a pas la place ; jamais hors de l'écran.
    const r = t.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8)}px`;
    tip.style.top = `${r.bottom + 8 + h > innerHeight - 8 ? Math.max(8, r.top - h - 8) : r.bottom + 8}px`;
  }
  function enter(e) {
    if (cur && cur.contains(e.target)) return;
    hide();
    const t = e.target.closest?.("[title]");
    if (t) show(t);
  }
  function leave(e) {
    if (cur && !cur.contains(e.relatedTarget)) hide();
  }
  document.addEventListener("mouseover", enter);
  document.addEventListener("mouseout", leave);
  document.addEventListener("focusin", enter);
  document.addEventListener("focusout", leave);
  addEventListener("scroll", hide, true);

  // Explication qui ouvre un panneau, juste sous son titre : derrière un « ? » à côté du
  // titre si elle dépasse deux lignes ; les courtes restent à l'écran.
  function helpButton(btn, p, name) {
    btn.title = "Comment lire ce panneau";
    btn.setAttribute("aria-label", `Comment lire : ${name}`);
    btn.setAttribute("aria-controls", p.id);
    btn.setAttribute("aria-expanded", "false");
    btn.addEventListener("click", () => {
      p.hidden = !p.hidden;
      btn.setAttribute("aria-expanded", String(!p.hidden));
      btn.classList.toggle("active", !p.hidden);
    });
  }
  function fold() {
    let n = 0;
    for (const p of document.querySelectorAll("main .panel > h2 + p.hint:not([id])")) {
      if (!p.offsetParent) continue;  // panneau encore masqué : mesuré quand il paraîtra
      const lineHeight = parseFloat(getComputedStyle(p).lineHeight) || 20;
      if (p.scrollHeight <= 2 * lineHeight + 2) continue;
      const h2 = p.previousElementSibling;
      p.id = `help-${++n}-${Math.random().toString(36).slice(2, 7)}`;
      p.hidden = true;
      p.classList.add("help-text");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "help-btn";
      btn.textContent = "?";
      helpButton(btn, p, h2.textContent.trim());
      h2.append(" ", btn);
    }
  }
  fold();
  // Panneaux de résultats affichés après coup (analyse, essai Kaggle).
  new MutationObserver(fold).observe(document.querySelector("main") || document.body,
    { attributes: true, attributeFilter: ["hidden"], subtree: true });
})();
