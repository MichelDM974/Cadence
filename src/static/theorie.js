// Page théorie (theorie.html) : pas de formulaire, juste des explications.
// setupSettingsMenu vient de menu.js, chargé avant ce fichier.

const el = (id) => document.getElementById(id);

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Améliore les équations principales avec un vrai rendu LaTeX (KaTeX, chargé
// en CDN dans theorie.html) quand la lib a pu charger. Le HTML de secours
// (sub/sup) déjà présent dans la page reste affiché tel quel sinon — ex.
// le navigateur intégré à VS Code bloque ce script CDN (comme window.prompt()).
function setupFormulaRendering() {
  if (typeof katex === "undefined") return;
  document.querySelectorAll(".model-formula-eq[data-latex]").forEach((noeud) => {
    try {
      katex.render(noeud.dataset.latex, noeud, { displayMode: true, throwOnError: false });
    } catch {
      // le HTML de secours reste affiché
    }
  });
}

function chartAxisOptions(showLegend = false) {
  const text = cssVar("--muted");
  const grid = cssVar("--chart-grid");
  return {
    responsive: true,
    animation: false,
    plugins: { legend: { display: showLegend, labels: { color: text } } },
    scales: {
      x: { ticks: { color: text, maxTicksLimit: 8 }, grid: { color: grid } },
      y: { ticks: { color: text }, grid: { color: grid } },
    },
  };
}

// Données 100% synthétiques (motif sinusoïdal reproductible, pas de vrai
// aléatoire) : ces graphiques illustrent un concept, ce ne sont pas des
// prévisions réelles de Cadence.

function renderArimaCharts() {
  if (typeof Chart === "undefined" || !el("theorie-arima-raw")) return;
  const n = 40;
  const raw = [];
  let level = 60;
  for (let i = 0; i < n; i++) {
    level += 0.8; // tendance qui monte régulièrement
    const wiggle = (Math.sin(i * 1.3) + Math.sin(i * 0.6)) * 3;
    raw.push(Math.round(level + wiggle));
  }
  const diff = raw.map((v, i) => (i === 0 ? null : v - raw[i - 1]));
  const labels = raw.map((_, i) => `J${i + 1}`);

  new Chart(el("theorie-arima-raw"), {
    type: "line",
    data: {
      labels,
      datasets: [{ data: raw, borderColor: cssVar("--accent"), backgroundColor: "transparent", tension: 0.3, pointRadius: 0 }],
    },
    options: chartAxisOptions(),
  });

  new Chart(el("theorie-arima-diff"), {
    type: "line",
    data: {
      labels,
      datasets: [{ data: diff, borderColor: cssVar("--series-1"), backgroundColor: "transparent", tension: 0.3, pointRadius: 0 }],
    },
    options: chartAxisOptions(),
  });
}

function renderSarimaChart() {
  if (typeof Chart === "undefined" || !el("theorie-sarima-chart")) return;
  const weeklyPattern = [70, 75, 78, 82, 95, 60, 50]; // lun..dim, illustratif
  const weeks = 5;
  const data = [];
  for (let i = 0; i < weeks * 7; i++) {
    data.push(Math.round(weeklyPattern[i % 7] + i * 0.3));
  }
  const labels = data.map((_, i) => `J${i + 1}`);

  new Chart(el("theorie-sarima-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [{ data, borderColor: cssVar("--series-7"), backgroundColor: "transparent", tension: 0.3, pointRadius: 2 }],
    },
    options: chartAxisOptions(),
  });
}

function renderSarimaxChart() {
  if (typeof Chart === "undefined" || !el("theorie-sarimax-chart")) return;
  const days = 21;
  const promoDays = new Set([5, 12, 18]);
  const y = [];
  const promoBars = [];
  for (let i = 0; i < days; i++) {
    const base = 65 + Math.sin(i / 2) * 4;
    const isPromo = promoDays.has(i);
    promoBars.push(isPromo ? 25 : 0);
    y.push(Math.round(base + (isPromo ? 22 : 0)));
  }
  const labels = y.map((_, i) => `J${i + 1}`);

  new Chart(el("theorie-sarimax-chart"), {
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "Jour de promotion (X_t)",
          data: promoBars,
          backgroundColor: cssVar("--accent-soft"),
          borderRadius: 4,
          yAxisID: "y1",
          order: 2,
        },
        {
          type: "line",
          label: "Ventes (y_t)",
          data: y,
          borderColor: cssVar("--accent"),
          backgroundColor: "transparent",
          tension: 0.3,
          pointRadius: 2,
          order: 1,
        },
      ],
    },
    options: {
      ...chartAxisOptions(true),
      scales: {
        ...chartAxisOptions().scales,
        y1: { display: false, min: 0, max: 100 },
      },
    },
  });
}

setupSettingsMenu();
setupFormulaRendering();
renderArimaCharts();
renderSarimaChart();
renderSarimaxChart();
