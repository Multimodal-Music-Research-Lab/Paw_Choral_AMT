const resultData = [
  { label: "PawCT w/o union", note: 0.190, frame: 0.379, color: "#8c948e" },
  { label: "PawCT", note: 0.217, frame: 0.458, color: "#d84d5a" },
  { label: "PawCT-RP", note: 0.209, frame: 0.510, color: "#efa63f" },
  { label: "PawCT-OC", note: 0.225, frame: 0.503, color: "#098896" },
  { label: "PagCT + Post-VA", note: 0.175, frame: 0.489, color: "#326b99" },
];

const resultBars = document.querySelector("#result-bars");
const metricButtons = [...document.querySelectorAll(".metric-toggle")];

function renderBars(metric) {
  const scaleMax = metric === "note" ? 0.25 : 0.55;
  resultBars.replaceChildren(...resultData.map((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "bar-row";
    wrapper.innerHTML = `
      <span class="bar-label">${row.label}</span>
      <span class="bar-track"><span class="bar-fill" style="--width:${Math.min(100, row[metric] / scaleMax * 100)}%;--color:${row.color}"></span></span>
      <span class="bar-value">${row[metric].toFixed(3)}</span>`;
    return wrapper;
  }));
}

metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    metricButtons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
    renderBars(button.dataset.metric);
  });
});

const panelCopy = {
  gt: {
    kind: "Reference",
    title: "Ground truth SATB",
    copy: "The annotated voice trajectories provide the comparison target.",
    metric: "Exsultate Deo · 10–100 s",
    alt: "Ground-truth SATB piano roll for Exsultate Deo",
  },
  paw: {
    kind: "Joint acoustic model",
    title: "PawCT-OC",
    copy: "Voice-specific heads preserve coherent SATB paths, with some short notes still missed.",
    metric: "Example note F1 · 0.3655",
    alt: "PawCT-OC SATB prediction piano roll for Exsultate Deo",
  },
  pag: {
    kind: "Part-agnostic model",
    title: "PagCT",
    copy: "The merged transcription retains more note content but does not identify the vocal part.",
    metric: "Example note F1 · 0.4208",
    alt: "PagCT merged prediction piano roll for Exsultate Deo",
  },
  post: {
    kind: "Two-stage baseline",
    title: "PagCT + Post-VA",
    copy: "Symbolic reassignment inherits missed and over-extended notes, then adds voice-label errors.",
    metric: "Example note F1 · 0.3059",
    alt: "PagCT plus Post-VA SATB prediction piano roll for Exsultate Deo",
  },
};

const viewer = document.querySelector(".viewer");
const viewButtons = [...document.querySelectorAll(".view-toggle")];
const viewerKind = document.querySelector("#viewer-kind");
const viewerTitle = document.querySelector("#viewer-title");
const viewerCopy = document.querySelector("#viewer-copy");
const viewerMetric = document.querySelector("#viewer-metric");
const crop = document.querySelector(".crop");

viewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const panel = button.dataset.panel;
    const content = panelCopy[panel];
    viewer.dataset.panel = panel;
    crop.setAttribute("aria-label", content.alt);
    viewerKind.textContent = content.kind;
    viewerTitle.textContent = content.title;
    viewerCopy.textContent = content.copy;
    viewerMetric.textContent = content.metric;
    viewButtons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
  });
});

renderBars("note");
