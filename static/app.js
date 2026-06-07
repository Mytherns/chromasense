const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const previewFrame = document.querySelector("#preview-frame");
const previewImage = document.querySelector("#preview-image");
const nColors = document.querySelector("#n-colors");
const nColorsOutput = document.querySelector("#n-colors-output");
const analyzeButton = document.querySelector("#analyze-button");
const statusPill = document.querySelector("#status-pill");
const emptyState = document.querySelector("#empty-state");
const results = document.querySelector("#results");

let previewUrl = null;

nColors.addEventListener("input", () => {
  nColorsOutput.textContent = nColors.value;
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  setFilePreview(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;

  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setFilePreview(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) return;

  setBusy(true);
  setStatus("Analyzing", "busy");

  try {
    const payload = new FormData();
    payload.append("file", file);

    const response = await fetch(`/analyze?n_colors=${encodeURIComponent(nColors.value)}`, {
      method: "POST",
      body: payload,
    });

    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const message = data?.error?.message || `Request failed with ${response.status}`;
      throw new Error(message);
    }

    renderResult(data);
    setStatus("Done", "ok");
  } catch (error) {
    renderError(error);
    setStatus("Failed", "error");
  } finally {
    setBusy(false);
  }
});

function setFilePreview(file) {
  analyzeButton.disabled = !file;
  fileName.textContent = file ? file.name : "JPEG, PNG, or WebP";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  if (!file) {
    previewFrame.hidden = true;
    previewImage.removeAttribute("src");
    return;
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewFrame.hidden = false;
}

function setBusy(isBusy) {
  analyzeButton.disabled = isBusy || !fileInput.files?.[0];
  analyzeButton.textContent = isBusy ? "Analyzing..." : "Analyze";
  form.classList.toggle("is-busy", isBusy);
}

function setStatus(text, state) {
  statusPill.textContent = text;
  statusPill.dataset.state = state;
}

function renderResult(data) {
  emptyState.hidden = true;
  results.hidden = false;

  const clip = data.clip_classification || {};
  document.querySelector("#mood").textContent = data.mood || clip.primary_mood || "Unknown";
  document.querySelector("#confidence").textContent = formatScore(clip.confidence);
  document.querySelector("#vibe").textContent = data.vibe || "";
  document.querySelector("#use-case").textContent = data.use_case || "";
  document.querySelector("#clip-primary").textContent = clip.primary_mood || "unknown";
  document.querySelector("#metadata").textContent = JSON.stringify(data.metadata || {}, null, 2);

  renderTags(data.tags || []);
  renderPalette(data.colors || []);
  renderClipScores(clip.top3_moods || []);
  renderDescriptionList("#feedback", data.feedback || {});
  renderPairing(data.pairing || {});
  renderWarnings(data.warnings || []);
}

function renderError(error) {
  emptyState.hidden = true;
  results.hidden = false;

  document.querySelector("#mood").textContent = "Analysis failed";
  document.querySelector("#confidence").textContent = "0%";
  document.querySelector("#vibe").textContent = error.message;
  document.querySelector("#use-case").textContent = "";
  document.querySelector("#clip-primary").textContent = "-";
  document.querySelector("#metadata").textContent = "{}";

  renderTags([]);
  renderPalette([]);
  renderClipScores([]);
  renderDescriptionList("#feedback", {});
  renderPairing({});
  renderWarnings([error.message]);
}

function renderTags(tags) {
  const container = document.querySelector("#tags");
  container.replaceChildren(...tags.map((tag) => {
    const item = document.createElement("span");
    item.className = "tag";
    item.textContent = tag;
    return item;
  }));
}

function renderPalette(colors) {
  const grid = document.querySelector("#palette-grid");
  document.querySelector("#palette-count").textContent = `${colors.length} colors`;

  grid.replaceChildren(...colors.map((color) => {
    const item = document.createElement("article");
    item.className = "color-card";

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = color.hex || "#888888";

    const body = document.createElement("div");
    const name = document.createElement("h4");
    name.textContent = color.name || color.hex || "Color";
    const meta = document.createElement("p");
    meta.textContent = `${color.role || "Palette"} · ${color.hex || "-"} · ${formatPercent(color.percentage)}`;

    body.append(name, meta);
    item.append(swatch, body);
    return item;
  }));
}

function renderClipScores(scores) {
  const list = document.querySelector("#clip-scores");
  list.replaceChildren(...scores.map((score) => {
    const row = document.createElement("div");
    row.className = "score-row";

    const label = document.createElement("span");
    label.textContent = score.mood || "unknown";

    const value = document.createElement("strong");
    value.textContent = formatScore(score.score);

    row.append(label, value);
    return row;
  }));
}

function renderDescriptionList(selector, entries) {
  const list = document.querySelector(selector);
  const nodes = Object.entries(entries).flatMap(([key, value]) => {
    const term = document.createElement("dt");
    term.textContent = formatLabel(key);
    const description = document.createElement("dd");
    description.textContent = value || "";
    return [term, description];
  });
  list.replaceChildren(...nodes);
}

function renderPairing(pairing) {
  const grid = document.querySelector("#pairing");
  const nodes = Object.entries(pairing).map(([key, value]) => {
    const item = document.createElement("article");
    const title = document.createElement("h4");
    title.textContent = formatLabel(key);
    const body = document.createElement("p");
    body.textContent = value || "";
    item.append(title, body);
    return item;
  });
  grid.replaceChildren(...nodes);
}

function renderWarnings(warnings) {
  const section = document.querySelector("#warnings-section");
  const list = document.querySelector("#warnings");
  section.hidden = warnings.length === 0;
  list.replaceChildren(...warnings.map((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    return item;
  }));
}

function formatLabel(value) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "-";
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "0%";
}
