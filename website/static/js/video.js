import { $, config, toast, resetPrediction, renderPrediction } from "./ui.js";

let file = null;
let objectUrl = null;
let busy = false;
let results = [];
function showError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = !message;
}
function clearResults() {
  results = [];
  $("timeline-card").hidden = true;
  resetPrediction("Analyze your clip to see the result.");
}
function chooseFile(nextFile) {
  if (!nextFile || busy) return;
  if (!/\.(mp4|webm|mov)$/i.test(nextFile.name)) { showError("Choose an MP4, WebM or MOV video."); return; }
  if (!nextFile.size) { showError("This file is empty. Choose another video."); return; }
  if (nextFile.size > 100 * 1024 * 1024) { showError("The clip is larger than 100 MB. Choose a shorter video."); return; }
  file = nextFile;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $("uploaded-preview").src = objectUrl;
  $("uploaded-preview").hidden = false;
  $("upload-empty").hidden = true;
  $("file-detail").hidden = false;
  $("file-name").textContent = file.name;
  $("file-size").textContent = (file.size / 1024 / 1024).toFixed(1) + " MB";
  $("analyze-video").disabled = !config.modelReady;
  $("upload-status").textContent = config.modelReady ? "Ready to analyze" : "The model is unavailable";
  showError("");
  clearResults();
}
function setBusy(value) {
  busy = value;
  document.querySelector(".upload-card").classList.toggle("is-processing", value);
  $("analyze-video").disabled = value || !file || !config.modelReady;
  $("change-video").disabled = $("choose-video").disabled = $("prefix-mode").disabled = value;
  $("analyze-label").textContent = value ? "Analyzing" : "Analyze video";
}
function displayFrame(index) {
  const result = results[index];
  const last = index === results.length - 1;
  renderPrediction(result, { final: last, detail: "Detected frame " + (index + 1) + " of " + results.length });
  $("prediction-hint").textContent = last ? "Full gesture analyzed." : "Prediction from the first " + (index + 1) + " detected frames.";
  $("timeline-counter").textContent = (index + 1) + " / " + results.length + " frames";
}
function drawTimeline() {
  const svg = $("timeline-chart");
  svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  function element(tag, attributes) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    svg.append(el);
    return el;
  }
  const x = (index) => results.length === 1 ? 400 : index * 800 / (results.length - 1);
  const y = (value) => 85 - Math.max(0, Math.min(1, Number(value) || 0)) * 75;
  element("line", { x1: 0, y1: y(config.confidenceThreshold), x2: 800, y2: y(config.confidenceThreshold), stroke: "#d6dcc9", "stroke-dasharray": "5 5", "stroke-width": 1 });
  const points = results.map((result, index) => x(index) + "," + y(result.confidence)).join(" ");
  element("polygon", { points: "0,90 " + points + " 800,90", fill: "#eef2e2" });
  element("polyline", { points, fill: "none", stroke: "#7c965c", "stroke-width": 2, "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke" });
  const last = results.length - 1;
  element("circle", { cx: x(last), cy: y(results[last].confidence), r: 3, fill: "#315943" });
  const title = document.createElementNS(ns, "title");
  title.textContent = "Top predicted class confidence over " + results.length + " detected frames. The predicted class may change.";
  svg.prepend(title);
}
$("choose-video").addEventListener("click", () => $("video-input").click());
$("change-video").addEventListener("click", () => $("video-input").click());
$("video-input").addEventListener("change", (event) => { chooseFile(event.target.files[0]); event.target.value = ""; });
["dragenter", "dragover"].forEach((name) => $("drop-zone").addEventListener(name, (event) => {
  event.preventDefault();
  if (!busy) $("drop-zone").classList.add("is-dragover");
}));
["dragleave", "drop"].forEach((name) => $("drop-zone").addEventListener(name, (event) => {
  event.preventDefault();
  $("drop-zone").classList.remove("is-dragover");
}));
$("drop-zone").addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
$("uploaded-preview").addEventListener("error", () => {
  if (file) showError("This browser can't preview the clip. You can still try analyzing it, or use an MP4.");
});
$("prefix-mode").addEventListener("change", () => { clearResults(); $("upload-status").textContent = file ? "Ready to analyze" : "Choose a clip to begin"; });
$("frame-slider").addEventListener("input", (event) => displayFrame(Number(event.target.value)));
$("analyze-video").addEventListener("click", async () => {
  if (!file || busy) return;
  const continuous = $("prefix-mode").checked;
  const body = new FormData();
  body.append("video", file);
  showError("");
  clearResults();
  setBusy(true);
  $("upload-status").textContent = continuous ? "Reading the clip and each gesture prefix…" : "Reading your gesture…";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
  try {
    const response = await fetch(continuous ? "/upload-video-continuous" : "/upload-video", { method: "POST", body, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.error?.includes("No hands") ? "No hands were detected. Try a clearer clip." : data?.message || "The video couldn't be analyzed. Please try again.");
    if (continuous) {
      if (!Array.isArray(data) || !data.length) throw new Error("No gesture frames were found in the clip.");
      results = data;
      $("frame-slider").max = results.length - 1;
      $("frame-slider").value = results.length - 1;
      drawTimeline();
      displayFrame(results.length - 1);
      $("timeline-card").hidden = false;
    } else {
      if (!data?.predicted_word) throw new Error("The server returned an incomplete prediction.");
      renderPrediction(data, { final: true, detail: "Full video gesture" });
      $("prediction-hint").textContent = data.confidence >= config.confidenceThreshold ? "Full gesture analyzed." : "Not quite certain. Try a clearer clip.";
    }
    $("upload-status").textContent = "Analysis complete";
    toast("Video analyzed.");
  } catch (error) {
    showError(error.name === "AbortError" ? "Analysis took too long. Try a shorter clip." : error.message);
    $("upload-status").textContent = "Couldn't finish analysis";
    resetPrediction("Try another clip or analyze again.");
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
});
window.addEventListener("pagehide", () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
