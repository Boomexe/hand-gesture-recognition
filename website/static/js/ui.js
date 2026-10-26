export const config = JSON.parse(document.getElementById("app-config").textContent);
export const $ = (id) => document.getElementById(id);
const names = { THANKYOU: "Thank you", DONTKNOW: "Don't know", "DONT-KNOW": "Don't know", ORANGEEYES: "Orange eyes", NEXTWEEK: "Next week", NICETOMEETYOU: "Nice to meet you" };
export function formatWord(word = "") {
  const label = String(word).replace(/\d+$/, "");
  return names[label] || label.toLowerCase().replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).replace(/\bI\b/g, "I");
}
let toastTimer;
export function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}
export function phase(label, kind = "") {
  $("prediction-phase").textContent = label;
  $("prediction-phase").className = "small-badge" + (kind ? " is-" + kind : "");
}
export function resetPrediction(hint = "") {
  document.querySelector(".recognition-card").classList.remove("has-result");
  $("prediction").textContent = "Waiting for a gesture.";
  $("prediction-caption").textContent = "Prediction";
  $("prediction-hint").textContent = hint;
  $("confidence-value").textContent = "—";
  $("confidence-fill").style.width = "0%";
  $("confidence-track").setAttribute("aria-valuenow", "0");
  $("confidence-track").classList.remove("is-caution");
  $("confidence-note").textContent = "Waiting for a gesture";
  $("alternatives").hidden = true;
  $("capture-detail").textContent = "Waiting for input";
  phase("Waiting");
}
export function renderPrediction(data, { final = false, detail = "" } = {}) {
  const rawConfidence = Number(data.confidence);
  const confidence = Number.isFinite(rawConfidence) ? Math.max(0, Math.min(1, rawConfidence)) : 0;
  const certain = confidence >= config.confidenceThreshold;
  const word = formatWord(data.predicted_word ?? data.pred_word);
  document.querySelector(".recognition-card").classList.add("has-result");
  if ($("prediction").textContent !== word) $("prediction").textContent = word;
  $("prediction-caption").textContent = certain ? "The word is" : "It might be";
  $("prediction-hint").textContent = final
    ? (certain ? "Gesture complete." : "Not quite certain. Try the gesture again.")
    : (certain ? "Keep going. Lower your hands to finish." : "Still taking shape. Keep signing.");
  $("confidence-value").textContent = (confidence * 100).toFixed(1) + "%";
  $("confidence-fill").style.width = confidence * 100 + "%";
  $("confidence-track").setAttribute("aria-valuenow", (confidence * 100).toFixed(1));
  $("confidence-track").classList.toggle("is-caution", !certain);
  $("confidence-note").textContent = certain ? "Above the recognition threshold" : "Below the recognition threshold";
  $("capture-detail").textContent = detail || (data.detected_frames ? data.detected_frames + " detected frames" : "Video analysis");
  phase(final ? (certain ? "Complete" : "Uncertain") : "Reading", certain ? "active" : "caution");
  const alternatives = (data.top_predictions || []).filter((p) => p.word !== data.predicted_word).slice(0, 2);
  $("alternative-list").replaceChildren();
  for (const item of alternatives) {
    const row = document.createElement("div");
    row.className = "alternative-row";
    const name = document.createElement("span");
    name.textContent = formatWord(item.word);
    const value = document.createElement("span");
    value.textContent = (Number(item.confidence) * 100).toFixed(1) + "%";
    row.append(name, value);
    $("alternative-list").append(row);
  }
  $("alternatives").hidden = !alternatives.length;
}
