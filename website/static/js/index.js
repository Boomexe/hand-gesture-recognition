import { $, config, formatWord, toast, phase, resetPrediction, renderPrediction } from "./ui.js";

const video = $("webcam");
const canvas = $("output_canvas");
const context = canvas.getContext("2d");
const stage = $("camera-stage");
const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
let socket = null;
let detector = null;
let detectorPromise = null;
let stream = null;
let active = false;
let starting = false;
let animation = null;
let lastVideoTime = -1;
let lastDetectorTimestamp = -1;
let detectorTimestampOffset = 0;
let showLandmarks = true;
let mirrored = true;
let acceptingResults = false;
let awaitingReset = false;
let hasGesture = false;
let sessionWords = [];
let rateStarted = 0;
let rateFrames = 0;
let attempt = 0;

function serverState(connected) {
  const ready = connected && config.modelReady;
  $("engine-status").classList.toggle("is-ready", ready);
  $("engine-status").classList.toggle("is-offline", !ready);
  $("engine-label").textContent = !connected ? "Server disconnected" : config.modelReady ? "Model ready" : "Model unavailable";
}
function cameraError(message) {
  $("camera-error").textContent = message;
  $("camera-error").hidden = !message;
}
function updateButtons() {
  $("webcamButton").disabled = starting;
  $("camera-empty-button").disabled = starting;
  $("camera-button-label").textContent = starting ? "Preparing camera…" : active ? "Turn off camera" : "Turn on camera";
  $("camera-empty-button").textContent = starting ? "Preparing camera…" : "Turn on camera";
  $("reset-capture").disabled = !active;
  $("camera-state").classList.toggle("is-on", active);
  $("camera-state").lastElementChild.textContent = starting ? "Preparing" : active ? "Camera on" : "Camera off";
}
async function getDetector() {
  if (detector) return detector;
  if (!detectorPromise) detectorPromise = (async () => {
    const { HandLandmarker, FilesetResolver } = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    const options = {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numHands: 2,
    };
    try { return await HandLandmarker.createFromOptions(vision, options); }
    catch { options.baseOptions.delegate = "CPU"; return await HandLandmarker.createFromOptions(vision, options); }
  })().catch((error) => { detectorPromise = null; throw error; });
  detector = await detectorPromise;
  return detector;
}
function drawHands(landmarks) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!showLandmarks) return;
  const scale = Math.max(1, canvas.width / 640);
  context.lineWidth = 2 * scale;
  context.strokeStyle = "#dceab4";
  for (const hand of landmarks) {
    for (const [a, b] of connections) {
      context.beginPath();
      context.moveTo(hand[a].x * canvas.width, hand[a].y * canvas.height);
      context.lineTo(hand[b].x * canvas.width, hand[b].y * canvas.height);
      context.stroke();
    }
    for (const point of hand) {
      context.beginPath();
      context.arc(point.x * canvas.width, point.y * canvas.height, 3 * scale, 0, Math.PI * 2);
      context.fillStyle = "#f6f5de";
      context.fill();
    }
  }
}
function tick() {
  if (!active) return;
  try {
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const sourceTimestampMs = video.currentTime * 1000;
      // Preserve source-frame timing, including after the camera is restarted.
      if (sourceTimestampMs + detectorTimestampOffset <= lastDetectorTimestamp) {
        detectorTimestampOffset = lastDetectorTimestamp + 1 - sourceTimestampMs;
      }
      lastDetectorTimestamp = sourceTimestampMs + detectorTimestampOffset;
      const result = detector.detectForVideo(video, lastDetectorTimestamp);
      drawHands(result.landmarks);
      const hands = result.landmarks.length;
      $("guidance-label").textContent = hands ? hands + (hands === 1 ? " hand in view" : " hands in view") : "Bring your hands into view";
      // Only the preview is mirrored. Keep the landmark coordinates and handedness
      // in their original order for the existing server preprocessing.
      if (socket?.connected && !awaitingReset) socket.emit("landmark_data", {
        landmarks: result.landmarks, handednesses: result.handednesses,
        timestampMs: sourceTimestampMs,
      });
      rateFrames++;
      const now = performance.now();
      if (now - rateStarted >= 1000) {
        $("frame-rate").textContent = Math.round(rateFrames * 1000 / (now - rateStarted)) + " tracking fps";
        rateStarted = now;
        rateFrames = 0;
      }
    }
    animation = requestAnimationFrame(tick);
  } catch (error) {
    console.error("Hand tracking stopped:", error);
    stopCamera();
    cameraError("Hand tracking stopped. Turn the camera on again to retry.");
  }
}
function resetCapture() {
  hasGesture = false;
  resetPrediction("Make a sign, then lower your hands to finish.");
  if (socket?.connected) {
    awaitingReset = true;
    socket.emit("reset_live");
  }
}
function stopCamera() {
  attempt++;
  active = false;
  starting = false;
  acceptingResults = false;
  awaitingReset = false;
  cancelAnimationFrame(animation);
  if (stream) stream.getTracks().forEach((track) => { track.onended = null; track.stop(); });
  stream = null;
  video.srcObject = null;
  context.clearRect(0, 0, canvas.width, canvas.height);
  stage.classList.remove("is-streaming");
  $("camera-empty").hidden = false;
  $("camera-guidance").hidden = true;
  $("frame-rate").hidden = true;
  $("camera-device").textContent = "Default camera";
  if (socket?.connected) socket.emit("reset_live");
  phase($("prediction").closest(".recognition-card").classList.contains("has-result") ? "Paused" : "Waiting");
  updateButtons();
}
async function startCamera() {
  if (starting) return;
  if (active) { stopCamera(); return; }
  cameraError("");
  if (!config.modelReady) { cameraError("The recognition model isn't loaded. Check the server's model configuration."); return; }
  if (!socket?.connected) { cameraError("Can't reach the recognition server. Check that the app is running, then retry."); return; }
  if (!navigator.mediaDevices?.getUserMedia) { cameraError("Camera access requires localhost or HTTPS and a supported browser."); return; }
  starting = true;
  const thisAttempt = ++attempt;
  updateButtons();
  let newStream;
  try {
    await getDetector();
    if (thisAttempt !== attempt) return;
    newStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    if (thisAttempt !== attempt) { newStream.getTracks().forEach((track) => track.stop()); return; }
    stream = newStream;
    video.srcObject = stream;
    await video.play();
    if (thisAttempt !== attempt) return;
    stream.getVideoTracks()[0].onended = () => { stopCamera(); cameraError("The camera was disconnected. Reconnect it and try again."); };
    $("camera-device").textContent = stream.getVideoTracks()[0].label || "Camera";
    active = true;
    acceptingResults = true;
    lastVideoTime = -1;
    rateStarted = performance.now();
    rateFrames = 0;
    resetCapture();
    stage.classList.add("is-streaming");
    $("camera-empty").hidden = true;
    $("camera-guidance").hidden = false;
    $("frame-rate").hidden = false;
    tick();
  } catch (error) {
    console.error("Camera setup failed:", error);
    newStream?.getTracks().forEach((track) => track.stop());
    stopCamera();
    const messages = {
      NotAllowedError: "Camera access was denied. Allow it in your browser's site settings, then try again.",
      NotFoundError: "No camera was found. Connect a webcam and try again.",
      NotReadableError: "The camera is in use or unavailable. Close other camera apps and retry.",
    };
    cameraError(messages[error.name] || "Couldn't prepare hand tracking. Check your connection and reload the page.");
  } finally {
    starting = false;
    updateButtons();
  }
}
function renderSession() {
  $("session-list").replaceChildren();
  sessionWords.forEach(({ word, confidence }) => {
    const item = document.createElement("li");
    item.textContent = formatWord(word);
    const score = document.createElement("span");
    score.textContent = Math.round(confidence * 100) + "%";
    item.append(score);
    $("session-list").append(item);
  });
  $("session-count").textContent = sessionWords.length;
  $("session-empty").hidden = sessionWords.length > 0;
  $("copy-session").disabled = $("clear-session").disabled = !sessionWords.length;
}
if (typeof window.io === "function") {
  socket = window.io(); // Same origin: works beyond localhost:8000.
  serverState(socket.connected);
  socket.on("connect", () => {
    serverState(true);
    if (active) {
      acceptingResults = true;
      cameraError("");
      resetCapture();
    }
  });
  socket.on("disconnect", () => {
    serverState(false);
    if (active) { acceptingResults = false; cameraError("Server disconnected. Turn the camera off and retry when it reconnects."); }
  });
  socket.on("connect_error", () => serverState(false));
  socket.on("prediction", (data) => {
    if (data.status === "reset") { awaitingReset = false; return; }
    if (!active || !acceptingResults || awaitingReset) return;
    if (data.status === "warming_up") {
      if (!hasGesture) { resetPrediction("Keep signing. Gathering the first few frames."); hasGesture = true; }
      phase("Gathering", "active");
      $("capture-detail").textContent = data.frames + " / " + data.needed + " starting frames";
    } else if (data.status === "prediction" || data.status === "low_confidence") {
      hasGesture = !data.final;
      renderPrediction(data, { final: data.final });
      if (data.final && data.confidence >= config.confidenceThreshold) {
        sessionWords.push({ word: data.predicted_word, confidence: data.confidence });
        renderSession();
      }
    } else if (data.status === "no_hands") {
      const wasGathering = hasGesture;
      hasGesture = false;
      // Keep a completed result visible while waiting for the next gesture.
      if (wasGathering && !document.querySelector(".recognition-card").classList.contains("has-result")) {
        resetPrediction("That gesture was too short. Try again.");
      }
    } else if (data.status === "error") {
      resetPrediction("The gesture couldn't be processed. Reset and try again.");
      phase("Try again", "caution");
      console.error("Recognition error:", data.error);
    }
  });
} else {
  serverState(false);
  cameraError("The connection library couldn't load. Check your internet connection and reload.");
}
$("webcamButton").addEventListener("click", startCamera);
$("camera-empty-button").addEventListener("click", startCamera);
$("reset-capture").addEventListener("click", resetCapture);
$("landmark-toggle").addEventListener("click", () => {
  showLandmarks = !showLandmarks;
  $("landmark-toggle").setAttribute("aria-pressed", String(showLandmarks));
  $("landmark-toggle").classList.toggle("selected", showLandmarks);
  if (!showLandmarks) context.clearRect(0, 0, canvas.width, canvas.height);
});
$("mirror-toggle").addEventListener("click", () => {
  mirrored = !mirrored;
  [video, canvas].forEach((element) => element.classList.toggle("mirrored", mirrored));
  $("mirror-toggle").setAttribute("aria-pressed", String(mirrored));
  $("mirror-toggle").classList.toggle("selected", mirrored);
});
$("fullscreen-button").addEventListener("click", async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await stage.requestFullscreen(); }
  catch { toast("Full screen isn't available in this browser."); }
});
$("copy-session").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(sessionWords.map((item) => formatWord(item.word)).join(" ")); toast("Session copied."); }
  catch { toast("Clipboard access isn't available in this browser."); }
});
$("clear-session").addEventListener("click", () => { sessionWords = []; renderSession(); toast("Session cleared."); });
function selectPractice(word) {
  if (!config.words.includes(word)) return;
  document.querySelectorAll("[data-practice]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.practice === word)));
  $("practice-target").textContent = "Practicing: " + formatWord(word);
  $("practice-target").hidden = false;
}
document.querySelectorAll("[data-practice]").forEach((button) => button.addEventListener("click", () => selectPractice(button.dataset.practice)));
selectPractice(new URLSearchParams(location.search).get("practice"));
window.addEventListener("pagehide", stopCamera);
updateButtons();
