// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// import io from "socket.io-client";

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const socket = io("http://localhost:8000");

socket.on("connect", () => {
  console.log("Connected to server");
});

socket.on("response", (data) => {
  console.log("Server response:", data);
});

socket.on("disconnect", () => {
  console.log("Disconnected from server");
});

socket.on("prediction", (data) => {
  console.log("Prediction:", data);

  const output = document.getElementById("prediction");

  if (!output) return;

  if (data.status === "prediction" || data.status === "low_confidence") {
    output.innerText = `${data.predicted_word} (${(data.confidence * 100).toFixed(1)}%)`;
  } else if (data.status === "warming_up") {
    output.innerText = `Warming up: ${data.frames}/${data.needed}`;
  } else if (data.status === "no_hands") {
    output.innerText = "No hands detected";
  } else if (data.status === "error") {
    output.innerText = `Error: ${data.error}`;
  } else if (data.status === "no_hands") {
    output.innerText = "No hands detected";
  }
});

// NOTE: These are typically provided by Mediapipe drawing utils in your HTML.
// Make sure you have something like:
// import { drawConnectors, drawLandmarks } from "...";
// and HAND_CONNECTIONS / HAND_CONNECTIONS available as well.

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton = null;
let webcamRunning = false;

// Before we can use HandLandmarker class we must wait for it to finish loading.
const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode,
    numHands: 2,
  });

  demosSection?.classList.remove("invisible");
};

createHandLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detect them
// upon click.
********************************************************************/

const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
  imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
  if (!handLandmarker) {
    console.log("Wait for handLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await handLandmarker.setOptions({ runningMode: "IMAGE" });
  }

  // Remove all landmarks drawn before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }

  const handLandmarkerResult = handLandmarker.detect(event.target);
  console.log(handLandmarkerResult?.handednesses?.[0]?.[0]);

  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.setAttribute("width", event.target.naturalWidth + "px");
  canvas.setAttribute("height", event.target.naturalHeight + "px");
  canvas.style =
    "left: 0px;" +
    "top: 0px;" +
    "width: " +
    event.target.width +
    "px;" +
    "height: " +
    event.target.height +
    "px;";

  event.target.parentNode.appendChild(canvas);

  const cxt = canvas.getContext("2d");
  for (const landmarks of handLandmarkerResult.landmarks) {
    drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5,
    });
    drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
  }
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement?.getContext("2d");

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton?.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam() {
  if (!handLandmarker) {
    console.log("Wait! Hand landmarker is not loaded yet.");
    return;
  }

  if (webcamRunning) {
    webcamRunning = false;
    socket.emit("reset_live");
    video.srcObject?.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    results = undefined;
    lastVideoTime = -1;
    if (enableWebcamButton) {
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    return;
  }

  socket.emit("reset_live");
  webcamRunning = true;
  lastVideoTime = -1;
  if (enableWebcamButton) {
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam, { once: true });
  } catch (error) {
    webcamRunning = false;
    if (enableWebcamButton) {
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    console.error("Unable to start webcam", error);
  }
}

let lastVideoTime = -1;
let results = undefined;

console.log(video);

function postLandmarks(landmarks, handednesses, timestampMs) {
  socket.emit("landmark_data", {
    landmarks: landmarks,
    handednesses: handednesses,
    timestampMs: timestampMs,
  });
}

function postEmptyVideoFeed(timestampMs) {
  socket.emit("landmark_data", {
    landmarks: [],
    handednesses: [],
    timestampMs: timestampMs,
  });
}

async function predictWebcam() {
  canvasElement.style.width = video.videoWidth + "px";
  canvasElement.style.height = video.videoHeight + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const sourceTimestampMs = video.currentTime * 1000;
  const isNewVideoFrame = lastVideoTime !== video.currentTime;

  if (isNewVideoFrame) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, sourceTimestampMs);

    if (results?.landmarks) {
      if (results.landmarks.length > 0) {
        postLandmarks(
          results.landmarks,
          results.handednesses,
          sourceTimestampMs,
        );
      } else {
        postEmptyVideoFeed(sourceTimestampMs);
        console.log("No landmarks detected");
      }
    }
  }

  // Draw every requestAnimationFrame using latest results
  canvasCtx.save();
  canvasCtx.clearRect(
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );

  if (results?.landmarks) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 5,
      });

      drawLandmarks(canvasCtx, landmarks, {
        color: "#FF0000",
        lineWidth: 2,
      });
    }
  }

  canvasCtx.restore();

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
