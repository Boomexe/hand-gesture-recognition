// Headless app tests. Uses installed Chrome and Node 22+; no npm dependencies.
// No physical camera access, real uploads, or fabricated product/demo results.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const base = process.env.WEBSITE_TEST_URL || "http://127.0.0.1:8765";
const chrome = process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "signbridge-browser-test-"));
const artifacts = path.resolve("experiments/results/website_preview");
fs.mkdirSync(artifacts, { recursive: true });
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const browser = spawn(chrome, ["--headless=new", "--no-first-run", "--no-default-browser-check",
  "--remote-debugging-port=0", "--user-data-dir=" + profile, "about:blank"],
  { windowsHide: true, stdio: "ignore" });
let ws;
let nextId = 0;
const pending = new Map();
const errors = [];
const failures = [];
function send(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error("CDP timeout: " + method)); }, 20000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const reply = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (reply.exceptionDetails) throw new Error(JSON.stringify(reply.exceptionDetails));
  return reply.result.value;
}
async function until(expression, label) {
  for (let i = 0; i < 150; i++) {
    if (await evaluate(expression)) return;
    await delay(100);
  }
  throw new Error("Timed out: " + label);
}
async function navigate(route) {
  await send("Page.navigate", { url: base + route });
  await until("document.readyState === 'complete' && !!document.querySelector('#main')", "page load " + route);
  await evaluate("document.fonts.ready");
  await delay(250);
}
async function screenshot(name) {
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  fs.writeFileSync(path.join(artifacts, name + ".png"), Buffer.from(data, "base64"));
}
async function noOverflow() {
  assert.equal(await evaluate("document.documentElement.scrollWidth <= innerWidth"), true, "Horizontal overflow");
}
try {
  let port;
  for (let i = 0; i < 100; i++) {
    const portFile = path.join(profile, "DevToolsActivePort");
    if (fs.existsSync(portFile)) { port = fs.readFileSync(portFile, "utf8").split("\n")[0]; break; }
    if (browser.exitCode !== null) throw new Error("Headless Chrome exited " + browser.exitCode);
    await delay(100);
  }
  if (!port) throw new Error("Chrome did not expose a test port");
  const pages = await (await fetch("http://127.0.0.1:" + port + "/json/list")).json();
  ws = new WebSocket(pages.find((page) => page.type === "page").webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.addEventListener("open", resolve, { once: true }); ws.addEventListener("error", reject, { once: true }); });
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const call = pending.get(message.id);
      if (!call) return;
      pending.delete(message.id);
      clearTimeout(call.timer);
      if (message.error) call.reject(new Error(JSON.stringify(message.error))); else call.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails);
    else if (message.method === "Network.loadingFailed" && !message.params.canceled) failures.push(message.params.errorText);
  });
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Network.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false });
  await navigate("/");
  await until("typeof window.io === 'function'", "Socket.IO library");
  await noOverflow();
  assert.equal(await evaluate("document.querySelector('#engine-label').textContent"), "Model ready");
  await screenshot("live-desktop");
  await evaluate("document.querySelector('[data-practice=MORE]').click()");
  assert.equal(await evaluate("document.querySelector('#practice-target').textContent"), "Practicing: More");
  assert.equal(await evaluate("document.querySelector('#confidence-value').textContent"), "—", "Practice must not generate predictions");
  await evaluate("document.querySelector('#mirror-toggle').click()");
  assert.equal(await evaluate("document.querySelector('#webcam').classList.contains('mirrored')"), false);
  await evaluate("document.querySelector('#landmark-toggle').click()");
  assert.equal(await evaluate("document.querySelector('#landmark-toggle').getAttribute('aria-pressed')"), "false");
  await evaluate("document.querySelector('[data-open-help]').click()");
  assert.equal(await evaluate("document.querySelector('#help-dialog').open"), true);
  await screenshot("help-desktop");
  await evaluate("document.querySelector('[data-close-help]').click()");
  await navigate("/library");
  const total = await evaluate("document.querySelectorAll('.gesture-tile').length");
  assert.equal(total, 52);
  await screenshot("library-desktop");
  await evaluate("document.querySelector('#gesture-search').value='thank you'; document.querySelector('#gesture-search').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate("document.querySelectorAll('.gesture-tile:not([hidden])').length"), 1);
  await evaluate("document.querySelector('#gesture-search').value='zzzzzz'; document.querySelector('#gesture-search').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate("document.querySelector('#library-empty').hidden"), false);
  await evaluate("document.querySelector('#reset-search').click(); document.querySelector('[data-category=questions]').click()");
  assert.equal(await evaluate("document.querySelectorAll('.gesture-tile:not([hidden])').length"), 6);
  await navigate("/?practice=THANKYOU");
  assert.equal(await evaluate("document.querySelector('#practice-target').textContent"), "Practicing: Thank you");
  // Verify result formatting in isolation, then restore. Never shipped as demo data.
  await evaluate("(async()=>{ const ui=await import('/static/js/ui.js');ui.renderPrediction({predicted_word:'MORE',confidence:.95,top_predictions:[{word:'MORE',confidence:.95},{word:'HELLO',confidence:.04}]},{final:true}); })()");
  assert.equal(await evaluate("document.querySelector('#prediction').textContent"), "More");
  assert.equal(await evaluate("document.querySelector('#confidence-value').textContent"), "95.0%");
  await evaluate("(async()=>{const ui=await import('/static/js/ui.js');ui.resetPrediction()})()");
  await navigate("/by-video");
  assert.equal(await evaluate("document.querySelector('#prefix-mode').checked"), false);
  await screenshot("video-desktop");
  // Test-only synthetic file and local API stub; never reads user videos.
  await evaluate(`(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(["test-only"], "sample.mp4", {type:"video/mp4"}));
    document.querySelector("#video-input").files = transfer.files;
    document.querySelector("#video-input").dispatchEvent(new Event("change"));
    window.__originalFetch = window.fetch;
    window.fetch = async (url, options) => {
      if (url === "/upload-video" && options.method === "POST") return new Response(JSON.stringify({predicted_word:"MORE",confidence:.95}),{headers:{"Content-Type":"application/json"}});
      return window.__originalFetch(url, options);
    };
  })()`);
  await evaluate("document.querySelector('#analyze-video').click()");
  await until("document.querySelector('#upload-status').textContent === 'Analysis complete'", "full video result");
  assert.equal(await evaluate("document.querySelector('#prediction').textContent"), "More");
  await evaluate(`(() => {
    document.querySelector("#prefix-mode").click();
    window.fetch = async (url, options) => {
      if (url === "/upload-video-continuous") return new Response(JSON.stringify([{pred_word:"HELLO",confidence:.5,frame:0},{pred_word:"MORE",confidence:.95,frame:1}]),{headers:{"Content-Type":"application/json"}});
      return window.__originalFetch(url, options);
    };
    document.querySelector("#analyze-video").click();
  })()`);
  await until("!document.querySelector('#timeline-card').hidden", "prefix timeline");
  assert.equal(await evaluate("document.querySelector('#prediction').textContent"), "More");
  assert.equal(await evaluate("document.querySelector('#frame-slider').value"), "1");
  await evaluate("document.querySelector('#frame-slider').value=0;document.querySelector('#frame-slider').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate("document.querySelector('#prediction').textContent"), "Hello");
  assert.equal(await evaluate("document.querySelector('#confidence-value').textContent"), "50.0%");
  await evaluate(`(() => {
    window.fetch = async () => new Response(JSON.stringify({message:"Prediction failed",error:"No hands detected in video."}),{status:500,headers:{"Content-Type":"application/json"}});
    document.querySelector("#analyze-video").click();
  })()`);
  await until("!document.querySelector('#upload-error').hidden && !document.querySelector('#analyze-video').disabled", "upload error recovery");
  assert.equal(await evaluate("document.querySelector('#timeline-card').hidden"), true);
  assert.equal(await evaluate("document.querySelector('#upload-error').textContent"), "No hands were detected. Try a clearer clip.");
  await navigate("/by-video-continuous");
  assert.equal(await evaluate("document.querySelector('#prefix-mode').checked"), true);
  for (const width of [1024, 768, 390, 320]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: 1, mobile: false });
    for (const route of ["/", "/by-video", "/library"]) {
      await navigate(route);
      await noOverflow();
      if (route === "/") {
        assert.equal(await evaluate("(() => { const card=document.querySelector('.camera-card').getBoundingClientRect();const stage=document.querySelector('#camera-stage').getBoundingClientRect(); return stage.left >= card.left && stage.right <= card.right; })()"), true, "Camera stage exceeds its card at " + width);
        assert.equal(await evaluate("(() => { const stage=document.querySelector('#camera-stage').getBoundingClientRect();const button=document.querySelector('#camera-empty-button').getBoundingClientRect(); return button.bottom <= stage.bottom && button.top >= stage.top; })()"), true, "Camera prompt clipped at " + width);
      }
      if (width === 390) await screenshot(route === "/" ? "live-mobile" : route === "/library" ? "library-mobile" : "video-mobile");
    }
  }
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: false });
  await navigate("/");
  await evaluate("document.querySelector('#menu-button').click()");
  assert.equal(await evaluate("document.querySelector('#sidebar').inert"), false);
  await evaluate("document.querySelector('#nav-scrim').click()");
  assert.equal(await evaluate("document.querySelector('#sidebar').inert"), true);
  await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await navigate("/");
  await noOverflow();
  await screenshot("live-recording-1080p");
  assert.equal(errors.length, 0, "Browser errors: " + JSON.stringify(errors));
  console.log(JSON.stringify({ status: "passed", supportedGestures: total, widths: [1440,1024,768,390,320], browserErrors: errors, networkFailures: failures, artifacts }, null, 2));
} finally {
  if (ws?.readyState === WebSocket.OPEN) {
    await send("Browser.close").catch(() => {});
    ws.close();
  }
  if (browser.exitCode === null) browser.kill();
  await delay(500);
  // Only our uniquely-created test profile; never touch the user's browser profile.
  assert.ok(path.basename(profile).startsWith("signbridge-browser-test-"));
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 250 });
}
