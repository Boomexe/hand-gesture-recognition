# Signbridge website

## What changed

The existing Flask website has a shared, responsive interface with warm paper,
forest-green controls, and small terracotta accents. No gradients on headings,
all-caps promotional labels, inflated accuracy claims, or simulated demo results.
Page introductions use compact, functional titles: Live studio, Video workspace,
and Gesture library. Slogan headings, introductory subtitles, the sidebar promo
card, and redundant practice/empty-state copy have been removed.

- **Live studio:** camera workspace, optional landmark overlay, preview mirroring,
  full-screen view, recognition scores, alternative predictions, and capture reset.
- **Session words:** completed gestures above the existing confidence threshold
  collect in a copyable, clearable list. This list stays in page memory and resets
  when you leave/reload. It is not sentence translation.
- **Video workspace:** drag-and-drop/file selection, preview, progress/error states,
  full-gesture classification, and optional detected-frame prefix inspection.
  A shared video template replaces the two old upload-only templates; both existing
  page URLs and prediction endpoints still work. Previous templates remain in Git history.
- **Gesture library:** searchable labels and lightweight category filters.
  The dictionary and count come from the server's existing class mapping:
  currently **52 labels**, despite the checkpoint's historical “fifty words” name.
- **Shared controls:** mobile navigation, help dialog, keyboard focus states,
  reduced-motion support, and actual model/connection status.

Practice selection only sets a target label. It never supplies a prediction.
Display names such as “Thank you” and “Drink” do not change model class IDs.
The library lists supported words; it does not teach the corresponding signs.

## Recognition behavior

No model architecture, checkpoint, training code, class mapping, preprocessing,
confidence threshold, or gesture-ending threshold was changed.
No retraining or dataset regeneration is required.

The camera still uses default browser video constraints, two-hand MediaPipe
tracking, original landmark coordinates/handedness, and source-video timestamps.
The existing server-side handedness workaround remains unchanged. Mirroring only
changes the displayed video/canvas. Detector timestamps remain increasing across
camera restarts without changing timestamps sent to the server.

Presentation/integration fixes:

- Socket.IO uses the current origin instead of hard-coded localhost:8000.
- MediaPipe loads when the camera is requested, with CPU fallback if GPU setup fails.
- Camera denial, missing camera, model absence, and disconnects have visible states.
- Completed predictions remain visible when hands leave view.
- The continuous video endpoint now includes the final detected frame. Previously
  its last prefix was one frame shorter than the complete gesture.
- Video uploads use unique temporary directories and are removed after analysis,
  including on failure. Pre-existing uploaded files are not removed.
- Upload requests have a size limit and JSON errors; preprocessing failures in
  continuous mode are handled consistently with full-video failures.

Webcam frames remain in the browser. Landmarks are sent to the Flask server;
the existing debug CSV writes remain enabled. Uploaded clips are sent to the server.
Confidence is the model's score, not a measured probability of correctness.

## Run

From the main project directory, with your existing environment:

```powershell
cd "D:\Coding\Projects\hand gesture recognition"
.\.venv\Scripts\Activate.ps1
python website\app.py
```

Open **http://localhost:8000**. Restart a running Flask process and hard-refresh
the browser after applying the redesign. No frontend build or npm install is needed.

An internet connection is needed for the existing MediaPipe model/runtime and
Socket.IO CDN. Google Fonts are optional; system fonts are the fallback.
Camera access requires localhost or HTTPS.

## Before recording

1. Open the app in a desktop browser at 1920 × 1080. Keep browser zoom at 100%.
2. Confirm “Model ready.” Open the library and pick a sign you know.
3. Turn on the camera and grant access yourself. Check lighting and hand visibility.
4. Test an entire sign, then lower your hands out of view to finish it.
5. Reset capture and clear session words before the take.
6. Demonstrate a few supported signs; do not describe this as full sign-language
   or sentence translation.
7. For the temporal-input demonstration, upload a short clip and enable
   “Frame-by-frame view.” Scrub the result to compare early prefixes with the
   full gesture. These points are detected-hand frames, not video playback times.

The redesign does not establish better recognition accuracy. Rehearse with the
actual webcam and checkpoint before recording.

## Verification

```powershell
python -m unittest discover -s tests -v
```

With a local server running on port 8765, installed Chrome, and Node 22+:

```powershell
node tests\test_website_browser.mjs
```

For a server on the normal app port instead:

```powershell
$env:WEBSITE_TEST_URL = "http://localhost:8000"
node tests\test_website_browser.mjs
```

The browser test uses an isolated temporary profile, no physical webcam, and
synthetic files/stubbed API responses only for UI interaction tests. Backend
tests separately verify real route behavior with prediction/preprocessing mocks.
Generated screenshots go in the already-ignored
`experiments/results/website_preview/` directory.

Verified: 31 Python tests; headless browser layouts at 1440, 1024, 768, 390,
and 320 pixels, plus a 1920 × 1080 recording layout; library search/filtering,
practice selection, help, mobile navigation, preview toggles, result formatting,
video full/prefix modes, timeline scrubbing, and upload error recovery.
Physical webcam recognition still needs a manual smoke test.
