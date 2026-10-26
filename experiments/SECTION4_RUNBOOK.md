# Running the Section 4 experiments

This guide turns Section 4 into a sequence of runnable evaluations and training comparisons. Start with the existing 52-class vocabulary. Do not mix results from its checkpoint with the old 13-class or 2,731-class checkpoints.

The 18-run feature/prefix suite has now completed. Its three-seed validation findings are in [ABLATION_RESULTS.md](report_data/ABLATION_RESULTS.md), with the updated interpretation in [the manuscript](../JHSS_PROJECT_ISSUES.md). The earlier-checkpoint validation/test figures below are a separate reference. The commands remain useful for reproduction; no new-model test-set or webcam evaluation has been performed.

The subsequent [existing-data inventory](EXISTING_DATA_EXPERIMENTS.md) confirms that all 83,399 source videos and the signer-labeled official split manifests are already on disk. Experiments needing raw/global positions can therefore use new extraction from existing videos without new recordings. The audit also verifies source links for 1,394 of the 1,610 selected HDF5 sequences; 216 remain unlinked. Read that inventory for the distinction between source-manifest separation and verified provenance of cached features.

## What was run

The existing `hand_gesture_modelv11_fifty_words.pt` contains **52 outputs**, despite its filename. Its embedded ordered label map matches the current external parameters. Evaluation used every matching clip in the supplied split, without a sample limit.

| Evaluation | Samples | Top-1 accuracy | Macro F1 |
|---|---:|---:|---:|
| Complete validation clips | 189 | 67.72% | 0.6437 |
| Complete test clips | 651 | 59.29% | 0.5883 |
| First 10% of detected test frames | 651 | 4.76% | 0.0297 |
| First 30% | 651 | 22.43% | 0.2205 |
| First 50% | 651 | 41.01% | 0.3945 |
| First 70% | 651 | 56.99% | 0.5627 |
| First 80% | 651 | 59.29% | 0.5846 |
| First 90% | 651 | 60.37% | 0.5979 |

The validation result reproduces the accuracy stored in the checkpoint. The earlier 89.03% measurement does not describe this checkpoint.

The test HDF5 file contains 32,902 clips across the wider vocabulary. Only its 651 clips matching the checkpoint's 52 classes were evaluated; the other classes were not treated as mistakes or unknown-input examples.

At the existing confidence threshold of 0.8, full-sequence accepted accuracy is **86.88%**, with **33.95% coverage** (221 accepted clips). Overall accuracy is still 59.29%. Do not report accepted accuracy without coverage. At 10% completion, accepted accuracy is only 12.16%, illustrating that this threshold does not make early predictions reliable.

At 90% completion, 22 clips were correct but became wrong at completion, while 15 wrong clips became correct. The seven-clip net change does not establish a significant benefit of truncation; do not select a cutoff using these test results. At 80%, accuracy equals complete-sequence accuracy, but 31 predictions improve and another 31 worsen, so the predictions are not identical.

Numerical results, per-class reports, confusion matrices, per-clip predictions, hashes, and the prefix plot are in:

- [Test results](results/v11_52classes_test/summary.json)
- [Validation results](results/v11_52classes_validation/summary.json)
- [Paired prediction changes](results/v11_52classes_test/paired_correctness.json)
- [Prefix plot](results/v11_52classes_test/prefix_accuracy.png)
- [Full-sequence per-class report and confusion matrix](results/v11_52classes_test/fraction_1.00.json)

## Review of Section 4

| Section | What it tests | Can use existing checkpoint? | What remains |
|---|---|---|---|
| 4.1 Protocol | Vocabulary, split, reproducibility | Yes, artifact audit completed | Official signer groups are disjoint; recover the remaining 216 cached-feature source associations |
| 4.2 Normalization | Centered versus scaled positions | Three-seed validation comparison completed | Raw/global positions require regenerating richer source data |
| 4.3 Motion | Positions, differences, stored motion | Three-seed validation comparisons completed | Audit motion provenance; sampling tests need original timestamps |
| 4.4 Sequence/model | Windows, resampling, recurrent/attention variants | Existing model is a baseline | Implement alternative sequence transforms and model classes, then retrain |
| 4.5 Prefixes | Accuracy under partial observations | Baseline and three-seed training comparison completed | Independent test and live confirmation |
| 4.6 Live policies | Accuracy, coverage, stability, finalization | Classifier can be reused | Collect labeled continuous webcam streams and actual timestamps |
| 4.7 Metrics | Per-class metrics, uncertainty, comparison | Yes; reports produced | Clustered intervals can use verified links with subset limitations, or a richer regenerated dataset; live timing needs recording |

The outline is too broad to run as one undifferentiated experiment. Use the controlled comparisons below first, with the same vocabulary, samples, training budget, seeds, and validation selection rule. Test data must not select hyperparameters or thresholds. Since its existing baseline results have now been inspected, document them as exploratory if they inform later design choices; reserve new webcam sessions or another untouched evaluation set for final confirmation.

## 1. Set up PowerShell

Run in the main project directory, which now contains the merged experiment scripts, results, source data, and environment.

```powershell
Set-Location 'D:\Coding\Projects\hand gesture recognition'
$experimentPython = 'D:\Coding\Projects\hand gesture recognition\.venv\Scripts\python.exe'
$experimentData = 'D:\Coding\Projects\hand gesture recognition\asl_citizen_gesture_data'
$experimentParameters = 'D:\Coding\Projects\hand gesture recognition\parameters.json'
$experimentCheckpoint = 'D:\Coding\Projects\hand gesture recognition\models\hand_gesture_modelv11_fifty_words.pt'
```

Freeze a copy of the parameters file with your experiment archive before regenerating the notebook's parameters. Every new checkpoint embeds this map. The evaluation refuses an external map that differs from the checkpoint. Keep the copied vocabulary constant across runs.

## 2. Reproduce the current prefix baseline

```powershell
& $experimentPython experiments/evaluate_prefixes.py --checkpoint $experimentCheckpoint --data "$experimentData/cleaned_test_data.h5" --parameters $experimentParameters --output experiments/results/baseline_test_repeat --device cuda
```

This evaluates 10% through 100% completion, plus 8-, 13-, 20-, and 31-frame caps. Shorter clips retain their complete sequence at a larger cap; the report counts those clips. Output directories must be new to prevent overwriting earlier results.

For a quick pipeline check, add `--limit-per-class 2`. This uses a seeded sample and must not be reported as the full benchmark.

Completion means proportion of **detected frames**, not elapsed seconds. The HDF5 data does not retain original timestamps. These are recorded-clip prefix results, not measured webcam accuracy or latency.

## 3. Train controlled feature comparisons

The new trainer preloads only the selected vocabulary into memory, uses zero workers, and trains a new model from scratch. It saves immutable epoch checkpoints, a `best_checkpoint.json` pointer, `history.json`, and `settings.json` in an isolated experiment directory. This avoids overwriting a checkpoint that Windows may have memory-mapped. It does not modify the notebook, application model, or parameters file. The evaluator accepts the run directory and resolves its best checkpoint; old `best.pt` runs remain supported.

| Variant | Input channels | Question answered |
|---|---:|---|
| `legacy` | 510 | How does the existing feature schema perform under the controlled training protocol? |
| `positions` | 129 | What is achieved by wrist-centered positions and hand presence alone? |
| `scaled-positions` | 129 | Does per-frame hand-scale normalization improve position-only recognition? |
| `positions-diff` | 255 | Do causal frame differences improve over positions alone? |
| `positions-motion` | 255 | Do stored timestamp-based motion channels improve over positions alone? |

Feature-only variants reserve a zero column at index 0 and retain presence at indexes 1 and 2 for the existing model mask. Capture-frame indexes are excluded from their learned input. This means a legacy-versus-clean comparison changes metadata inclusion as well as motion representation; do not attribute the whole difference solely to velocity. Input dimension also changes the first LSTM's parameter count, which is recorded for interpretation.

The scale measure is the per-frame norm of the wrist-relative middle-finger MCP coordinate (landmark 9), with a lower bound of 1e-6. This is coordinate-space normalization, not physical hand-size measurement. Scaling is causal and performed only on observed frames. Motion resets when a hand is absent or newly present. `positions-motion` reuses stored motion values; its interpretation depends on verifying how the source HDF5 was generated.

Example position-only run:

```powershell
& $experimentPython experiments/train_features.py --data-dir $experimentData --parameters $experimentParameters --output experiments/results/positions_s42 --variant positions --seed 42 --device cuda
```

Run the five variants with three seeds:

```powershell
foreach ($experimentSeed in 17, 42, 73) {
    foreach ($experimentVariant in 'legacy', 'positions', 'scaled-positions', 'positions-diff', 'positions-motion') {
        $experimentRun = "experiments/results/${experimentVariant}_s${experimentSeed}"
        & $experimentPython experiments/train_features.py --data-dir $experimentData --parameters $experimentParameters --output $experimentRun --variant $experimentVariant --seed $experimentSeed --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $experimentRun" }
    }
}
```

Defaults are batch size 32, Adam at 0.001, up to 150 epochs, patience 15, clipping norm 1.0, and best complete-sequence validation loss selection. Specify identical overrides for every compared run. Seeds are recorded, but exact GPU determinism is not guaranteed. These are 15 training runs; use a single seed first to check the pipeline, then complete the repetitions for the final comparison.

Evaluate each trained model using its own feature schema, read automatically from its metadata:

```powershell
& $experimentPython experiments/evaluate_prefixes.py --checkpoint experiments/results/positions_s42 --data "$experimentData/cleaned_val_data.h5" --output experiments/results/positions_s42_validation --device cuda
```

Use validation to select the representation and any settings. Freeze the comparison plan before running the test command:

```powershell
& $experimentPython experiments/evaluate_prefixes.py --checkpoint experiments/results/positions_s42 --data "$experimentData/cleaned_test_data.h5" --output experiments/results/positions_s42_test --device cuda
```

Changing features at inference on the old checkpoint is not an ablation. All feature variants require retraining.

## 4. Train complete-only versus prefix-augmented models

Use the same chosen representation and seed in both conditions. For example:

```powershell
& $experimentPython experiments/train_features.py --data-dir $experimentData --parameters $experimentParameters --output experiments/results/positions_full_s42 --variant positions --prefix-probability 0 --seed 42 --device cuda
& $experimentPython experiments/train_features.py --data-dir $experimentData --parameters $experimentParameters --output experiments/results/positions_prefix_s42 --variant positions --prefix-probability 0.5 --seed 42 --device cuda
```

In the augmented condition, half the sample presentations select a prefix length uniformly from 1 through the clip's detected-frame length; the rest retain the complete clip. Features are constructed after truncation. Validation uses complete clips in both conditions and selects best validation loss. These settings are explicit experimental choices; select alternatives on validation data rather than on test prefix curves.

Evaluate both checkpoints with the same evaluator, fractions, and held-out clips. Report the whole completion curve and final accuracy. Repeat with the same three seeds. Tiny prefixes can be ambiguous; do not assume their labels are inferable merely because they inherit a source clip's label.

## 5. Experiments requiring richer data

The existing wrist-centered HDF5 rows cannot recover absolute wrist positions or inter-hand displacement. Integrating velocity does not recover the missing starting position, and stored files have no timestamps to support it anyway. Regenerate from the original videos rather than inventing those coordinates. The source videos are present locally, so this step does not require collecting new recordings.

For each newly extracted clip, save original landmark coordinates, hand presence, canonical handedness, capture-frame index, timestamps, clip ID, signer ID, source split, and class label. Compute raw, centered, and centered-plus-global representations from this common extraction. Fit any dataset-level normalization on training data only.

Before comparing robustness to frame rate, recompute motion from the retained timestamps after subsampling. Keeping precomputed full-rate motion while dropping frames would not simulate lower-rate extraction correctly.

The current scripts do not implement window/resampling variants or alternative architectures. To run Section 4.4, define separate configurations for fixed frame/time windows, prefix-only resampling, pooled non-recurrent input, unidirectional recurrence, and BiLSTM without attention. Train each with the same split and feature representation. Record parameter counts and measure model inference with batch size 1, GPU synchronization, warm-up, and preprocessing timed separately. Never use unseen future frames to construct a prefix.

## 6. Live policy experiment

Collect labeled webcam sessions containing complete gestures, inter-gesture idle periods, internal pauses, occlusion, hands-visible endings, and non-vocabulary movements. Save landmarks even during empty frames, actual timestamps, gesture start/end labels, signer/session IDs, and class labels. Choose a sample size based on the desired uncertainty and participant/session variation, rather than treating individual frames as independent samples.

Replay every session through each policy using the same classifier. Compare the existing rule with minimum duration, calibrated confidence rejection, agreement across consecutive predictions, and time-based end detection. Choose thresholds using separate validation sessions.

For each gesture, record the first accepted class, its timestamp, whether it is correct, the final class, and finalization time. Report accepted accuracy, coverage, time to correct decision, premature finalization, missed completion, and false activations during idle/unknown input. Detection gaps must not be confused with gesture completion by labeling them as equivalent in the ground truth.

Run actual live captures separately to measure end-to-end latency; recorded-clip replay does not measure camera/transport delay. Bootstrap by signer or session for final confidence intervals. The evaluator's Wilson intervals assume independent clips and are descriptive, not a substitute for clustered analysis.

## Implementation verification

Seven unit tests verify prefix isolation from future frames, class-map compatibility, sequence validity, feature layout, scale invariance, and motion resets after a hand disappears. A one-epoch, two-clips-per-class training run is a smoke test only and must not enter the research results.

```powershell
& $experimentPython -m unittest discover -s tests -p test_prefix_evaluation.py -v
```
