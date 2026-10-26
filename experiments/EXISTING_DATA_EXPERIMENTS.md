# Experiments supported by the existing project data

## Scope and audit

This inventory distinguishes experiments using the current landmark arrays from experiments requiring new processing of already downloaded videos. Neither category requires recording new participants. Genuine live-webcam accuracy, end-to-end delay, and natural segmentation still require a suitable labeled live evaluation.

Update: the recommended 18-run feature/prefix suite has completed. See [the validation results](report_data/ABLATION_RESULTS.md) and [updated manuscript](../JHSS_PROJECT_ISSUES.md). The additional experiments in this inventory remain available directions, rather than completed evidence.

The inventory is supported by [the saved audit](results/existing_data_audit/audit.json) and [verified source links](results/existing_data_audit/source_links.json). Existing commands are documented in [the Section 4 runbook](SECTION4_RUNBOOK.md).

### Available artifacts

| Artifact | Verified availability | Use |
|---|---|---|
| Original ASL Citizen videos | 83,399 MP4 files | Re-extract positions, timestamps, hand presence, and global geometry |
| Official source manifests | Train, validation, test; participant IDs included | Source vocabulary, signer groups, clip labels, split membership |
| HDF5 sequences, all labels | 40,135 train; 10,302 validation; 32,902 test | Current feature experiments; wider-vocabulary or held-out-class comparisons |
| Current 52-class HDF5 subset | 770 train; 189 validation; 651 test | Controlled comparisons with the current vocabulary |
| Existing classifier | 52 outputs with embedded label map | Prefix, confidence, error, and model-execution baselines |
| Existing raw landmark CSVs | Exact content matches for 1,394 selected HDF5 sequences | Verified clip/signer association for the matched subset |
| Saved uploaded webcam video | One readable 182-frame recording, about 29.97 FPS | Individual replay case; label/boundaries require confirmation |
| Legacy project material | Present under `.old_stuff`; not validated for this protocol | Excluded from the controlled dataset until separately audited |

For the 52-class subset, training lengths range from 2 to 312 detected frames (median 29), validation from 6 to 91 (median 30), and test from 3 to 95 (median 27). Training support ranges from 12 to 18 clips per class. A train/validation/test split count is not a count of independent signers.

Every selected HDF5 row passed checks for finite features, binary hand presence, at least one hand per retained frame, and the expected 255-channel layout. Wrist position channels are zero throughout, and frame indexes are contiguous detected-frame indexes. All selected sequences contain some nonzero stored motion. These facts do not prove the timestamp convention used to generate motion.

No exact full-array feature hashes overlap between selected splits. This check does not rule out every possible duplicate or leakage mechanism.

The official split manifests contain 35 training, 6 validation, and 11 test signers with no signer overlap. The selected vocabulary covers 32, 5, and 11 source signers, respectively. Content matching verifies 664 training, 163 validation, and 567 test HDF5/source associations. The other 106/26/84 associations remain unverified; do not assign their signer IDs by guessing row order. Signer-clustered analysis of the verified subset must report its reduced coverage and class composition.

## A. Experiments using current HDF5 arrays

| Experiment | Requires retraining? | Supported comparison and limitation | Tool readiness |
|---|---|---|---|
| Complete-sequence classification | No | Accuracy, top-k, per-class F1, confusions | Implemented and evaluated for the current checkpoint |
| Prefix completeness | No for a baseline | 10%–100% of detected frames; fixed frame caps | Implemented and evaluated |
| Position-only versus motion features | Yes | Positions, frame differences, stored motion | Trainer and evaluator implemented |
| Hand-scale normalization | Yes | Wrist-centered versus scale-normalized positions | Implemented; neither condition restores global position |
| Legacy schema versus feature-only input | Yes | Current 510 channels versus cleaned feature views | Implemented; metadata and parameter-count changes are confounders to report |
| Complete-only versus prefix training | Yes | Same representation/model with and without prefix augmentation | Implemented; choose the protocol before testing |
| Repeated-seed variation | Yes | Repeat the same configurations, report means and variation | Six configurations across three seeds completed on validation |
| Non-recurrent pooled baseline | Yes | Does temporal ordering help over a pooled feature baseline? | Existing arrays sufficient; model implementation needed |
| Unidirectional versus bidirectional LSTM | Yes | Causal recurrence versus repeated full-prefix recomputation | Existing arrays sufficient; model implementation needed |
| With versus without attention | Yes | Contribution of attention with other settings controlled | Existing arrays sufficient; architecture variant needed |
| Pooling strategy | Yes | Mean versus final-state or learned temporal pooling | Existing arrays sufficient; model variant needed |
| Prefix-only resampling | Yes for the main comparison | Resample only observed positions; recompute frame differences afterward | Transform/configuration needed; detected-frame time only |
| Sliding detected-frame windows | Yes for a fair design comparison | Context retained versus bounded input length | Transform/configuration needed; windows cannot be called milliseconds |
| Recognition versus sequence length | No | Stratify current errors by detected-frame length | Can derive from saved predictions and HDF5 |
| Recognition versus hand visibility | No | Stratify by one/two-hand presence and changes | Can derive; presence is not a linguistic class annotation |
| Calibration and selective classification | No full retraining | Fit calibration/thresholds on validation, apply frozen policy to test | Additional logits/probability collection needed for temperature calibration |
| Confidence/coverage at existing threshold | No | Current score threshold, accepted accuracy, fraction accepted | Already measured; threshold not calibrated |
| Prefix prediction stability | No | Changes and stable correctness at sampled completion points | Existing predictions support a coarse analysis; all-prefix replay needed for frame-level decisions |
| Unseen-class rejection | No initially | Other vocabulary labels as non-target gestures | Labels available in both validation and test; implement held-out-class scoring and fix threshold on validation |
| Synthetic feature-corruption sensitivity | No initially; augmentation comparisons require training | Controlled noise or missing-observation probes | Define precisely; not evidence of natural webcam robustness |
| Training-set size | Yes | Fixed per-class sampling budgets, same held-out set | `--limit-per-class` currently limits validation too; change to train-only sampling before this experiment |
| Parameter/computation comparison | Model-dependent | Counts and batch-1 execution versus prefix length | Parameter counts recorded; timed benchmark still needed |
| Paired clip-level statistical comparison | No | Compare correctness on identical clips | Per-clip predictions available; clustered version needs verified signer links |

The six feature/prefix configurations have now been trained and evaluated across three seeds. The saved two-clips-per-class runs remain smoke tests and were excluded from the 18 reported runs. Architecture, sequence-transform, and live-policy experiments remain unperformed.

### Valid comparisons from these arrays

The existing feature-only variants keep hand presence fixed and remove the capture index from the learned input. They do not recover absolute wrist location. Compare `positions` with `scaled-positions` to test hand-scale normalization, and `positions` with `positions-diff` or `positions-motion` to test motion information. Report the changed input dimensions and resulting parameter counts.

Use the current stored-motion view as a representation whose numeric values are fixed. Claiming those channels are correctly normalized physical or image-coordinate velocity requires a provenance check. A new extraction with explicitly saved timestamps is a stronger foundation for sampling-rate experiments.

Dropping frames while retaining precomputed motion is not equivalent to rerunning landmark detection at a slower frame rate. It may be described as losing observations from an already computed stream, if that is the intended experiment. Do not conflate those two processes.

Corruption experiments must preserve the model's valid-frame assumptions. An internal all-absent row is not ordinary trailing batch padding. Decide whether the system discards it, explicitly models a missing timestep, or resets motion; apply that policy consistently.

## B. Experiments using original videos already on disk

These require new landmark extraction, but no new video collection. All original videos belonging to the current 52-class vocabulary are present: 771 train, 189 validation, and 651 test. One training clip is absent from the selected HDF5 subset; retain a log of detection failures and exclusions during regeneration.

| Experiment | What to regenerate or label |
|---|---|
| Raw coordinates versus wrist centering | Save absolute coordinates first, derive both representations from the same observations |
| Local versus global/inter-hand geometry | Retain absolute wrist coordinates, relative wrist displacement, and hand presence |
| Time-aware motion versus frame differences | Save actual extraction timestamps/source frame indexes and recompute both feature views |
| Frame-rate robustness | Sample each original video at defined rates, rerun extraction, and recompute motion from retained timestamps |
| Occlusion/detection-gap behavior | Retain every capture timestep, including no-hand observations; label or distinguish tracker failures from actual gesture completion |
| Orientation/hand-slot robustness | Replay identical frames under explicitly defined conventions; keep class semantics and handedness labels consistent |
| Body/face feature extension | Original RGB video is available; add a separate extraction/model path if justified by the study scope |
| Signer-separated evaluation | Store clip name and source participant ID alongside each regenerated sequence |
| Signer/session-clustered intervals | Regenerated identities support clustered evaluation; avoid counting individual frames as independent trials |
| Recorded-video decision latency | Reconstruct timestamps and annotate relevant boundaries; report as video-replay timing rather than real webcam latency |
| Boundary-policy comparison | Annotate start/end in recorded clips and retain idle frames; natural between-sign streams are not supplied by isolated clips |

Extraction should write absolute positions, timestamps, capture indexes, presence, clip ID, signer ID, original split, and labels to a new output directory. Preserve the existing HDF5 files. Use one extraction to derive all feature variants, rather than comparing inconsistent detection passes or code versions.

Orientation normalization is a representation choice; do not assume arbitrary reflections preserve every class's semantics. Artificially concatenating isolated clips can test a controlled synthetic stream but cannot establish natural continuous-signing performance.

## C. What these data cannot establish by themselves

- Accuracy across newly recorded webcam sessions, lighting, camera placement, and new participants.
- Camera/network/display delay during actual live operation.
- Natural false activations and boundary errors during unscripted inter-gesture activity.
- Continuous-ASL or sentence-translation performance.
- A verified signer-clustered analysis of every existing HDF5 clip while 216 source associations remain unresolved.

One saved upload and isolated source videos are useful diagnostics, but are not a representative labeled live dataset.

## Recommended finite experiment set

Prioritize the study's primary question rather than executing every possible comparison:

1. Freeze the current 52 labels and audit their split/sample identities.
2. Train the five existing feature variants with seeds 17, 42, and 73.
3. Add position-only prefix-augmented training for the same seeds. Its complete-only controls are already included in step 2: six configurations per seed, **18 training runs total**.
4. Evaluate completion curves and final performance on validation; freeze the comparisons before their final test evaluation.
5. Compare accuracy with coverage and prediction stability. Use validation to choose thresholds and policies.
6. Regenerate a richer 52-class extraction from existing videos if raw-position, inter-hand geometry, or elapsed-time claims remain central to the paper.

Repeated-seed variation measures training variability; it is not a replacement for uncertainty across signers. The earlier inspected test baseline should be disclosed as exploratory where it informs design choices.

The feature-training commands and evaluator are in [SECTION4_RUNBOOK.md](SECTION4_RUNBOOK.md). To reproduce the inventory without modifying data:

```powershell
& 'D:\Coding\Projects\hand gesture recognition\.venv\Scripts\python.exe' experiments/audit_existing_data.py --project-root 'D:\Coding\Projects\hand gesture recognition' --parameters 'D:\Coding\Projects\hand gesture recognition\parameters.json' --output experiments/results/existing_data_audit_repeat
```

The inventory audit and subsequent 18-run validation suite are complete. No independent test-set evaluation of the new models or labeled live-webcam evaluation has been performed.
