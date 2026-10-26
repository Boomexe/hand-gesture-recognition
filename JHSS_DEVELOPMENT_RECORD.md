# Development Record for the Hand Gesture Recognition Project

> Working manuscript documentation for the *Journal of High School Science* (JHSS). The project objective is to build a real-time hand gesture recognition model and improve its accuracy, with particular attention to predictions made before a gesture is complete. The technical problems encountered during development are included as evidence motivating the experimental design. This is not yet a submission-ready manuscript.

## Journal positioning and formatting notes

This work is best framed as an **original paper** if the final manuscript includes controlled experiments on a labeled test set and a separately collected live-webcam set. The manuscript should investigate how a recognition system can preserve accuracy while receiving only an incomplete temporal prefix of a gesture. The engineering problems are not the central objective; they explain the design decisions and define the variables tested in the study.

Before submission, transfer the manuscript to a Word or LibreOffice document and check the current [JHSS author guidance](https://jhss.scholasticahq.com/for-authors). The journal's published guidance and recent editorial comments indicate that the document should use a single column, 12-point Times New Roman font, 1.15 line spacing, 1-inch left and right margins, 0.5-inch top and bottom margins, no first-line indentation, and a blank line between paragraphs. Main headings and subheadings should be numbered manually. References should be numbered manually in order of appearance, cited in curved parentheses, formatted in APA style, and include a live DOI or source link when available.

The title, author names, school affiliations, contact information, acknowledgments, and any ethics statement must be completed by the author. Do not submit the development measurements below as final results until they have been reproduced from a frozen version of the code and model.

## Abstract

Real-time hand gesture recognition systems must make decisions while a gesture is still unfolding, whereas conventional video classifiers typically receive a complete sequence. This project develops a landmark-based isolated-gesture recognition system using MediaPipe hand landmarks and a bidirectional long short-term memory network with attention, and investigates how the system can maintain accuracy despite incomplete temporal input. The study examines three connected factors: the amount of temporal context available at prediction time, the effects of spatial normalization and motion features, and the consistency of preprocessing and temporal modeling across training and deployment. A variable-length input pipeline with masked batching was introduced to remove dependence on a fixed 93-frame sequence. Preliminary prefix experiments showed that accuracy increased from 47.74% with the first eight frames to 89.03% with the complete sequence, indicating that early predictions were substantially less reliable. Development also revealed that mirrored camera geometry, unstable handedness ordering, inconsistent coordinate and velocity calculations, and checkpoint metadata could obscure the effect of temporal completeness. In one paired webcam example, the uploaded-video path classified the sign MORE with greater than 0.99 confidence after the initial frames, whereas the live path favored HELLO; swapping live hand slots or mirroring landmark geometry changed the prediction to MORE with greater than 0.99 confidence. These observations motivate a controlled comparison of spatial representations, motion features, sequence models, and decision strategies at different stages of gesture completion. A labeled live-data evaluation is required to determine which combination best preserves real-time accuracy.

**Keywords:** sign recognition, American Sign Language, hand landmarks, recurrent neural network, variable-length sequence, real-time inference, domain shift, PyTorch

## 1. Introduction

The project aims to build an accurate model for recognizing isolated hand gestures from webcam video. Each frame is processed to obtain hand landmarks, and the resulting temporal sequence is classified by a recurrent neural network. Unlike prerecorded-video classification, a live system receives only a partial sequence at the beginning of a gesture and must decide whether to predict, wait for more evidence, or reject an uncertain input. Prerecorded-video inference appeared reliable, while live inference was substantially less accurate. This discrepancy motivated an investigation of temporal completeness and of the spatial and motion representations supplied to the model.

The primary research question is:

> **How can real-time gesture recognition systems be designed to maintain accuracy despite incomplete temporal input?**

Two supporting research questions are:

> **How do spatial normalization and temporal motion features affect the accuracy of hand landmark-based gesture recognition?**

> **What preprocessing and temporal modeling strategies improve robustness in landmark-based gesture recognition systems?**

The working hypothesis is that real-time accuracy can be improved by combining a representation invariant to irrelevant spatial differences, temporally meaningful motion features, variable-length sequence modeling, and a decision policy that delays or rejects a prediction until sufficient evidence is available. The investigation also showed that the model was not always receiving equivalent representations of the same action. Sequence length, gesture completion, camera mirroring, hand-slot assignment, missing-hand handling, coordinate normalization, class metadata, and frame timing all affected the model input. The study therefore evaluates the complete recognition pipeline rather than only the neural-network architecture.

### 1.1 Study objective

The objective is to design, train, and evaluate a real-time hand gesture recognition model that maximizes final classification accuracy while also producing useful predictions before a gesture is complete. The study will quantify the tradeoff between prediction accuracy and prediction latency.

### 1.2 Independent variables

The planned experiments manipulate:

1. Gesture completion level or number of observed frames.
2. Spatial representation, including raw coordinates, wrist-relative coordinates, scale normalization, and canonical mirroring.
3. Temporal representation, including positions alone, frame differences, and time-normalized velocity.
4. Temporal model and pooling strategy.
5. Prediction policy, including fixed-prefix prediction, end-of-gesture finalization, confidence rejection, and temporal stability requirements.

### 1.3 Dependent variables

The primary outcome is classification accuracy. Secondary outcomes include macro F1 score, per-class recall, calibration error, time to stable correct prediction, finalization latency, and the proportion of inputs rejected as uncertain.

## 2. System overview

### 2.1 Input representation

The system extracts hand landmarks from video frames. Features include landmark position and temporal motion information. A consistent feature vector must be produced for every valid frame, with left- and right-hand data placed in stable slots.

### 2.2 Model

The active implementation uses PyTorch and contains bidirectional long short-term memory layers, masked normalization, multi-head attention, and masked temporal pooling. The classifier output dimension is equal to the number of gesture classes used when the model is created. The checkpoint, output dimension, and label-to-index map must therefore describe the same training run.

### 2.3 Inference paths

Three paths were compared:

1. Test-set inference on previously prepared landmark sequences.
2. Uploaded-video inference, in which a complete clip is available before classification.
3. Live-webcam inference, in which the system repeatedly classifies an incomplete prefix while the gesture is still being performed.

This distinction is important because an uploaded clip supplies the complete temporal context, while live inference initially supplies only the first few frames.

## 3. Problems encountered and corrective actions

### 3.1 Fixed 93-frame training sequences

**Observed problem.** Early training data used a fixed length of 93 frames. Shorter gestures were padded with zero-valued frames at the end. This encouraged assumptions throughout the pipeline that every gesture should contain exactly 93 frames.

**Why it matters.** Zero padding is not part of the sign. If padding is not masked correctly, the network can learn sequence-duration or padding patterns rather than only meaningful motion. Requiring 93 live frames also delays a prediction after the gesture has ended and makes short gestures contain a large proportion of artificial input.

**Correction.** The active PyTorch pipeline was redesigned to accept variable-length sequences. Each batch is padded only to the longest sequence in that batch, and a length mask prevents padded positions from contributing to normalization, attention, pooling, or loss-related decisions. A fixed length is no longer a property of the gesture representation.

**Verification required.** Retrain the model from raw, unpadded sequences. Confirm that padding values do not alter logits when the valid sequence and its recorded length are unchanged.

### 3.2 Invalid indexes on missing-hand rows

**Observed problem.** Debug output initially showed a non-monotonic or negative index, including repeated values such as `-153`, on rows where no hand data were present. Valid data rows later resumed at a positive index such as `25.0`.

**Root cause.** A value derived from array position or padding state was being written as though it were the original video-frame index. Empty or placeholder rows did not represent real observed frames, so their derived index was meaningless.

**Correction.** The pipeline should preserve the original capture-frame number separately from model-sequence position. Missing-hand frames may be used by the gesture-segmentation state machine, but synthetic empty rows should not be appended to the model sequence. If missing frames are retained for analysis, their frame number should be the real monotonically increasing capture index and their validity should be stored in a separate Boolean column.

**Verification required.** Generate a new debug file and assert that capture indexes strictly increase for all captured frames. Separately assert that model-sequence indexes are contiguous only across frames actually supplied to the model.

### 3.3 Incomplete live prefixes

**Observed problem.** Uploaded-video inference had access to the complete gesture and eventually produced the correct class. Live inference evaluated the first few frames repeatedly, often before the discriminating movement had occurred.

**Preliminary evidence.** A development test measured the following accuracy when only the beginning of each sequence was supplied to the same classifier:

| Input available to model | Preliminary accuracy |
|---|---:|
| First 8 frames | 47.74% |
| First 13 frames | 73.15% |
| First 20 frames | 86.03% |
| First 31 frames | 88.61% |
| Complete sequence | 89.03% |

These values are diagnostic results from the development test split and must be regenerated before publication. They should not be interpreted as the final accuracy of the later 2,731-class model.

**Interpretation.** The early portion of a sign may resemble several classes. Confidence on an incomplete prefix is therefore not directly comparable to confidence on a completed uploaded video.

**Correction.** Use missing-hand frames and/or an explicit end-of-gesture rule to decide when to finalize a prediction. A short inactivity period can trigger final classification without adding empty frames to the model input. Intermediate predictions may still be displayed, but they should be labeled provisional. A maximum gesture duration should protect against a missed end event without forcing every gesture to the same frame count.

### 3.4 Mirrored live camera and handedness mismatch

**Observed problem.** The browser displayed a mirrored webcam preview. Mirroring the display with CSS does not necessarily mirror the pixel data processed by MediaPipe. As a result, the user saw one orientation while the landmark extractor processed another. Hand labels or horizontal geometry could therefore disagree with the orientation represented in training data.

**Paired diagnostic example.** A short webcam recording of the sign MORE was tested through both paths. Uploaded-video predictions changed from HELLO and SOPHOMORE during the first two frames to MORE from the third frame onward, reaching approximately 0.998 confidence. The live path continued to favor HELLO, with a final distribution led by HELLO at approximately 0.710 and MORE at approximately 0.119.

For that single paired capture:

| Live-landmark transformation | Result for MORE |
|---|---:|
| Original live representation | 0.1185 confidence |
| Swap left/right hand slots | 0.9946 confidence |
| Mirror horizontal landmark geometry | 0.9982 confidence |

The mean normalized positional difference from the uploaded representation was approximately 0.0851 for the original live sequence, 0.0244 after swapping hand slots, 0.0313 after mirroring geometry, and 0.0803 after both operations. This is a one-example engineering diagnostic, not population-level evidence.

**Root cause.** The live and uploaded paths did not consistently apply the same mirroring and handedness convention.

**Correction.** Choose one canonical representation and apply it everywhere. If inference pixels are horizontally flipped, do so before landmark detection and interpret MediaPipe handedness in that same image space. If pixels remain unflipped, transform landmarks and handedness labels explicitly to match the training convention. Do not rely on `zip(reversed(handednesses), landmarks)`: detection order is not a guaranteed left/right ordering, and reversing a one-hand list does nothing.

**Verification required.** Record the same clip once, then feed identical frames through uploaded and live preprocessing. After alignment, compare the resulting feature arrays numerically. They should match within floating-point tolerance.

### 3.5 Unstable hand ordering

**Observed problem.** Detection order can change from frame to frame. Treating the first detected hand as one fixed side and the second as the other can swap feature blocks during a gesture.

**Correction.** Allocate explicit left- and right-hand slots based on the canonical handedness label. For ambiguous or low-confidence labels, use temporal continuity rather than list order. Log label, confidence, wrist position, and assigned slot during validation.

### 3.6 Coordinate and velocity inconsistency

**Observed problem.** Different paths combined absolute coordinates, wrist-relative coordinates, and frame-to-frame differences inconsistently. Live frame rate also varied, while some velocity features implicitly assumed a constant interval.

**Correction.** Move feature construction into one shared preprocessing function used by training, uploaded videos, and live inference. Apply the same coordinate origin, scale normalization, wrist treatment, feature order, missing-hand values, and data type in all paths. Compute motion using elapsed time where timestamp-normalized velocity is intended.

**Verification required.** Add unit tests using a saved landmark sequence. Training and both inference paths must produce the same tensor and sequence length from the same input.

<!-- REMOVE THIS ONE -->
<!-- ### 3.7 Color-space inconsistency 

**Observed problem.** OpenCV supplies frames in blue-green-red order, while MediaPipe processing expects red-green-blue input.

**Correction.** Convert every OpenCV frame from BGR to RGB immediately before MediaPipe processing. Perform the same conversion in uploaded and live paths. -->

### 3.8 Checkpoint and class-map mismatch

**Observed problem.** The project contained checkpoints with different output dimensions. One checkpoint contained 13 outputs, while the current parameter mapping contained 2,731 classes indexed from 0 through 2,730. A website configuration still pointed to the older 13-class checkpoint. When logits from that model were decoded with the new map, the application could only select indexes 0 through 12, making predictions appear restricted to the first few alphabetically indexed gestures.

**Correction.** Store the ordered class map and model configuration inside every checkpoint. At application startup, reject a model when any of the following differ: classifier output dimension, number of labels, ordered label map, feature dimension, or architecture version. Do not silently combine a newly generated `parameters.json` file with an older checkpoint.

**Checkpoint rule.** Resuming training is valid only when the class-to-index mapping is identical to the mapping used to create the checkpoint. Regenerating the parameters file is safe only if it deterministically reproduces the same ordered mapping. Adding, removing, or reordering classes requires a new classifier head and ordinarily a new training run.

### 3.10 Slow and progressively worsening training

**Observed problem.** Training took approximately three minutes per epoch and slowed over time. During one run, CPU use was about 35% and GPU use only 5–10%.

**Root causes identified.** Landmark samples were stored in HDF5 files on a hard disk drive and loaded lazily during training. Only labels and indexes were held in memory; the full dataset was not preloaded. Randomly shuffled small reads from a hard disk therefore limited input throughput. On Windows and Jupyter, using ten persistent data-loader workers also produced many Python processes and high committed memory, leading to paging and worsening performance.

**Correction.** Restart the kernel after runaway worker processes, begin with zero data-loader workers on Windows, disable persistent workers during diagnosis, and use a stable batch size. Moving HDF5 data to a solid-state drive or deliberately preloading a size-checked subset into memory can further reduce the bottleneck. Later optimizations may include length-bucketed batches, mixed-precision training, GPU-side feature calculations, and less frequent full validation.

**Verification required.** Report samples per second, GPU utilization, data-loading time, epoch time, memory use, storage type, worker count, and batch size. Change one setting at a time so that speed improvements can be attributed correctly.

## 4. Current evidence and its interpretation

### 4.1 What the existing observations support

The prefix experiment supports the conclusion that gesture completion materially affects classification accuracy. The paired MORE example supports the conclusion that live and uploaded preprocessing were not equivalent for at least one recording. The 13-output checkpoint explains why one application configuration could select only the first 13 mapped classes. System monitoring supports an input-pipeline bottleneck during the affected Windows training run.

### 4.2 What the existing observations do not yet support

The current evidence does not establish the final accuracy of the full 2,731-class model. One paired MORE example does not establish that mirroring or slot assignment explains every live error. GPU utilization from one run does not quantify the independent effect of hard-disk storage. Test-set accuracy alone does not demonstrate performance across new signers, backgrounds, camera positions, lighting conditions, or continuous signing.

## 5. Experimental design

### 5.1 Freeze the experimental system

Record the Git commit, model checkpoint hash, ordered class map, feature schema, dependency versions, device, and inference settings. The web application and notebook must load the same preprocessing implementation and metadata.

### 5.2 Build a paired evaluation set

Collect labeled webcam clips in which each captured frame can be replayed through both uploaded and simulated-live inference. Include multiple signs, signers, sessions, gesture durations, one- and two-handed signs, and both easy and confusable classes. Determine whether institutional review, participant consent, or parental consent is required before collecting data from people.

### 5.3 Experiment 1: accuracy under incomplete temporal input

This experiment directly answers the primary research question. Evaluate each test gesture using prefixes representing fixed percentages of its valid duration, such as 10%, 20%, 30%, and continuing through 100%. Percentage-based prefixes are preferable to only fixed frame counts because gesture durations vary. Compare at least the following decision strategies:

1. Predict at every prefix.
2. Predict only after a minimum amount of temporal evidence.
3. Finalize after a short no-hand or low-motion interval.
4. Require the same class to remain above a calibrated threshold for multiple consecutive observations.
5. Reject the input when uncertainty remains high.

Report accuracy and macro F1 as functions of gesture completion, as well as time to stable correct prediction, final prediction accuracy, false early-finalization rate, and finalization latency. This reveals whether an approach improves accuracy by genuinely recognizing partial gestures or merely waiting longer.

### 5.4 Experiment 2: spatial normalization and temporal motion features

This experiment answers the first supporting question. Train or evaluate controlled variants using the same data split and optimization settings:

1. Raw landmark coordinates.
2. Wrist-relative coordinates.
3. Wrist-relative coordinates with hand-scale normalization.
4. Canonically mirrored and handedness-aligned coordinates.
5. Position features without motion.
6. Position plus unnormalized frame differences.
7. Position plus timestamp-normalized velocity.

Where changing the input representation changes the feature dimension or meaning, retrain the model rather than applying a new representation to an incompatible checkpoint. Compare full-sequence performance and prefix performance. This determines whether a feature improves offline classification, early recognition, or both.

### 5.5 Experiment 3: preprocessing and temporal modeling robustness

This experiment answers the second supporting question. Evaluate the same frozen clips under controlled alternatives:

1. Current preprocessing.
2. Canonical mirroring only.
3. Explicit handedness-slot assignment only.
4. Shared coordinate and velocity preprocessing only.
5. End-of-gesture finalization only.
6. All corrections combined.

Then compare suitable temporal models under the same feature representation and split. Candidate baselines include a position-only multilayer perceptron with temporal pooling, a unidirectional LSTM, the bidirectional LSTM used for complete-sequence analysis, and an attention-based recurrent model. Because a bidirectional model uses all frames within the supplied prefix, it is valid for repeatedly classifying the prefix but cannot use future frames that have not arrived. A strictly causal model should also be evaluated if predictions are updated frame by frame without recomputing the full prefix.

This experiment will show which preprocessing corrections and model components actually improve results and whether any interaction exists among them.

### 5.6 Calibrate confidence

The existing confidence threshold is an engineering choice, not yet a calibrated probability boundary. Use a held-out validation set to examine reliability diagrams, expected calibration error, and the accuracy-coverage tradeoff at different rejection thresholds. A system should be allowed to return “uncertain” rather than always selecting one of 2,731 signs.

### 5.7 Apply suitable statistical analysis

Report confidence intervals for primary metrics. Because the same clips can be processed before and after each correction, use paired comparisons where appropriate. Keep signer or recording-session groups separate across training, validation, and test splits to reduce information leakage. Clearly distinguish exploratory diagnostics from prespecified final experiments.

### 5.8 Primary hypotheses

The experiments test the following hypotheses:

- **H1:** Accuracy will increase as a larger proportion of the gesture becomes available, with the largest gains occurring before the gesture's most discriminating motion is complete.
- **H2:** Wrist-relative, scale-normalized, and canonically oriented landmarks will outperform raw image-space coordinates on signer- and session-separated test data.
- **H3:** Time-normalized motion features will improve recognition of gestures distinguished primarily by movement, but may provide little benefit for static or posture-dominant gestures.
- **H4:** A temporal stability and uncertainty-based decision policy will improve finalized live accuracy compared with unconditional prediction at every early prefix, at the cost of additional latency or rejected predictions.
- **H5:** A shared preprocessing implementation will reduce the performance gap between uploaded-video and live-webcam inference.

## 6. Limitations

### 6.1 Isolated signs are not continuous sign language

The system classifies segmented isolated signs. It does not perform sentence-level recognition, language modeling, sign-boundary discovery in continuous discourse, or translation.

### 6.2 Hand landmarks omit important linguistic information

American Sign Language can use facial expression, mouth movement, body posture, and spatial grammar. A hand-landmark-only model cannot represent all of these signals.

### 6.3 Live domain shift remains

Camera placement, image mirroring, signer anatomy, signing style, background, occlusion, lighting, and frame rate can differ from the training data. Shared preprocessing removes avoidable software differences but does not eliminate real-world domain shift.

### 6.4 Large-vocabulary class imbalance

The expanded project uses 2,731 classes. Performance can be dominated by frequent or visually distinctive classes, so aggregate accuracy must be accompanied by class-aware metrics and support counts.

### 6.5 Preliminary measurements came from different development stages

Some reported diagnostics were collected before the full-vocabulary checkpoint and before all preprocessing changes. They are useful for explaining root causes but should not be combined into a single final performance claim.

### 6.6 No sufficiently large labeled live test set has yet been reported

The largest remaining evidentiary gap is a labeled, signer-separated webcam evaluation. Without it, the project cannot quantify whether live performance has become equivalent to uploaded-video performance.

## 7. Reproducibility record to complete

Before the manuscript is submitted, record the following:

- Git commit: `[COMMIT HASH]`
- Dataset name and version: `[DATASET VERSION]`
- Train/validation/test sample counts: `[COUNTS]`
- Number of classes: `[CLASS COUNT]`
- Class-map checksum: `[CHECKSUM]`
- Feature dimension and definition: `[FEATURE SCHEMA]`
- Checkpoint filename and SHA-256: `[CHECKPOINT HASH]`
- PyTorch and CUDA versions: `[VERSIONS]`
- GPU and CPU: `[HARDWARE]`
- Storage device used during training: `[HDD/SSD AND MODEL]`
- Batch size, worker count, and precision mode: `[SETTINGS]`
- Random seeds and determinism settings: `[SEEDS]`
- Model-selection rule and stopping criterion: `[RULE]`
- Repository/archive link: `[URL]`

## 8. Ethics, contributions, and disclosure placeholders

**Human participants.** `[State whether newly collected webcam recordings include human participants; describe consent/assent, parental permission where applicable, privacy protections, and IRB determination or approval. Do not claim exemption without a formal determination when one is required.]`

**Data availability.** `[Identify the public dataset and provide its persistent link. State whether newly recorded live clips can be shared and explain any privacy restriction.]`

**Code availability.** `[Provide a versioned repository or archived release containing training, preprocessing, evaluation, and application code.]`

**Author contributions.** `[Describe the author's roles in conceptualization, software, data curation, experiments, analysis, visualization, and writing.]`

**Mentor contributions and acknowledgments.** `[Name each mentor or guide and state their precise contribution, as requested by JHSS.]`

**Conflicts of interest.** `[State any conflicts or state that none are known.]`

**Use of generative artificial intelligence.** This working document was organized with assistance from OpenAI Codex using the project's development history supplied in the working session. The author must verify every technical statement, regenerate all reported values, rewrite or revise the prose as needed, and follow the journal's current disclosure policy.

## 9. Conclusion

The project investigates how a real-time hand gesture recognition system can maintain accuracy before the complete gesture is available. Preliminary results indicate that accuracy depends strongly on the amount of temporal context, while the paired webcam diagnostic demonstrates that spatial and preprocessing inconsistencies can overwhelm the model's learned temporal behavior. Variable-length masked training removes the need to wait for 93 frames or append artificial empty frames. Spatial normalization, time-aware motion features, consistent preprocessing, and an uncertainty-aware finalization policy are the principal strategies to be evaluated. The remaining work is to compare these strategies using a frozen model-development protocol and a controlled paired webcam dataset. That evaluation is necessary before concluding which design best balances early prediction, latency, and final accuracy.

## 10. Pre-submission checklist

- [ ] Frame the final submission as an original paper centered on incomplete temporal input.
- [ ] State the primary and supporting research questions consistently in the abstract, introduction, methods, and discussion.
- [ ] Replace all bracketed placeholders.
- [ ] Regenerate every number using one frozen full-vocabulary checkpoint.
- [ ] Verify that the web app rejects checkpoint/class-map mismatch.
- [ ] Test uploaded and live preprocessing on identical saved frames.
- [ ] Collect a sufficiently sized, labeled live-webcam evaluation set with appropriate ethics review and consent.
- [ ] Report per-class and streaming metrics, confidence intervals, and ablation results.
- [ ] Include figures for the pipeline, sequence masking, prefix accuracy, confusion matrix, and paired live/upload comparison.
- [ ] Distinguish isolated-sign classification from continuous ASL recognition throughout.
- [ ] Add numbered in-text citations and an APA-formatted reference list with live links.
- [ ] Manually number headings and references; do not rely on automatic numbering.
- [ ] Convert to a one-column Word or LibreOffice file using the current JHSS formatting rules.
- [ ] Verify all author, school, mentor, ethics, data, code, conflict, and AI-assistance disclosures.

## References to add and verify

The final manuscript should cite the primary sources for the dataset, MediaPipe hand tracking, long short-term memory networks, attention mechanism, PyTorch implementation, and any statistical methods used. Number references in order of first appearance, format them in APA style, and provide a live DOI or official source link. Do not treat this placeholder as a completed reference section.
