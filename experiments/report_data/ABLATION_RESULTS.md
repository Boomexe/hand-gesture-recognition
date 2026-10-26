# Completed validation ablations

52 classes; 770 training clips; 189 validation clips; seeds 17, 42, 73. Means +/- sample SD across training seeds. No new test-set evaluation.

| Configuration | Parameters | Full accuracy (%) | Full macro F1 | Half-prefix accuracy (%) |
|---|---:|---:|---:|---:|
| Legacy features | 932,212 | 71.43 +/- 2.31 | 0.7024 +/- 0.0257 | 56.79 +/- 4.28 |
| Wrist-centered positions | 542,068 | 81.83 +/- 4.01 | 0.8094 +/- 0.0372 | 75.66 +/- 1.59 |
| Scale-normalized positions | 542,068 | 79.19 +/- 3.52 | 0.7811 +/- 0.0373 | 73.02 +/- 4.61 |
| Positions + frame differences | 671,092 | 83.25 +/- 2.39 | 0.8221 +/- 0.0277 | 76.19 +/- 1.91 |
| Positions + stored motion | 671,092 | 76.19 +/- 3.17 | 0.7529 +/- 0.0245 | 68.78 +/- 2.31 |
| Positions + prefix training | 542,068 | 82.54 +/- 1.40 | 0.8083 +/- 0.0271 | 79.89 +/- 1.91 |

## Matched prefix-training comparison

| Detected frames observed | Complete-only accuracy (%) | Prefix-trained accuracy (%) | Paired mean gain (percentage points) |
|---|---:|---:|---:|
| 10% | 20.46 +/- 0.81 | 41.09 +/- 2.39 | +20.63 |
| 20% | 42.68 +/- 1.70 | 62.96 +/- 2.80 | +20.28 |
| 30% | 58.02 +/- 2.39 | 73.54 +/- 3.22 | +15.52 |
| 40% | 70.90 +/- 0.92 | 77.43 +/- 0.81 | +6.53 |
| 50% | 75.66 +/- 1.59 | 79.89 +/- 1.91 | +4.23 |
| 60% | 79.37 +/- 2.80 | 82.01 +/- 2.12 | +2.65 |
| 70% | 80.78 +/- 4.11 | 83.25 +/- 1.70 | +2.47 |
| 80% | 82.19 +/- 3.36 | 83.77 +/- 1.86 | +1.59 |
| 90% | 82.89 +/- 3.23 | 82.36 +/- 1.86 | -0.53 |
| 100% | 81.83 +/- 4.01 | 82.54 +/- 1.40 | +0.71 |

## Individual full-sequence validation accuracies

| Configuration | Seed 17 (%) | Seed 42 (%) | Seed 73 (%) |
|---|---:|---:|---:|
| Legacy features | 74.07 | 69.84 | 70.37 |
| Wrist-centered positions | 83.60 | 84.66 | 77.25 |
| Scale-normalized positions | 81.48 | 80.95 | 75.13 |
| Positions + frame differences | 83.07 | 85.71 | 80.95 |
| Positions + stored motion | 76.19 | 79.37 | 73.02 |
| Positions + prefix training | 83.60 | 80.95 | 83.07 |

Sample SD is across three training seeds, not a confidence interval across signers. Models were selected on this validation set; results are exploratory and not held-out test or live-webcam estimates.
