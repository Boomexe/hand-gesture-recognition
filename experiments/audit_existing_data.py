"""Inventory reusable experiment data and verify available HDF5/source links.

No source files are changed. Content links require exact float32 array equality
(including frame indexes, presence, positions, and stored motion channels).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def array_digest(values):
    values = np.asarray(values, dtype="<f4", order="C")
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def distribution(values):
    if not values:
        return None
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    parameters = json.loads(args.parameters.read_text())
    vocabulary = set(parameters["numbered_unique_words"])
    source = args.project_root/"asl citizen data/ASL_Citizen"
    extracted = args.project_root/"asl_citizen_gesture_data"
    manifests, all_signers = {}, {}
    for split in ("train", "val", "test"):
        with (source/"splits"/f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            manifests[split] = list(csv.DictReader(stream))
        all_signers[split] = set(row["Participant ID"] for row in manifests[split])
    report = {"project_root": str(args.project_root.resolve()),
              "vocabulary": sorted(vocabulary), "class_count": len(vocabulary),
              "source_video_file_count": sum(1 for _ in (source/"videos").glob("*.mp4")),
              "source_manifest_signer_overlap": {}, "splits": {}}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        report["source_manifest_signer_overlap"][f"{first}_{second}"] = sorted(all_signers[first] & all_signers[second])
    links = []
    selected_hashes = {}
    for split, source_rows in manifests.items():
        rows = [row for row in source_rows if row["Gloss"] in vocabulary]
        raw_lookup = defaultdict(list)
        available_csv, malformed_csv, missing_csv, available_video = 0, [], [], 0
        selected_signers = sorted(set(row["Participant ID"] for row in rows))
        for row in rows:
            video = source/"videos"/row["Video file"]
            available_video += int(video.is_file())
            raw_file = extracted/"raw_data"/split/(row["Video file"]+".csv")
            if not raw_file.is_file():
                missing_csv.append(row["Video file"])
                continue
            available_csv += 1
            raw = pd.read_csv(raw_file).to_numpy(dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != 255 or len(raw) == 0:
                malformed_csv.append(row["Video file"])
                continue
            raw_lookup[(row["Gloss"], array_digest(raw))].append(row)
        lengths, support, matches = [], Counter(), Counter()
        invalid, non_contiguous_index, nonzero_wrists, zero_motion_samples = [], [], [], []
        max_motion = 0.0
        hashes = set()
        h5_path = extracted/f"cleaned_{split}_data.h5"
        with h5py.File(h5_path, "r") as h5:
            total = int(h5.attrs["num_samples"])
            for index in range(total):
                group = h5[str(index)]
                word = group.attrs["word"]
                if isinstance(word, bytes):
                    word = word.decode("utf-8")
                if word not in vocabulary:
                    continue
                raw = group["data"][:].astype(np.float32, copy=False)
                support[word] += 1
                lengths.append(len(raw))
                valid_shape = raw.ndim == 2 and raw.shape[1] == 255 and len(raw) > 0
                if not valid_shape or not np.isfinite(raw).all():
                    invalid.append(index)
                    continue
                if not np.isin(raw[:, 1:3], (0, 1)).all() or not raw[:, 1:3].any(axis=1).all():
                    invalid.append(index)
                if not np.array_equal(raw[:, 0], np.arange(len(raw))):
                    non_contiguous_index.append(index)
                if np.any(raw[:, [3, 4, 5, 129, 130, 131]] != 0):
                    nonzero_wrists.append(index)
                motion = raw[:, 3:].reshape(len(raw), 2, 21, 6)[..., 3:]
                max_motion = max(max_motion, float(np.abs(motion).max()))
                if not np.any(motion):
                    zero_motion_samples.append(index)
                digest = array_digest(raw)
                hashes.add(digest)
                candidates = raw_lookup.get((word, digest), [])
                embedded = str(group.attrs.get("clip_name", ""))
                status = "unique_content_match" if len(candidates) == 1 else "ambiguous_content_match" if candidates else "unmatched"
                matches[status] += 1
                link = {"split": split, "hdf5_sample_id": str(index), "word": word,
                        "digest": digest, "detected_frames": len(raw), "status": status,
                        "embedded_clip_name": embedded,
                        "source_candidates": [{"video": c["Video file"], "signer": c["Participant ID"]} for c in candidates]}
                links.append(link)
        selected_hashes[split] = hashes
        report["splits"][split] = {
            "manifest_rows_all_classes": len(source_rows),
            "manifest_signers_all_classes": len(all_signers[split]),
            "selected_manifest_rows": len(rows), "selected_manifest_signers": selected_signers,
            "selected_videos_present": available_video, "selected_raw_csvs_present": available_csv,
            "missing_selected_raw_csvs": missing_csv, "malformed_selected_raw_csvs": malformed_csv,
            "hdf5_samples_all_classes": total, "hdf5_selected_samples": sum(support.values()),
            "selected_class_support": dict(sorted(support.items())), "detected_frame_lengths": distribution(lengths),
            "content_link_counts": dict(matches), "invalid_selected_sample_ids": invalid,
            "non_contiguous_capture_index_sample_ids": non_contiguous_index,
            "nonzero_wrist_position_sample_ids": nonzero_wrists,
            "all_zero_stored_motion_sample_ids": zero_motion_samples,
            "max_absolute_stored_motion": max_motion,
            "hdf5_bytes": h5_path.stat().st_size,
            "hdf5_mtime_ns": h5_path.stat().st_mtime_ns}
        print(f"{split}: selected={sum(support.values())}; links={dict(matches)}", flush=True)
    report["exact_feature_hash_overlap_between_splits"] = {
        f"{a}_{b}": len(selected_hashes[a] & selected_hashes[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    report["interpretation"] = [
        "Source split manifests include participant IDs and their overlap is checked explicitly.",
        "A unique content match verifies a saved HDF5 sequence against an available source-labelled CSV.",
        "Unmatched rows cannot inherit a signer ID based solely on a label or guessed row order.",
        "Manifest signer separation does not by itself verify every unlinked HDF5 row's provenance.",
        "A contiguous detected-frame index cannot locate dropped detections in source-video time.",
        "Original videos permit new extraction without new participant recordings.",
        "Stored motion values are observable, but their unit convention is not proved by numeric presence.",
        "Exact feature-hash overlap checks do not detect all possible duplicate-video or leakage cases."]
    args.output.mkdir(parents=True)
    (args.output/"audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output/"source_links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    print(f"Audit saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
