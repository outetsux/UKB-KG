"""Bind evaluation outputs to the label matrices and sampled row indices."""
import hashlib
import json
import os

import numpy as np

RESULT_FILES = ("metrics.csv", "thresholds.csv", "probs_full.npy", "probs_full_labels.npy")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(array):
    array = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    for start in range(0, len(array), 4096):
        h.update(np.ascontiguousarray(array[start:start + 4096]).view(np.uint8))
    return h.hexdigest()


def indices_sha256(indices):
    h = hashlib.sha256()
    for label in sorted(indices):
        rows = np.asarray(indices[label], dtype="<i8")
        h.update(np.asarray([int(label), rows.size], dtype="<i8").tobytes())
        h.update(rows.tobytes())
    return h.hexdigest()


def save_run_manifest(output_dir, train_labels, val_labels, test_labels,
                      val_indices, test_indices, checkpoint, ratio, seed):
    manifest = {
        "version": 1,
        "pairing": {
            "train_labels": array_sha256(train_labels),
            "val_labels": array_sha256(val_labels),
            "test_labels": array_sha256(test_labels),
            "val_indices": indices_sha256(val_indices),
            "test_indices": indices_sha256(test_indices),
            "neg_to_pos_ratio": ratio,
        },
        "seed_requested": seed,  # Existing caches retain their original sampling.
        "checkpoint_sha256": file_sha256(checkpoint),
        "files": {name: file_sha256(os.path.join(output_dir, name)) for name in RESULT_FILES},
    }
    path = os.path.join(output_dir, "run_manifest.json")
    temporary = path + f".{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temporary, path)


def validate_run_pairing(model_dirs, val_label_path, test_label_path,
                         sample_idx_dir, neg_to_pos_ratio):
    expected = {"neg_to_pos_ratio": neg_to_pos_ratio}
    for split, path in (("val", val_label_path), ("test", test_label_path)):
        expected[f"{split}_labels"] = array_sha256(np.load(path, mmap_mode="r"))
        cache = os.path.join(sample_idx_dir, f"eval_sample_idx_{split}_r{neg_to_pos_ratio}.npy")
        expected[f"{split}_indices"] = indices_sha256(np.load(cache, allow_pickle=True).item())
    reference = None
    for directory in model_dirs:
        path = os.path.join(directory, "run_manifest.json")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing {path}; rerun evaluation to record actual sampling.")
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        pairing = manifest.get("pairing", {})
        if manifest.get("version") != 1 or not pairing.get("train_labels"):
            raise RuntimeError(f"Invalid evaluation manifest: {path}")
        if any(pairing.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Evaluation labels or sampled patients do not match the supplied cache: {directory}")
        if reference is not None and pairing != reference:
            raise RuntimeError(f"Evaluation pairing differs across model results: {directory}")
        for name in RESULT_FILES:
            if manifest.get("files", {}).get(name) != file_sha256(os.path.join(directory, name)):
                raise RuntimeError(f"Result changed after evaluation: {directory}/{name}; rerun evaluation.")
        reference = pairing
    print("[pairing] run manifests, actual sampling and result files verified")
