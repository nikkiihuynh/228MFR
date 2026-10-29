"""
Preprocessing follows the MaskInv report and repository: 112x112 images and pixel values scaled to [-1, 1].
Run: python baseline.py --weights models/MaskInvHG_backbone.pth
"""

import argparse
import csv
import hashlib
import json
from os import mkdir
import random
import sys
from pathlib import Path
import numpy as np

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# Make the MaskInv backbone and metrics helpers importable.
sys.path.insert(0, str(REPO_ROOT / "scripts")) # make scripts dir and move baseline & preprocess.py into it.
sys.path.insert(0, str(REPO_ROOT / "external" / "MaskInv"))
import metrics

# The network's last layer expects 7x7 feature maps, thus input should be 112x112.
IMAGE_SIZE = 112
EMBEDDING_SIZE = 512

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "rmfd"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "MaskInvHG_backbone.pth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "baseline"
DEFAULT_SEED = 228
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64

# FAR target threshold on val split
TARGET_FARS = [0.01, 0.001]

# --------------------------------------------------
# 2. Read Verification Pairs
# --------------------------------------------------

# Each line in masked_pairs.txt has the format: 
# relative_image_path_1 relative_image_path_2 label
# label: 1 = same identity, 0 = different identities

def resolve_image(relative_path, image_dir, verification_root):
    """
    Find an image under the image folder or verification root.
    """
    for root in (image_dir, verification_root):
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    return None

def read_pairs(pairs_file, image_dir, verification_root):
    if not pairs_file.is_file():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    pairs = []
    missing = 0

    with pairs_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            # Ignore blank lines.
            if not parts:
                continue
            if len(parts) != 3 or parts[2] not in ("0", "1"):
                raise ValueError(
                    f"Expected '<image a> <image b> <0 or 1>' on line "
                    f"{line_number} of {pairs_file}: {line.strip()}"
                )
            relative_a, relative_b, label= parts
            path_a = resolve_image(relative_a, image_dir, verification_root)
            path_b = resolve_image(relative_b, image_dir, verification_root)
            
            # Skip pairs where either image could not be resolved.
            if path_a is None or path_b is None:
                missing += 1
                continue
            pairs.append({
                "relative_a": relative_a,
                "relative_b": relative_b,
                "path_a": path_a,
                "path_b": path_b,
                "label": int(label)
            })
    if missing > 0:
        print(f"Warning: {missing} image pairs could not be resolved.")
    return pairs

def identity_of(relative_path):
    """
    Identity is the name of the image's parent directory.
    """
    parent = Path(relative_path).parent.name
    return parent or None

# --------------------------------------------------
# 3. Load MaskInv Model
# --------------------------------------------------
def load_model(weights_path, device):
    import torch

    # From MaskInv submodule
    from backbones import iresnet100
    model = iresnet100(num_features=EMBEDDING_SIZE)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove the prefix added when a model is saved from DataParallel.
    state = {
        key.replace("module.", "", 1): value
        for key, value in state.items()
    }

    model.load_state_dict(state)
    # Evaluation mode required for batch normalization.
    return model.to(device).eval()

# --------------------------------------------------
# 4. Preprocess Images and Compute Embeddings
# --------------------------------------------------
def load_image(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Scale pixel values to the range [-1, 1].
    array = (array - 0.5) / 0.5

    # Convert from height * width * channels to channels * height * width.
    return array.transpose(2, 0, 1)

def compute_embeddings(paths, weights_path, batch_size, device_name):
    import torch
    device = torch.device(device_name)
    model = load_model(weights_path, device)
    embeddings = {}
    unreadable = []

    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = []
            batch = []

            for path in paths[start:start + batch_size]:
                try:
                    batch.append(load_image(path))
                    batch_paths.append(path)
                except Exception as Error:
                    unreadable.append((str(path), str(Error)))
            if not batch:
                continue
            images = torch.from_numpy(np.stack(batch)).to(device)

            # Add embedding of the image and of the horizontal flip and normalize to unit length.
            output = model(images) + model(torch.flip(images, dims=[3]))
            output = torch.nn.functional.normalize(output, dim=1)
            output = output.cpu().numpy()
            for path, vector in zip(batch_paths, output):
                embeddings[path] = vector

            if (start // batch_size) % 20 == 0:
                print(f" Embedded {start + len(batch_paths)}/{len(paths)}")

    return embeddings, unreadable

# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------
def run_evaluation(pairs, embeddings, output_dir, seed, validation_split, target_fars):
    """
    Score the pairs, choose thresholds, and report on test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only keep pairs where both images have embeddings.
    kept = [
        pair for pair in pairs
        if pair["path_a"] in embeddings and pair["path_b"] in embeddings
    ]
    skipped = len(pairs) - len(kept)

    # Both embeddings have unit length, thus dot product is equivalent to cosine similarity.
    scores = np.array([
        float(np.dot(embeddings[pair["path_a"]], embeddings[pair["path_b"]]))
        for pair in kept
    ])
    labels = np.array([pair["label"] for pair in kept])

    # Split into validation and test sets.
    pair_ids = [
        (identity_of(pair["relative_a"]), identity_of(pair["relative_b"]))
        for pair in kept
    ]

    try:
        val_index, test_index, dropped = metrics.identity_disjoint_split(
            pair_ids, validation_split, seed
        )
        split_kind = "identity_disjoint"
    except ValueError:
        val_index, test_index = metrics.random_pair_split(
            len(kept), validation_split, seed
        )
        dropped = 0
        split_kind = "random pairs(identiy could not be read)"
        print("Warning: Identity disjoint split failed, using random pair split instead.")

    val_scores, val_labels = scores[val_index], labels[val_index]
    test_scores, test_labels = scores[test_index], labels[test_index]

    results = {
        "pairs_total": len(pairs),
        "pairs_kept": len(kept),
        "pairs_skipped": skipped,
        "split": split_kind,
        "seed": seed,
        "dropped": dropped,
        "val": {
            "pairs": int(val_index.size),
            "genuine": int(val_labels.sum()),
            "impostor": int((1 - val_labels).sum()),
        },
        "test": {
            "pairs": int(test_index.size),
            "genuine": int(test_labels.sum()),
            "impostor": int((1 - test_labels).sum()),
        },
        "AUC_test": metrics.auc(test_scores, test_labels),
    }

    eer, eer_threshold = metrics.equal_error_rate(test_scores, test_labels)
    results["EER_test"] = eer
    results["EER_threshold"] = eer_threshold

    # Choose each threshold on the validation split only, then apply to test.
    operating_points = {}
    for target in target_fars:
        threshold = metrics.threshold_for_far(val_scores, val_labels, target)
        operating_points[f"FAR<={target:g}"] = {
            "val": metrics.summarize_at(val_scores, val_labels, threshold),
            "test": metrics.summarize_at(test_scores, test_labels, threshold),
        }
    results["operating_points"] = operating_points

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    primary_name = f"FAR<={target_fars[0]:g}"
    primary_threshold = operating_points[primary_name]["val"]["tau"]

    save_scores(output_dir, kept, scores, val_index, test_index, primary_threshold)
    save_error_examples(output_dir, kept, test_index, test_scores, test_labels)
    plot_results(output_dir, test_scores, test_labels, eer, eer_threshold, primary_threshold, primary_name)

    return results

# --------------------------------------------------
# 6. Save Scores and Error Examples
# --------------------------------------------------

def save_scores(output_dir, kept, scores, val_index, test_index, threshold):
    split_names = {int(i): "val" for i in val_index}
    split_names.update({int(i): "test" for i in test_index})

    with open(output_dir / "scores.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image_a", "image_b", "label", "score", "split", "accepted"])
        for i, pair in enumerate(kept):
            writer.writerow([
                pair["relative_a"], 
                pair["relative_b"],
                pair["label"],
                f"{scores[i]:.6f}",
                split_names.get(i, "dropped"),
                int(scores[i] >= threshold),
            ])

def save_error_examples(output_dir, kept, test_index, test_scores, test_labels):
    test_pairs = [kept[i] for i in test_index]
    order= np.argsort(test_scores)

    # Genuine Pairs with Lowest Scores = False Rejects
    false_rejects = [
        (test_pairs[i], test_scores[i]) for i in order if test_labels[i] == 1
    ][:20]

    # Impostor Pairs with Highest Scores = False Accepts
    false_accepts = [
        (test_pairs[i], test_scores[i])
        for i in order[::-1] if test_labels[i] == 0
    ][:20]
    for name, rows in (("false_rejects.csv", false_rejects),
                       ("false_accepts.csv", false_accepts)):
        with open(output_dir / name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_a", "image_b", "score"])
            for pair, score in rows:
                writer.writerow(
                    [pair["relative_a"], pair["relative_b"], f"{score:.6f}"])

# --------------------------------------------------
# 7. Plot Results
# --------------------------------------------------
def plot_results(output_dir, test_scores, test_labels, eer, eer_threshold, primary_threshold, primary_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # FAR and FRR vs decision threshold
    thresholds, far, frr = metrics.far_frr_curve(test_scores, test_labels)
    figure, axis = plt.subplots(figsize=(5, 3.4))
    axis.plot(thresholds, far, label="FAR")
    axis.plot(thresholds, frr, label="FRR")
    axis.axvline(eer_threshold, color="gray", linestyle=":", label=f"EER={eer:.3f}")
    axis.axvline(primary_threshold, color="red", linestyle="--", label=f"Operating Point ({primary_name})")
    axis.set_xlabel("Decision Threshold (cosine similarity)")
    axis.set_ylabel("Error Rate")
    axis.set_title("FAR and FRR vs Decision Threshold")
    axis.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve.png", dpi=200)
    plt.close(figure)

    # Score distributions of genuine and impostor pairs.
    figure, axis = plt.subplots(figsize=(5, 3.4))
    bins = np.linspace(min(test_scores.min(), -0.2), max(test_scores.max(), 1.0), 50)
    axis.hist(test_scores[test_labels == 1], bins=bins, alpha=0.6, label="Genuine")
    axis.hist(test_scores[test_labels == 0], bins=bins, alpha=0.6, label="Impostor")
    axis.axvline(primary_threshold, color="red", linestyle="--")
    axis.set_xlabel("Cosine Similarity")
    axis.set_ylabel("Number of Pairs")
    axis.set_title("Score Distributions of Genuine and Impostor Pairs (test split)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "score_hist.png", dpi=200)
    plt.close(figure)

# --------------------------------------------------
# 8. Main
# --------------------------------------------------
def sha_256_of(path):
    """
    Helps record which weights were used.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-split", type=int, default=VALIDATION_SPLIT)
    parser.add_argument("--batch-size", type=float, default=BATCH_SIZE)
    parser.add_argument("--target-far", type=float, nargs="+", default=TARGET_FARS, help="FAR targets")
    args = parser.parse_args()
    set_seed(args.seed)

    import torch
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_name}, PyTorch version: {torch.__version__}")

    verification_root = args.data_root / "verification"
    pairs = read_pairs(
        verification_root / "masked_pairs.txt",
        verification_root / "masked_whn",
        verification_root,
    )
    print(f"Verification pairs: {len(pairs)}")

    # Embed each image once
    paths = sorted({pair[key] for pair in pairs for key in ("path_a", "path_b")})
    embeddings, unreadable = compute_embeddings(paths, args.weights, args.batch_size,device_name)

    if unreadable:
        print(f"Warning: {len(unreadable)} unreadable images,"
              f" for example {unreadable[:3]}")

    results = run_evaluation(pairs, embeddings, args.out, args.seed, args.validation_split, args.target_far,)

    # Record what was used for reproducibility.
    results["weights_file"] = args.weights.name
    results["weights_sha256"] = sha_256_of(args.weights)
    results["unreadable_images"] = len(unreadable)
    results["torch_version"] = torch.__version__

    with open(args.out / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results["operating_points"], indent=2))
    print(f"EER (test): {results['EER_test']:.4f}")
    print(f"AUC (test): {results['AUC_test']:.4f}")
    print(f"Outputs written to {args.out}")

if __name__ == "__main__":
    main()