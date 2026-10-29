# Masked-face preprocessing, EDA, and MaskInv baseline

A documented Python 3.10+ program for the supplied `structure.txt` layout and the CS228 Checkpoint 1 requirements. It produces preprocessing artifacts, report figures, count-backed verification results, and an HTML evidence scaffold. It does **not** create the final 3-4 page IEEE/ACM submission or publish a repository.

The main program is `face_pipeline.py`. Its numbered sections follow data engineering (II), EDA (III), and baseline implementation (IV). Functions include docstrings explaining assumptions and statistical choices. The output report also provides a starting problem statement (I).

## 1. What the supplied listing reveals

These are **directory-listing counts**, not validated images or measured model results:

| Branch | Image entries | Identity folders |
|---|---:|---:|
| `train/masked` | 2,203 | 525 |
| `train/unmasked` | 90,468 | 460 |
| `verification/masked_whn` | 4,248 | 427 |

There are 543 unique training identity names across both branches. Under the provisional mapping `00_0001_tankai_07 -> tankai`, **422 of the 427 verification identity names overlap training**. Review `identity_mapping.csv` and supply corrected aliases if necessary. Empty folders are included in folder counts. Original filenames include `.jfif`, Unicode/legacy characters, and repeated basenames; paths, not basenames, identify samples.

The supplied files include no actual images, original dataset license/provenance, model weights, or contents of `verification/masked_pairs.txt`. Therefore the program does not infer a dataset name or fabricate actual EDA/results. `provided_structure_inventory.json` and `provided_structure_identity_mapping.csv` record evidence extracted from the listing.

## 2. Install

Create and activate a virtual environment, then install the core dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:       .venv\Scripts\activate.bat
# Linux/macOS:       source .venv/bin/activate
python -m pip install -r requirements.txt
```

Core preprocessing, EDA, and the pixel/PCA sanity baseline need no PyTorch or GPU. For the requested MaskInv model and optional MTCNN alignment, install a suitable PyTorch/torchvision build and the extra requirements in a compatible environment:

```bash
python -m pip install -r requirements-model.txt
```

`facenet-pytorch` constrains its compatible torch/torchvision/Pillow/NumPy versions; let pip resolve them together in a fresh environment. Its MTCNN detector may download detector weights on first use. The program itself does not automatically download the dataset or MaskInv checkpoints. CPU inference is supported but IResNet-100 is computationally expensive.

Record the actual resolved environment at your final submission commit:

```bash
python -m pip freeze > requirements-lock.txt
```

`environment.json` also records installed versions automatically. The constraints in the supplied requirements files are not an exact cross-platform lock.

## 3. Place the dataset

Keep data outside Git and pass its root directory:

```text
dataset/
  train/
    masked/<identity>/<image>
    unmasked/<identity>/<image>
  verification/
    masked_pairs.txt
    masked_whn/
      00_0001_tankai_07/
        unmasked.jpg
        0.jpg
        1.jfif
```

Training mask labels come from branch names. In `masked_whn`, `unmasked.*` is an unmasked reference and **all other supported image files are provisionally labeled masked**. Confirm this convention against the real dataset. Identity names are case-folded; verification prefixes/suffixes are stripped by the regex `\d+_\d+_(.+)_\d+`. The numeric fields are **not** interpreted as sessions, folds, image counts, age, or demographics.

Correct identities with an optional UTF-8 CSV:

```csv
origin,folder,identity
verification,00_0001_tankai_07,tankai
train,alternate_spelling,canonical_name
```

Copy and complete `dataset_metadata.example.json` with the actual origin, dataset citation, license, access/consent terms and session information. Model licensing does not establish dataset licensing. Set `identity_mapping_verified` only after reviewing the name mapping. No acquisition script is supplied because the data source is unknown; these placement instructions satisfy the code-side dataset placement requirement.

## 4. Preprocess and generate EDA

For an initial integrity/geometry audit requiring only core dependencies:

```bash
python face_pipeline.py --data-root "D:/datasets/faces" --output runs/prepared_pad --metadata dataset_metadata.json --alignment pad
```

For landmark detection and actual five-point alignment before the model baseline:

```bash
python face_pipeline.py --data-root "D:/datasets/faces" --output runs/prepared_aligned --metadata dataset_metadata.json --alignment mtcnn --device cpu
```

Add `--aliases identity_aliases.csv` after auditing identities. Each output directory must be new. The originals are never modified. PNG caches are created for all successfully decoded images and can use several GB for this listing. Excluded/quarantined images can remain in the cache; only manifest-selected samples are used for fitting and evaluation.

Alignment modes:

| Mode | Actual transformation | Appropriate interpretation |
|---|---|---|
| `pad` (default) | EXIF orientation, RGB conversion, aspect-preserving letterboxing to 112x112 | Initial audit or preprocessing ablation; **not face alignment** |
| `mtcnn` | EXIF/RGB, MTCNN detection, five-point similarity alignment to 112x112 | Recommended if images are not already aligned; failures audited |
| `prealigned` | EXIF/RGB; require input exactly 112x112 | Use only when the dataset's prior alignment is known |

By default, MTCNN rejects multiple faces and detections below confidence 0.9; `--multiple-faces largest` explicitly selects the largest detected face. Masked-face detection can fail more often, so compare rejection rates by condition. There is no automatic padding fallback after detection failure.

### Cleaning and split decisions

1. Validate expected branches; audit empty folders, nested directories and unsupported files.
2. Strictly decode images, apply EXIF orientation, convert to RGB, and measure original image statistics. Reject corruption, animated/multiframe images, unsafe decompression sizes, and images below `--min-side` (16 by default).
3. Compute brightness, contrast, entropy, geometry, clipping fractions and sharpness. Laplacian variance is measured on a common 112x112 gray scale; it is a heuristic, not a validated face-quality score. IQR flags are descriptive; blur/exposure outliers are retained by default.
4. Find duplicate connected components using file SHA-256, decoded RGB pixels, and processed RGB pixels. Quarantine all members with conflicting identity or mask labels. Otherwise retain one deterministic representative, preferring verification over training.
5. Exclude **every training identity that appears in verification**, including verification identities with rejected images. This can substantially shrink training; report the resulting counts.
6. Split usable verification identities into calibration (30% by default) and final test (70%) with seed 42. At least two identities are kept in each. All images of an identity stay together; identities and hashes are asserted disjoint across train/calibration/test. Genuine-pair coverage may still be insufficient for a scenario and is reported explicitly.
7. Fit optional PCA only on retained training identities; select verification thresholds only on calibration. The frozen pretrained MaskInv model needs no local identity training or PCA.

No missing identity or image is imputed. No augmentation or oversampling changes final evaluation. Near duplicates, unknown aliases, and incorrect identity labels still require review. Pretrained-model training overlap with this dataset remains unknown even after local train/verification purging.

### Listing-only mode

This does not require the actual dataset:

```bash
python face_pipeline.py --structure structure.txt --output runs/listing_only
```

It produces `structure_inventory.json` and `structure_identity_mapping.csv` without claiming to measure image content.

## 5. Run the requested MaskInv baseline

The official repository is [fdbtrs/Masked-Face-Recognition-KD](https://github.com/fdbtrs/Masked-Face-Recognition-KD). The upstream `evaluate.py` instantiates `iresnet100(num_features=512)` and loads backbone weights; this program adapts that frozen-network evaluation pattern to the supplied folder dataset.

Clone the reference source separately:

```bash
git clone https://github.com/fdbtrs/Masked-Face-Recognition-KD.git external/Masked-Face-Recognition-KD
```

Follow **Pretrained Models -> Our models** in its README and obtain a MaskInv backbone checkpoint: [official model folder](https://drive.google.com/drive/folders/1HMpNlECooNJ3d-on8KGTBEuxUjWnrT1I). Record exactly which checkpoint variant you chose. An ElasticFace teacher or another face model is not automatically a MaskInv student. Place the downloaded state dictionary in `weights/` and pass its actual filename; the example filename below is a placeholder.

Reuse a prepared cache to avoid reprocessing all training images:

```bash
python face_pipeline.py --evaluate-run runs/prepared_aligned --output runs/maskinv --baseline maskinv --repo external/Masked-Face-Recognition-KD --checkpoint "weights/YOUR_MASKINV_BACKBONE.pth" --device cpu
```

Or run preprocessing, EDA and the model in one command:

```bash
python face_pipeline.py --data-root "D:/datasets/faces" --output runs/full_maskinv --alignment mtcnn --baseline maskinv --repo external/Masked-Face-Recognition-KD --checkpoint "weights/YOUR_MASKINV_BACKBONE.pth" --metadata dataset_metadata.json
```

The model uses the upstream `backbones/iresnet.py` factory, strict checkpoint loading, evaluation mode, disabled gradients, deterministic torch operations, and L2-normalized 512-dimensional embeddings. A `module.` prefix and common state-dictionary wrappers are supported; missing/unexpected weights cause an error. No random model or pixel model is substituted when weights are missing. Only calibration/test images require CNN inference.

**Input contract:** default channel order is RGB and normalization is `(pixel / 255 - 0.5) / 0.5` (`--normalization half`). `--normalization arcface128` uses `(pixel - 127.5)/128`, `--channel-order BGR` changes channel order, and `--flip-test` enables original+flipped embedding summation. Compare these explicitly with the evaluator for your downloaded checkpoint before reporting an official-model baseline. The upstream evaluation entry point and README were available for review, but its detailed preprocessing module could not be retrieved here; these settings are explicit assumptions, not a claimed verified reproduction of that evaluator. Record the chosen contract in your report. Strict checkpoint compatibility alone cannot establish semantic compatibility of preprocessing or the selected model variant.

`run_config.json` and `baseline_results.json` record options, upstream commit/dirty state, source hash, and checkpoint hash. `--alignment pad` generates a warning when combined with MaskInv because padded images do not establish a faithful aligned reference. `--evaluate-run` keeps the original alignment and seed from preparation; pass `--seed` if a nondefault seed was used.

This is a **frozen pretrained MaskInv transfer baseline**, not local knowledge-distillation training or a reproduction of the paper's benchmark results. The checkpoint permits a pretrained reference baseline. Implementing full teacher/student KD training on a different dataset would require a separate experimental protocol; the official repository's training configuration targets MS1MV2, not this folder layout.

### Pair protocol and metrics

- NM: unmasked reference versus masked probe (primary experiment).
- MM: masked versus masked (requires >=2 distinct masked samples per genuine identity).
- NN: unmasked versus unmasked (usually unavailable here because there is one reference per identity). Unavailable scenarios are explicitly recorded, never replaced by self-pairs.
- Genuine means same identity; impostor means different identity. Mask state is not the recognition target.
- Genuine sampling is capped at 20 pairs per identity and 10,000 per class per scenario/split. Impostor sampling is balanced against the genuine sample and draws identities uniformly. Duplicate/self pairs are prohibited. These caps bound memory and runtime rather than enumerating all image pairs.
- Each scenario has an independently calibrated threshold. Compare conditions with that fact stated; these are not a single universal operating threshold.
- The operating threshold maximizes empirical calibration TAR subject to FAR <= 1% by default (`--target-far`). A reject-all threshold is possible. The threshold is then frozen for test metrics.
- FAR = FP / impostor pairs; FRR = FN / genuine pairs; TAR = TP / genuine pairs. JSON includes numerator/denominator for accuracy, precision, F1, specificity and components of balanced accuracy. ROC AUC and EER include genuine/impostor counts. Per-identity FRR includes each identity's counts and a macro mean.
- `*_thresholds.csv` includes every distinct empirical ROC operating point and integer error counts. EER linearly interpolates the FAR/FRR crossing; closest realizable points are recorded separately. Test EER/ROC are diagnostics and do not set the deployed threshold.
- With N impostors, empirical FAR resolution is 1/N. Zero observed errors do not establish population risk. Pairs share images and identities; pair counts are not counts of independent users. No independence-based confidence interval is claimed. Balanced-pair accuracy/precision does not reflect operational class prevalence.

`verification/masked_pairs.txt` exists in the listing, but its contents/format were not supplied. Its presence/hash is recorded and **it is not used**. This implementation instead defines a documented custom identity-disjoint pair protocol. To reproduce a benchmark tied to that pair file, first inspect its specification and implement a format-specific importer and prescribed fold protocol. Never compare these custom-pair numbers directly against published benchmark numbers.

## 6. Optional pixel/PCA sanity baseline

This checks the machinery without pretrained model dependencies and is **not the requested KD baseline**:

```bash
python face_pipeline.py --evaluate-run runs/prepared_pad --output runs/pixel_pca --baseline pixel-pca --pca-components 64
```

Gray 32x32 pixels are projected using PCA fitted on at most 5,000 retained training images (seeded sample), then unit-normalized for cosine verification. PCA automatically caps components by available training samples/features and exports its mean/components/variance and selected training samples. It needs at least three retained training images. No PCA whitening or test-fitted reduction is used. The cumulative explained-variance figure documents the reduction.

## References and licensing

1. Huber, M., Boutros, F., Kirchbuchner, F., and Damer, N. *Mask-invariant Face Recognition through Template-level Knowledge Distillation*. FG 2021. DOI: [10.1109/FG52635.2021.9667081](https://doi.org/10.1109/FG52635.2021.9667081). [Paper](https://arxiv.org/abs/2112.05646).
2. [Official code and model links](https://github.com/fdbtrs/Masked-Face-Recognition-KD), particularly `README.md` and `evaluate.py`, inspected October 29, 2026. Its README states CC BY-NC 4.0; verify terms for your intended use. The upstream code/weights are not redistributed in this package.
