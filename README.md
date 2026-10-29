# CS228 Checkpoint 1

## 1. Download and extract dataset

Download and extract [the full dataset](https://drive.google.com/file/d/1jgRz26UA5TJEPcAxA-B8WlX8cyx7K3cF/view?usp=sharing)

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

The dataset should have the following structure:

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

Training mask labels come from branch names. In `masked_whn`, `unmasked.*` is an unmasked reference and all other supported image files are provisionally labeled masked. masked_pairs.txt contains a given list of genuine and impostor verification pairs, but we generate our own.

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

## 5. Run the requested MaskInv baseline

The official repository is [fdbtrs/Masked-Face-Recognition-KD](https://github.com/fdbtrs/Masked-Face-Recognition-KD). The upstream `evaluate.py` instantiates `iresnet100(num_features=512)` and loads backbone weights; this program adapts that frozen-network evaluation pattern to the supplied folder dataset.

Clone the reference source separately:

```bash
git clone https://github.com/fdbtrs/Masked-Face-Recognition-KD.git external/Masked-Face-Recognition-KD
```

Follow **Pretrained Models -> Our models** in its README and obtain a MaskInv backbone checkpoint: [official model folder](https://drive.google.com/drive/folders/1HMpNlECooNJ3d-on8KGTBEuxUjWnrT1I). Note that we used the supplied pre-trained model **MaskInvHG_backbone.pth** available from the drive link. Place the downloaded state dictionary in `weights/` and pass its filename.

Reuse a prepared cache to avoid reprocessing all training images:

```bash
python face_pipeline.py --evaluate-run runs/prepared_aligned --output runs/maskinv --baseline maskinv --repo external/Masked-Face-Recognition-KD --checkpoint "weights/MaskInvHG_backbone.pth" --device cpu
```

Or run preprocessing, EDA and the model in one command (not recommended, takes a lot more time):

```bash
python face_pipeline.py --data-root "D:/datasets/faces" --output runs/full_maskinv --alignment mtcnn --baseline maskinv --repo external/Masked-Face-Recognition-KD --checkpoint "weights/MaskInvHG_backbone.pth" --metadata dataset_metadata.json
```

The model uses the upstream `backbones/iresnet.py` factory, strict checkpoint loading, evaluation mode, disabled gradients, deterministic torch operations, and L2-normalized 512-dimensional embeddings. A `module.` prefix and common state-dictionary wrappers are supported; missing/unexpected weights cause an error. No random model or pixel model is substituted when weights are missing. Only calibration/test images require CNN inference.

Default channel order is RGB and normalization is `(pixel / 255 - 0.5) / 0.5` (`--normalization half`). `--normalization arcface128` uses `(pixel - 127.5)/128`, `--channel-order BGR` changes channel order, and `--flip-test` enables original and flipped embedding summation.

`run_config.json` and `baseline_results.json` record options, upstream commit/dirty state, source hash, and checkpoint hash. `--alignment pad` generates a warning when combined with MaskInv because padded images do not establish a faithful aligned reference. `--evaluate-run` keeps the original alignment and seed from preparation; pass `--seed` if a nondefault seed was used.

### Pair protocol and metrics

- NM: unmasked reference versus masked probe (primary experiment).
- MM: masked versus masked (requires >=2 distinct masked samples per genuine identity).
- NN: unmasked versus unmasked (usually unavailable here because there is one reference per identity). Unavailable scenarios are explicitly recorded, never replaced by self-pairs.
- Genuine means same identity; impostor means different identity. Mask state is not the recognition target.
- Genuine sampling is capped at 20 pairs per identity and 10,000 per class per scenario/split. Impostor sampling is balanced against the genuine sample and draws identities uniformly. Duplicate/self pairs are prohibited. These caps bound memory and runtime rather than enumerating all image pairs.
- Each scenario has an independently calibrated threshold.
- The operating threshold maximizes empirical calibration TAR subject to FAR <= 1% by default (`--target-far`). A reject-all threshold is possible. The threshold is then frozen for test metrics.
- FAR = FP / impostor pairs; FRR = FN / genuine pairs; TAR = TP / genuine pairs. JSON includes numerator/denominator for accuracy, precision, F1, specificity and components of balanced accuracy. ROC AUC and EER include genuine/impostor counts. Per-identity FRR includes each identity's counts and a macro mean.
- `*_thresholds.csv` includes every distinct empirical ROC operating point and integer error counts. EER linearly interpolates the FAR/FRR crossing; closest realizable points are recorded separately. Test EER/ROC are diagnostics and do not set the deployed threshold.
- With N impostors, empirical FAR resolution is 1/N. Zero observed errors do not establish population risk. Pairs share images and identities; pair counts are not counts of independent users. No independence-based confidence interval is claimed. Balanced-pair accuracy/precision does not reflect operational class prevalence.

`verification/masked_pairs.txt` exists in the listing, but this implementation instead defines a custom identity-disjoint pair protocol.

## References and licensing

1. Huber, M., Boutros, F., Kirchbuchner, F., and Damer, N. *Mask-invariant Face Recognition through Template-level Knowledge Distillation*. FG 2021. DOI: [10.1109/FG52635.2021.9667081](https://doi.org/10.1109/FG52635.2021.9667081). [Paper](https://arxiv.org/abs/2112.05646).
2. [Official code and model links](https://github.com/fdbtrs/Masked-Face-Recognition-KD), particularly `README.md` and `evaluate.py`, inspected October 29, 2026. Its README states CC BY-NC 4.0; verify terms for your intended use. The upstream code/weights are not redistributed in this package.
3. [Dataset download](https://drive.google.com/file/d/1jgRz26UA5TJEPcAxA-B8WlX8cyx7K3cF/view?usp=sharing) [Original source](https://github.com/deenovita/RWMFD)
