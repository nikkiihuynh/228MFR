from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ZIP_PATH = Path("Datasets\\RMFD.zip")
DATA_DIR = Path("Datasets\\RMFD")

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 228
NUM_WORKERS = 0  # Works reliably on Windows too.

# Adjust these paths if your extracted directory layout differs.
TRAIN_ROOT = DATA_DIR / "train"
VERIFICATION_ROOT = DATA_DIR / "verification"
VERIFICATION_DIR = VERIFICATION_ROOT / "masked_whn"
PAIRS_FILE = VERIFICATION_ROOT / "masked_pairs.txt"

torch.manual_seed(SEED)


# --------------------------------------------------
# 2. Extract the archive safely
# --------------------------------------------------

def extract_dataset(zip_path, destination):
    if destination.exists() and any(destination.iterdir()):
        print(f"Using existing extracted directory: {destination}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # Prevent archive paths from escaping the destination.
            output_path = (destination / member.filename).resolve()

            if not output_path.is_relative_to(destination_resolved):
                raise ValueError(
                    f"Unsafe path in archive: {member.filename}"
                )

        archive.extractall(destination)

    print(f"Extracted dataset to {destination}")


# --------------------------------------------------
# 3. Discover training identities and image paths
# --------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


extract_dataset(ZIP_PATH, DATA_DIR)

# Training images are organized by identity.
# This assumes the training layout is:
# train/
#   unmasked/<identity>/
#   masked/<identity>/

unmasked_train_root = TRAIN_ROOT / "unmasked"

if not unmasked_train_root.is_dir():
    raise FileNotFoundError(
        f"Could not find training identity folders: {unmasked_train_root}"
    )

identities = sorted(
    path.name
    for path in unmasked_train_root.iterdir()
    if path.is_dir()
)

if not identities:
    raise RuntimeError(
        f"No identity folders found in {unmasked_train_root}"
    )

class_to_idx = {
    identity: index
    for index, identity in enumerate(identities)
}
idx_to_class = {
    index: name
    for name, index in class_to_idx.items()
}

print("\nIdentity mapping:")
for name, index in class_to_idx.items():
    print(f"  {name}: {index}")


# --------------------------------------------------
# 4. Define image preprocessing
# --------------------------------------------------

# ImageNet normalization is appropriate when using an
# ImageNet-pretrained model later.
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# --------------------------------------------------
# 5. Build records for the training set
# --------------------------------------------------

# Each record:
# (image_path, identity_label, is_masked)
train_records = []

for identity in identities:
    label = class_to_idx[identity]

    for split_name, is_masked in [
        ("unmasked", 0),
        ("masked", 1),
    ]:
        folder = TRAIN_ROOT / split_name / identity

        if not folder.exists():
            print(f"Warning: missing folder {folder}")
            continue

        for image_path in list_images(folder):
            train_records.append(
                (image_path, label, is_masked)
            )


# --------------------------------------------------
# 6. Read verification pairs from masked_pairs.txt
# --------------------------------------------------

# Each line in masked_pairs.txt has the format:
# relative_image_path_1 relative_image_path_2 label
#
# Label:
#   1 = genuine pair (same identity)
#   0 = impostor pair (different identities)
#
# The paths are resolved relative to VERIFICATION_ROOT.

def read_verification_pairs(pairs_file, image_root):
    if not pairs_file.is_file():
        raise FileNotFoundError(
            f"Could not find verification pairs file: {pairs_file}"
        )

    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Could not find verification image directory: {image_root}"
        )

    image_root = image_root.resolve()
    pairs = []

    with pairs_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()

            # Ignore blank lines.
            if not parts:
                continue

            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 fields on line {line_number} of "
                    f"{pairs_file}, got {len(parts)}: {line.strip()}"
                )

            relative_path_a, relative_path_b, label_text = parts

            try:
                label = int(label_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid label on line {line_number}: {label_text}"
                ) from error

            if label not in (0, 1):
                raise ValueError(
                    f"Expected label 0 or 1 on line {line_number}, "
                    f"got {label}"
                )

            path_a = (image_root / relative_path_a).resolve()
            path_b = (image_root / relative_path_b).resolve()

            # Ensure paths stay inside the verification image root.
            if (
                not path_a.is_relative_to(image_root)
                or not path_b.is_relative_to(image_root)
            ):
                raise ValueError(
                    f"Pair on line {line_number} points outside "
                    f"the verification directory."
                )

            pairs.append((path_a, path_b, label))

    return pairs


verification_pairs = read_verification_pairs(
    pairs_file=PAIRS_FILE,
    image_root=VERIFICATION_DIR,
)


# --------------------------------------------------
# 7. Validate images
# --------------------------------------------------

def keep_readable_records(records):
    """Validate image-classification records."""
    valid = []
    invalid = []

    for path, label, is_masked in records:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append((path, label, is_masked))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            invalid.append((path, str(error)))

    if invalid:
        print(f"\nUnreadable images: {len(invalid)}")
        for path, error in invalid[:10]:
            print(f"  {path}: {error}")

    return valid


def keep_readable_pairs(pairs):
    """Keep only verification pairs whose two images are readable."""
    valid = []
    invalid = []

    for path_a, path_b, label in pairs:
        try:
            with Image.open(path_a) as image:
                image.verify()

            with Image.open(path_b) as image:
                image.verify()

            valid.append((path_a, path_b, label))

        except (OSError, UnidentifiedImageError, ValueError) as error:
            invalid.append((path_a, path_b, str(error)))

    if invalid:
        print(f"\nInvalid verification pairs: {len(invalid)}")
        for path_a, path_b, error in invalid[:10]:
            print(f"  A: {path_a}")
            print(f"  B: {path_b}")
            print(f"  Error: {error}")

    return valid


# --------------------------------------------------
# 8. Define the datasets
# --------------------------------------------------

class RMFDDataset(Dataset):
    """Image-classification dataset for identification."""

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, identity_label, is_masked = self.records[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return (
            image,
            torch.tensor(identity_label, dtype=torch.long),
            torch.tensor(is_masked, dtype=torch.long),
        )


class VerificationPairsDataset(Dataset):
    """Image-pair dataset for same-person verification."""

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path_a, path_b, same_person = self.pairs[index]

        with Image.open(path_a) as image_a:
            image_a = image_a.convert("RGB")

            if self.transform is not None:
                image_a = self.transform(image_a)

        with Image.open(path_b) as image_b:
            image_b = image_b.convert("RGB")

            if self.transform is not None:
                image_b = self.transform(image_b)

        return (
            image_a,
            image_b,
            torch.tensor(same_person, dtype=torch.long),
        )


train_records = keep_readable_records(train_records)
verification_pairs = keep_readable_pairs(verification_pairs)

train_dataset = RMFDDataset(
    records=train_records,
    transform=preprocess,
)

verification_dataset = VerificationPairsDataset(
    pairs=verification_pairs,
    transform=preprocess,
)


# --------------------------------------------------
# 9. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

verification_loader = DataLoader(
    verification_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# --------------------------------------------------
# 10. Print dataset statistics
# --------------------------------------------------

def print_identification_statistics(name, records):
    identity_counts = Counter(label for _, label, _ in records)
    mask_counts = Counter(masked for _, _, masked in records)

    print(f"\n{name}:")
    print(f"  Total images: {len(records)}")
    print("  Identity counts:")

    for label, count in sorted(identity_counts.items()):
        print(f"    {idx_to_class[label]}: {count}")

    print(f"  Unmasked images: {mask_counts.get(0, 0)}")
    print(f"  Masked images:   {mask_counts.get(1, 0)}")


def print_verification_statistics(pairs):
    label_counts = Counter(label for _, _, label in pairs)

    print("\nVerification set:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Genuine pairs (label 1): {label_counts.get(1, 0)}")
    print(f"  Impostor pairs (label 0): {label_counts.get(0, 0)}")


print_identification_statistics("Training set", train_records)
print_verification_statistics(verification_pairs)


# --------------------------------------------------
# 11. Inspect example batches
# --------------------------------------------------

if len(train_dataset) > 0:
    images, identity_labels, mask_labels = next(iter(train_loader))

    print("\nExample identification batch:")
    print("  Images:", images.shape)  # [B, 3, 224, 224]
    print("  Identity labels:", identity_labels.shape)  # [B]
    print("  Mask labels:", mask_labels.shape)  # [B]
    print("  First identity labels:", identity_labels[:5].tolist())
    print("  First mask labels:", mask_labels[:5].tolist())

if len(verification_dataset) > 0:
    images_a, images_b, pair_labels = next(iter(verification_loader))

    print("\nExample verification batch:")
    print("  Images A:", images_a.shape)  # [B, 3, 224, 224]
    print("  Images B:", images_b.shape)  # [B, 3, 224, 224]
    print("  Pair labels:", pair_labels.shape)  # [B]
    print("  First pair labels:", pair_labels[:5].tolist())