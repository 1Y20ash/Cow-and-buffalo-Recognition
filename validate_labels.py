"""Validate the 42-class cattle/buffalo breed dataset and label contract.

Run from the repository root:
    python validate_labels.py

The dataset itself is intentionally not committed to GitHub. The script validates
the local dataset directory when it is available and always validates labels.txt.
"""
from __future__ import annotations

from pathlib import Path
import sys

EXPECTED_CLASSES = [
    "Alambadi",
    "Amritmahal",
    "Ayrshire",
    "Banni",
    "Bargur",
    "Bhadawari",
    "Brown_Swiss",
    "Dangi",
    "Deoni",
    "Gir",
    "Guernsey",
    "Hallikar",
    "Hariana",
    "Holstein_Friesian",
    "Jaffrabadi",
    "Jersey",
    "Kangayam",
    "Kankrej",
    "Kasargod",
    "Kenkatha",
    "Kherigarh",
    "Khillari",
    "Krishna_Valley",
    "Malnad_gidda",
    "Mehsana",
    "Murrah",
    "Nagori",
    "Nagpuri",
    "Nili_Ravi",
    "Nimari",
    "Ongole",
    "Pandharpuri",
    "Pulikulam",
    "Rathi",
    "Red_Dane",
    "Red_Sindhi",
    "Sahiwal",
    "Surti",
    "Tharparkar",
    "Toda",
    "Umblachery",
    "Vechur",
]

DATASET_CANDIDATES = [
    Path("Indian_bovine_breeds") / "Indian_bovine_breeds",
    Path("Indian_bovine_breeds"),
]


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    return labels


def validate_labels(labels: list[str]) -> list[str]:
    errors: list[str] = []

    if len(labels) != len(EXPECTED_CLASSES):
        errors.append(
            f"labels.txt contains {len(labels)} labels; expected {len(EXPECTED_CLASSES)}."
        )

    if len(labels) != len(set(labels)):
        errors.append("labels.txt contains duplicate labels.")

    if labels != EXPECTED_CLASSES:
        missing = sorted(set(EXPECTED_CLASSES) - set(labels))
        unexpected = sorted(set(labels) - set(EXPECTED_CLASSES))
        if missing:
            errors.append(f"Missing expected labels: {', '.join(missing)}")
        if unexpected:
            errors.append(f"Unexpected labels: {', '.join(unexpected)}")
        if not missing and not unexpected:
            errors.append("Label order differs from the canonical training order.")

    return errors


def find_dataset() -> Path | None:
    for candidate in DATASET_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def validate_dataset(dataset: Path) -> list[str]:
    errors: list[str] = []
    folders = sorted(
        path.name for path in dataset.iterdir() if path.is_dir()
    )

    if folders != EXPECTED_CLASSES:
        missing = sorted(set(EXPECTED_CLASSES) - set(folders))
        unexpected = sorted(set(folders) - set(EXPECTED_CLASSES))
        if missing:
            errors.append(f"Dataset is missing class folders: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"Dataset contains unexpected class folders: {', '.join(unexpected)}"
            )
        if not missing and not unexpected:
            errors.append("Dataset class-folder order/content differs from canonical names.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    empty_classes = []
    total_images = 0

    for class_name in EXPECTED_CLASSES:
        class_dir = dataset / class_name
        if not class_dir.is_dir():
            continue
        count = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        total_images += count
        if count == 0:
            empty_classes.append(class_name)

    if empty_classes:
        errors.append(
            f"Class folders contain no supported images: {', '.join(empty_classes)}"
        )

    print(f"Dataset: {dataset}")
    print(f"Classes: {len(folders)}")
    print(f"Images found: {total_images}")

    return errors


def main() -> int:
    errors = []

    labels = load_labels(Path("labels.txt"))
    errors.extend(validate_labels(labels))
    print(f"Labels: {len(labels)}")

    dataset = find_dataset()
    if dataset is None:
        print("Dataset: not present in repository (expected; dataset is gitignored).")
        print("Skipping local dataset-folder validation.")
    else:
        errors.extend(validate_dataset(dataset))

    if errors:
        print("\nVALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nVALIDATION PASSED")
    print(f"Canonical class count: {len(EXPECTED_CLASSES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
