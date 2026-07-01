"""
This script runs the chosen segformer model on either the the clear Frankfurt validation images or the generated snowy Frankfurt validation images. It loads the matching Cityscapes 
ground-truth labels, converts the original Cityscapes label IDs to the 19 train IDs used by SegFormer, accumulates a confusion matrix over the dataset, and reports per-class IoU and mean IoU.

Earlier tests were done by loading segformer-b0, but now segformer-b5 is used for better accuracy.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# change image path to either directory for clear or snow validation
IMAGE_DIR = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\val_frankfurt_snow")
LABEL_ROOT = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\cityscapes\gtFine\val")

NUM_CLASSES = 19
IGNORE_LABEL = 255


CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,
    10: 255,
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,   # pole
    18: 255,
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255,
    30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255
}


CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]


def cityscapes_label_ids_to_train_ids(label: np.ndarray) -> np.ndarray:
    converted = np.full_like(label, IGNORE_LABEL, dtype=np.uint8)

    for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        converted[label == label_id] = train_id

    return converted


def get_original_cityscapes_stem(image_stem: str) -> str:
    # generated example:
    # frankfurt_000000_000294_leftImg8bit_snow_v1
    # should become:
    # frankfurt_000000_000294
    stem = image_stem

    for suffix in ["_snow_v1", "_snow_v2", "_snow_v3", "_snow"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    if stem.endswith("_leftImg8bit"):
        stem = stem.replace("_leftImg8bit", "")

    return stem


def find_label_path(image_path: Path) -> Path:
    original_stem = get_original_cityscapes_stem(image_path.stem)
    city = original_stem.split("_")[0]
    label_name = f"{original_stem}_gtFine_labelIds.png"
    return LABEL_ROOT / city / label_name


def update_confusion_matrix(confusion_matrix, prediction, target):
    valid = target != IGNORE_LABEL

    prediction = prediction[valid].astype(np.int64)
    target = target[valid].astype(np.int64)

    mask = (target >= 0) & (target < NUM_CLASSES) & (prediction >= 0) & (prediction < NUM_CLASSES)
    prediction = prediction[mask]
    target = target[mask]

    indices = NUM_CLASSES * target + prediction
    matrix = np.bincount(indices, minlength=NUM_CLASSES ** 2)
    matrix = matrix.reshape(NUM_CLASSES, NUM_CLASSES)

    confusion_matrix += matrix

def print_class_counts(confusion_matrix):
    gt_pixels = confusion_matrix.sum(axis=1)
    pred_pixels = confusion_matrix.sum(axis=0)
    correct_pixels = np.diag(confusion_matrix)

    print("\nClass pixel counts")
    print("------------------")
    for name, gt, pred, correct in zip(CLASS_NAMES, gt_pixels, pred_pixels, correct_pixels):
        print(f"{name:15s} GT: {gt:10d} | Pred: {pred:10d} | Correct: {correct:10d}")    

def print_top_confusions_for_class(confusion_matrix, class_name):
    class_index = CLASS_NAMES.index(class_name)

    row = confusion_matrix[class_index]
    total = row.sum()

    print(f"\nTop predictions for GT class: {class_name}")
    print("--------------------------------")
    if total == 0:
        print("No ground-truth pixels for this class.")
        return

    top_indices = np.argsort(row)[::-1][:5]

    for idx in top_indices:
        percentage = 100 * row[idx] / total
        print(f"{CLASS_NAMES[idx]:15s}: {row[idx]:10d} pixels ({percentage:.2f}%)")

def compute_iou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth = confusion_matrix.sum(axis=1)
    predicted = confusion_matrix.sum(axis=0)
    union = ground_truth + predicted - intersection

    iou = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    valid_classes = union > 0
    iou[valid_classes] = intersection[valid_classes] / union[valid_classes]

    miou = np.nanmean(iou)

    return iou, miou, valid_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    image_paths = sorted(
        list(IMAGE_DIR.rglob("*_leftImg8bit.png")) +
        list(IMAGE_DIR.rglob("*_leftImg8bit_snow_v1.png")) +
        list(IMAGE_DIR.rglob("*_leftImg8bit_snow_v2.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for image_path in tqdm(image_paths):
        label_path = find_label_path(image_path)

        if not label_path.exists():
            print(f"Skipping, label not found: {image_path.name}")
            continue

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        label_image = Image.open(label_path)

        raw_label = np.array(label_image)

        if label_image.size != original_size:
            label_image = label_image.resize(original_size, Image.NEAREST)

        label_ids = np.array(label_image)

        if label_ids.ndim == 3:
            label_ids = label_ids[:, :, 0]

        target = cityscapes_label_ids_to_train_ids(label_ids)

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )

        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        update_confusion_matrix(confusion_matrix, prediction, target)

    class_iou, miou, valid_classes = compute_iou(confusion_matrix)

    print("\nResults")
    print("-------")
    print_class_counts(confusion_matrix)
    print_top_confusions_for_class(confusion_matrix, "car")
    print(f"mIoU over present classes: {miou:.4f}")

    print("\nClass IoU")
    print("---------")
    for name, iou, valid in zip(CLASS_NAMES, class_iou, valid_classes):
        if valid:
            print(f"{name:15s}: {iou:.4f}")
        else:
            print(f"{name:15s}: not present")


if __name__ == "__main__":
    main()