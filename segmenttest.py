"""
Some tests on segformer b5 for semantic segmentation, loads some generated snowy images and applies the segformer model to them, and saves the segmentation results to the output folder. 
Also converts the segmentation results to color images using the Cityscapes color palette for visualization.
"""

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

input_folder = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\snowsegmen")
output_folder = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\segmentationpredictssnow")
output_folder.mkdir(parents=True, exist_ok=True)

processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

CITYSCAPES_PALETTE = np.array([
    [128, 64,128], [244, 35,232], [70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170,30], [220,220,0],
    [107,142,35], [152,251,152], [70,130,180], [220,20,60],
    [255,0,0], [0,0,142], [0,0,70], [0,60,100],
    [0,80,100], [0,0,230], [119,11,32],
], dtype=np.uint8)


def save_segmentation(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    original_size = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = F.interpolate(
        outputs.logits,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False
    )

    pred = logits.argmax(dim=1)[0].cpu().numpy()
    color_seg = CITYSCAPES_PALETTE[pred]
    Image.fromarray(color_seg).save(output_path)


for image_path in input_folder.glob("*.png"):
    output_path = output_folder / f"{image_path.stem}_seg.png"
    save_segmentation(image_path, output_path)
    print(f"Saved: {output_path}")