"""
Resizes images from the input folder to 1536x768, and saves them to output folder. had to do this for the FID evaluation, so all images were the same resolution as the generated snow images

"""

from pathlib import Path
from PIL import Image

INPUT_DIR = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\FID test\clear")
OUTPUT_DIR = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\FID test\clear_resized")

TARGET_SIZE = (1536, 768)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg"))
)

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.BICUBIC)
    image.save(OUTPUT_DIR / image_path.name)

print(f"Resized {len(image_paths)} images.")