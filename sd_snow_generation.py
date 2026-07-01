"""
uses AUTOMATIC1111 WebUI and control net to run stable diffusion

input images from cityscapes leftImg8bit_trainvaltest dataset
2 outputs per input image(variable)

steps:
1. start AUTOMATIC1111 with API enabled:
   edit webui-user.bat and set:
   set COMMANDLINE_ARGS=--api

2. WebUI runs at:
   http://127.0.0.1:7860

3. ControlNet installed and working in webui

4. controlnet model used:
   control_v11p_sd15_canny.pth

   put model at
   stable-diffusion-webui/extensions/sd-webui-controlnet/models/

"""

import base64
import time
from pathlib import Path
import requests


#generation config
# ---------------- 

WEBUI_URL = "http://127.0.0.1:7860"

INPUT_DIR = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\final_dataset_clear_2")
OUTPUT_DIR = Path(r"C:\Users\Gurnoor\Documents\UVA\Scriptie\dataset\final_dataset_snow_2")



PROMPT = ( "realistic snowy urban driving scene, heavy snow accumulation on road and sidewalks, icy asphalt, winter weather, "
            "cloudy sky, heavy snowfall, photorealistic dashcam image, unchanged objects" ) 
NEGATIVE_PROMPT = ( "dissapearing objects, cartoon, anime, painting, render, unrealistic, blurry, distorted cars, " 
                   "melted objects, deformed vehicles, bad perspective, low quality" )



# main generation settings


SAMPLER_NAME = "DPM++ 2M"
SCHEDULER = "Karras"


STEPS = 30
CFG_SCALE = 7
DENOISING_STRENGTH = 0.75


#output image size in pixels
WIDTH = 1536
HEIGHT = 768

#number of snowy variants per input image
VARIANTS_PER_IMAGE = 1

#ControlNet settings
CONTROLNET_PIXEL_PERFECT = True
CONTROLNET_PREPROCESSOR = "canny"
CONTROLNET_MODEL = "control_v11p_sd15_canny"
CONTROLNET_WEIGHT = 0.9

#-1 for random seeds
BASE_SEED = -1


# HELPERS
#--------

def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(base64_string: str, output_path: Path) -> None:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    output_path.write_bytes(image_data)

#sanity check before generating the images
def check_webui_connection() -> None:
    try:
        response = requests.get(f"{WEBUI_URL}/sdapi/v1/sd-models", timeout=10)
        response.raise_for_status()
        print("Connected to WebUI API.")
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to WebUI. Make sure it is running at "
            "http://127.0.0.1:7862 with --api enabled."
        ) from exc
    
def generate_snowy_image(input_image_path: Path, output_path: Path, seed: int) -> None:
    input_image_b64 = encode_image_to_base64(input_image_path)

    payload = {
        "init_images": [input_image_b64],
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": STEPS,
        "cfg_scale": CFG_SCALE,
        "denoising_strength": DENOISING_STRENGTH,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "sampler_name": SAMPLER_NAME,
        "scheduler": SCHEDULER,

        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "enabled": True,
                        "module": CONTROLNET_PREPROCESSOR,
                        "model": CONTROLNET_MODEL,
                        "weight": CONTROLNET_WEIGHT,
                        "image": input_image_b64,
                        "resize_mode": "Crop and Resize",
                        "low_vram": False,
                        "processor_res": 768,
                        "threshold_a": 50,
                        "threshold_b": 200,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "pixel_perfect": CONTROLNET_PIXEL_PERFECT,
                        "control_mode": "Balanced"
                    }
                ]
            }
        }
    }

    response = requests.post(
        f"{WEBUI_URL}/sdapi/v1/img2img",
        json=payload,
        timeout=300
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Generation failed for {input_image_path.name}\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    result = response.json()

    if "images" not in result or not result["images"]:
        raise RuntimeError(f"No image returned for {input_image_path.name}")

    save_base64_image(result["images"][0], output_path)

#main
#----

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    check_webui_connection()

    image_paths = sorted(
        list(INPUT_DIR.glob("*.png")) +
        list(INPUT_DIR.glob("*.jpg")) +
        list(INPUT_DIR.glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    print(f"Found {len(image_paths)} input images.")
    print(f"Saving outputs to: {OUTPUT_DIR}")

    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"\n[{image_index}/{len(image_paths)}] Processing: {image_path.name}")

        for variant in range(VARIANTS_PER_IMAGE):
            if BASE_SEED == -1:
                seed = -1
            else:
                seed = BASE_SEED + variant

            output_name = f"{image_path.stem}_snow_v{variant + 1}.png"
            output_path = OUTPUT_DIR / output_name

            print(f"  Generating variant {variant + 1}/{VARIANTS_PER_IMAGE}...")

            try:
                generate_snowy_image(image_path, output_path, seed)
                print(f"  Saved: {output_path}")
            except Exception as exc:
                print(f"  FAILED: {exc}")

            time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()


