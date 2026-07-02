# Hyperborea: Snowy Driving Scene Generation

This repository contains the code used for my bachelor thesis:  
**Hyperborea: A Generative AI Approach to Snowy Driving Scene Datasets and Benchmarks**.

The project uses Stable Diffusion, ControlNet and RealisticVision v5.1 to generate snowy versions of clear-weather Cityscapes images. The generated images are evaluated using manual inspection, Fréchet Inception Distance (FID) and semantic segmentation with SegFormer-B5.

## Repository structure

- `docs/`: Final thesis and delta document.
- `figures/`: Figures used in the thesis.
- `results/`: Output tables and evaluation results.
- `scripts/`: Python scripts for generation, FID preparation and segmentation evaluation.
- `settings/`: Used prompts and generation settings.
- `requirements.txt`: Python dependencies.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
