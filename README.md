
# American Sign Language (ASL) Translation

This project to be as part of UST Vision AI SEIS766.

## Achitecture
- R(2+1)D-18 pretrained on Kinetics-400.

## Setup

### Create and activate conda environment
  conda create -p ./env python=3.10 -y
  conda activate ./env

### Install PyTorch with CUDA 12.6
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

### Install remaining dependencies
  pip install -r requirements.txt

  ---
## Download Dataset
- Dataset used can be found on: 
    - WLASL: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed (last access 4/15/2026).
    - ASL-Citizen: https://www.kaggle.com/datasets/abd0kamel/asl-citizen (last access 4/21/2026)
    - How2Sign: https://how2sign.github.io/ (last access 4/25/2026)

### ASL-Citizen (~26 GB)
  env/python.exe scripts/download_data.py --dataset abd0kamel/asl-citizen

### WLASL
  env/python.exe scripts/download_data.py --dataset risangbaskoro/wlasl-processed

  ---
## Train

### ASL-Citizen — all 2731 classes
  env/python.exe train.py --config configs/config_aslcitizen_full.yaml

### ASL-Citizen — top 100 classes only
  env/python.exe train.py --config configs/config_aslcitizen100.yaml

### Resume from checkpoint
  env/python.exe train.py --config configs/config_aslcitizen_full.yaml --resume checkpoints/aslcitizen_full/last.pth

  ---
  Save Results (plots + summary)

  env/python.exe save_results.py --config configs/config_aslcitizen_full.yaml


## Results
![Examples of Dataset](docs/images/dataset_samples.png)
![Training Curve](docs/images/training_curve.png)
![Confusion Matrix](docs/images/conf_matrix.png)

## Example Outputs
![Example of prediction](docs/images/example_prediction.png)

## Future Work
- convert notebooks to scripts
- Add Description generator model
- Add more robust classification (live image)
