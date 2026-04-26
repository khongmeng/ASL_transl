
# American Sign Language (ASL) Translation

This project to be as part of UST Vision AI SEIS766.

---
## Achitecture
- R(2+1)D-18 pretrained on Kinetics-400.

---
## Usage

### Create and activate conda environment
```bash
  conda create -p ./env python=3.10 -y
  conda activate ./env
```
### Install PyTorch with CUDA 12.6
```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
### Install remaining dependencies
```bash
  pip install -r requirements.txt
```
  ---
## Dataset
- Dataset used can be found on: 
    - WLASL: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed (last access 4/15/2026).
    - ASL-Citizen: https://www.kaggle.com/datasets/abd0kamel/asl-citizen (last access 4/21/2026)
    - How2Sign: https://how2sign.github.io/ (last access 4/25/2026)

```bash
env/python.exe scripts/download_data.py --dataset <choose a dataset>
```
  ---
### Train

```bash
env/python.exe train.py --config configs/<choose a config file>
```
### Resume from checkpoint
```bash
env/python.exe train.py --config <choose a config file> --resume checkpoints/<choose a checkpoint>
```
---
## Save Results (plots + summary)
```bash
env/python.exe save_results.py --config configs/<choose a config file>
```
---
## Results
![ASL-Citizen 100](results/aslcitizen100/training_curves.png)
![WLASL 100](results/wlasl100/training_curves.png)
![WLASL 2000](results/wlasl2000/training_curves.png)


---
## Example
![Input](docs/images/example_prediction.png)
![Output](docs/images/example_prediction.png)
