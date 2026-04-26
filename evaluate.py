"""
Evaluate a trained checkpoint on the WLASL test split.

Usage:
  conda run python evaluate.py --checkpoint checkpoints/best.pth
  conda run python evaluate.py --checkpoint checkpoints/best.pth --split test
"""

import argparse
import json

import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import WLASLDataset
from src.model import build_model
from src.utils import accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    label_to_idx = ckpt.get('label_to_idx', {})
    num_classes  = len(label_to_idx) if label_to_idx else cfg['model']['num_classes']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device    : {device}")
    print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    print(f"Split     : {args.split}")

    ds = WLASLDataset(
        json_path=cfg['data']['json_path'],
        video_dir=cfg['data']['video_dir'],
        split=args.split,
        num_frames=cfg['data']['num_frames'],
        resize=cfg['data']['resize'],
        crop=cfg['data']['crop'],
        augment=False,
    )
    num_classes = ds.num_classes
    print(f"Samples   : {len(ds)}  |  Classes: {num_classes}\n")

    loader = DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
    )

    model = build_model(cfg['model']['backbone'], num_classes, dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    top1_sum = top5_sum = n = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating"):
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(clips)

            t1, t5 = accuracy(logits, labels, topk=(1, 5))
            bs       = labels.size(0)
            top1_sum += t1 * bs
            top5_sum += t5 * bs
            n        += bs

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    top1 = top1_sum / n
    top5 = top5_sum / n

    print(f"\nResults on [{args.split}] split:")
    print(f"  Top-1 accuracy: {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  Top-5 accuracy: {top5:.4f}  ({top5*100:.2f}%)")
    print(f"  Samples evaluated: {n}")


if __name__ == '__main__':
    main()
