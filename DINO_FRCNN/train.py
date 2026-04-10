import os
import json
import math
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import AutoBackbone

KITTI_ROOT = "/scratch/work/heidarr1/KITTI_ROOT"
SPLIT_DIR = os.path.join(KITTI_ROOT, "splits")
CACHE_DIR = "/scratch/work/heidarr1/cache/huggingface"
OUTPUT_DIR = "./outputs_frcnn_dinov3_kitti"

BACKBONE_NAME = "facebook/dinov3-convnext-base-pretrain-lvd1689m"

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
NUM_CLASSES = 1 + len(CLASS_NAMES)  # + background
CLASS_TO_LABEL = {c: i + 1 for i, c in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS = {i + 1: c for i, c in enumerate(CLASS_NAMES)}

SEED = 42
IMAGE_MAX_SIZE = 1024
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PRINT_FREQ = 50
FREEZE_BACKBONE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def read_split_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_kitti_label_file(label_path):
    boxes = []
    labels = []
    areas = []
    iscrowd = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls == "DontCare" or cls not in CLASS_TO_LABEL:
                continue

            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1 or h < 1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_LABEL[cls])
            areas.append(w * h)
            iscrowd.append(0)

    return boxes, labels, areas, iscrowd


class KittiDetectionDataset(Dataset):
    def __init__(self, root, split_ids, train=True, max_size=1024):
        self.root = Path(root)
        self.ids = split_ids
        self.train = train
        self.image_dir = self.root / "training" / "image_2"
        self.label_dir = self.root / "training" / "label_2"

        if train:
            self.aug = A.Compose(
                [
                    A.LongestMaxSize(max_size=max_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.0,
                    clip=True,
                ),
            )
        else:
            self.aug = A.Compose(
                [A.LongestMaxSize(max_size=max_size, p=1.0)],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.0,
                    clip=True,
                ),
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image_path = self.image_dir / f"{img_id}.png"
        label_path = self.label_dir / f"{img_id}.txt"

        image = np.array(Image.open(image_path).convert("RGB"))
        boxes, labels, areas, iscrowd = parse_kitti_label_file(label_path)

        out = self.aug(image=image, bboxes=boxes, class_labels=labels)

        if self.train and len(out["bboxes"]) == 0 and len(boxes) > 0:
            fallback = A.Compose(
                [A.LongestMaxSize(max_size=IMAGE_MAX_SIZE, p=1.0)],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_labels"],
                    min_visibility=0.0,
                    clip=True,
                ),
            )
            out = fallback(image=image, bboxes=boxes, class_labels=labels)

        image = out["image"]
        boxes = out["bboxes"]
        labels = out["class_labels"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([int(img_id)], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([int(img_id)], dtype=torch.int64),
                "area": areas,
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class DinoV3SingleScaleBackbone(nn.Module):
    def __init__(self, backbone_name, cache_dir=None):
        super().__init__()
        self.backbone = AutoBackbone.from_pretrained(backbone_name, cache_dir=cache_dir)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            out = self.backbone(dummy)
            self.out_channels = out.feature_maps[-1].shape[1]

    def forward(self, x):
        out = self.backbone(x)
        feat = out.feature_maps[-1]
        return {"0": feat}


def build_model():
    backbone = DinoV3SingleScaleBackbone(BACKBONE_NAME, cache_dir=CACHE_DIR)

    if FREEZE_BACKBONE:
        for p in backbone.parameters():
            p.requires_grad = False

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=512,
        max_size=1333,
    )
    return model


def move_to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    was_training = model.training
    model.train()

    losses = []
    for images, targets in data_loader:
        images, targets = move_to_device(images, targets, device)
        loss_dict = model(images, targets)
        losses.append(sum(loss_dict.values()).item())

    if not was_training:
        model.eval()

    return float(np.mean(losses)) if losses else math.nan


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ids = read_split_ids(os.path.join(SPLIT_DIR, "train.txt"))
    val_ids = read_split_ids(os.path.join(SPLIT_DIR, "val.txt"))

    train_ds = KittiDetectionDataset(KITTI_ROOT, train_ids, train=True, max_size=IMAGE_MAX_SIZE)
    val_ds = KittiDetectionDataset(KITTI_ROOT, val_ids, train=False, max_size=IMAGE_MAX_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = build_model().to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    run_cfg = {
        "backbone_name": BACKBONE_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "freeze_backbone": FREEZE_BACKBONE,
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = []

        for step, (images, targets) in enumerate(train_loader, start=1):
            images, targets = move_to_device(images, targets, DEVICE)

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running.append(loss.item())

            if step % PRINT_FREQ == 0 or step == len(train_loader):
                mean_loss = sum(running[-PRINT_FREQ:]) / min(len(running), PRINT_FREQ)
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={mean_loss:.4f} "
                    f"cls={loss_dict['loss_classifier'].item():.4f} "
                    f"box={loss_dict['loss_box_reg'].item():.4f} "
                    f"obj={loss_dict['loss_objectness'].item():.4f} "
                    f"rpn={loss_dict['loss_rpn_box_reg'].item():.4f}"
                )

        val_loss = evaluate_loss(model, val_loader, DEVICE)
        print(f"\nepoch={epoch} val_loss={val_loss:.4f}\n")

        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            with open(os.path.join(OUTPUT_DIR, "best_epoch.txt"), "w") as f:
                f.write(str(epoch))

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last_model.pt"))


if __name__ == "__main__":
    main()