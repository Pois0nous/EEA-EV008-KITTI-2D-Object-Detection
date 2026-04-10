import os
import json
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import AutoBackbone

KITTI_ROOT = "/scratch/work/heidarr1/KITTI_ROOT"
SPLIT_FILE = os.path.join(KITTI_ROOT, "splits", "val.txt")
CACHE_DIR = "/scratch/work/heidarr1/cache/huggingface"

MODEL_DIR = "./outputs_frcnn_dinov3_kitti"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
PRED_DIR = os.path.join(MODEL_DIR, "kitti_val_predictions")

BACKBONE_NAME = "facebook/dinov3-convnext-base-pretrain-lvd1689m"

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
CLASS_TO_LABEL = {c: i + 1 for i, c in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS = {i + 1: c for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = 1 + len(CLASS_NAMES)

IOU_THRESH = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
MIN_HEIGHT = {"easy": 40, "moderate": 25, "hard": 25}
MAX_OCC = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNC = {"easy": 0.15, "moderate": 0.30, "hard": 0.50}
DIFFS = ["easy", "moderate", "hard"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_split_ids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_gt_file(path):
    objs = []
    dontcare = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            trunc = float(parts[1])
            occ = int(float(parts[2]))
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            obj = {
                "cls": cls,
                "trunc": trunc,
                "occ": occ,
                "bbox": [x1, y1, x2, y2],
                "height": y2 - y1,
            }

            if cls == "DontCare":
                dontcare.append(obj)
            else:
                objs.append(obj)

    return objs, dontcare


def parse_pred_file(path):
    preds = []
    if not os.path.exists(path):
        return preds

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:
                continue
            cls = parts[0]
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])
            score = float(parts[15])
            preds.append(
                {
                    "cls": cls,
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "height": y2 - y1,
                }
            )
    preds.sort(key=lambda x: -x["score"])
    return preds


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def difficulty_status(obj, diff):
    if obj["height"] < MIN_HEIGHT[diff]:
        return False
    if obj["occ"] > MAX_OCC[diff]:
        return False
    if obj["trunc"] > MAX_TRUNC[diff]:
        return False
    return True


def compute_ap40(recalls, precisions):
    recall_points = np.linspace(0.0, 1.0, 40)
    ap = 0.0
    for r in recall_points:
        valid = precisions[recalls >= r]
        p = 0.0 if len(valid) == 0 else np.max(valid)
        ap += p
    return 100.0 * ap / 40.0


def evaluate_class_diff(gt_dir, pred_dir, image_ids, cls_name, diff):
    iou_thr = IOU_THRESH[cls_name]

    all_scores = []
    all_tp = []
    all_fp = []
    num_valid_gt = 0

    for img_id in image_ids:
        gt_objs, dontcare = parse_gt_file(os.path.join(gt_dir, f"{img_id}.txt"))
        preds = parse_pred_file(os.path.join(pred_dir, f"{img_id}.txt"))

        valid_gt = []
        ignored_gt = []

        for g in gt_objs:
            if g["cls"] != cls_name:
                continue
            if difficulty_status(g, diff):
                valid_gt.append(g)
            else:
                ignored_gt.append(g)

        num_valid_gt += len(valid_gt)
        matched = [False] * len(valid_gt)

        preds_cls = [p for p in preds if p["cls"] == cls_name and p["height"] >= MIN_HEIGHT[diff]]

        for p in preds_cls:
            best_iou = -1.0
            best_idx = -1

            for i, g in enumerate(valid_gt):
                if matched[i]:
                    continue
                ov = box_iou(p["bbox"], g["bbox"])
                if ov >= iou_thr and ov > best_iou:
                    best_iou = ov
                    best_idx = i

            if best_idx >= 0:
                matched[best_idx] = True
                all_scores.append(p["score"])
                all_tp.append(1)
                all_fp.append(0)
                continue

            ignore_det = False
            for g in ignored_gt:
                if box_iou(p["bbox"], g["bbox"]) >= iou_thr:
                    ignore_det = True
                    break

            if not ignore_det:
                for dc in dontcare:
                    if box_iou(p["bbox"], dc["bbox"]) >= 0.5:
                        ignore_det = True
                        break

            if ignore_det:
                continue

            all_scores.append(p["score"])
            all_tp.append(0)
            all_fp.append(1)

    if num_valid_gt == 0 or len(all_scores) == 0:
        return {"AP40": 0.0, "num_gt": num_valid_gt}

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(1, num_valid_gt)
    precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)

    ap = compute_ap40(recalls, precisions)
    return {"AP40": float(ap), "num_gt": int(num_valid_gt)}


class KittiImageOnlyDataset(Dataset):
    def __init__(self, root, ids):
        self.root = root
        self.ids = ids
        self.image_dir = os.path.join(root, "training", "image_2")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.image_dir, f"{img_id}.png")
        image = np.array(Image.open(path).convert("RGB"))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return img_id, image


def collate_fn(batch):
    ids, images = zip(*batch)
    return list(ids), list(images)


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


@torch.no_grad()
def export_predictions():
    os.makedirs(PRED_DIR, exist_ok=True)

    image_ids = read_split_ids(SPLIT_FILE)
    ds = KittiImageOnlyDataset(KITTI_ROOT, image_ids)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    done = 0
    for batch_ids, images in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for img_id, out in zip(batch_ids, outputs):
            lines = []
            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score < 0.05:
                    continue
                cls_name = LABEL_TO_CLASS.get(int(label))
                if cls_name is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                lines.append(
                    f"{cls_name} -1 -1 -10 "
                    f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                    f"-1 -1 -1 -1000 -1000 -1000 -10 {float(score):.6f}"
                )

            with open(os.path.join(PRED_DIR, f"{img_id}.txt"), "w") as f:
                for line in lines:
                    f.write(line + "\n")

            done += 1
            if done % 100 == 0 or done == len(image_ids):
                print(f"exported {done}/{len(image_ids)}")

    return image_ids


def main():
    image_ids = export_predictions()
    gt_dir = os.path.join(KITTI_ROOT, "training", "label_2")

    print("\nKITTI-style local evaluation (AP40)\n")
    for cls_name in CLASS_NAMES:
        print(f"{cls_name}:")
        for diff in DIFFS:
            out = evaluate_class_diff(gt_dir, PRED_DIR, image_ids, cls_name, diff)
            print(f"  {diff:9s} AP40 = {out['AP40']:.2f}   num_gt = {out['num_gt']}")
        print()


if __name__ == "__main__":
    main()