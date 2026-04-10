# EEA-EV008-KITTI-2D-Object-Detection

Group Project for EEA-EV008 Visual Perception and Planning for Autonomous Driving.

This project demonstrates the implementation of a 2D Object Detection pipeline using the KITTI dataset, specifically focusing on the detection of Cars, Pedestrians, and Cyclists. It contains two parallel approaches to object detection: an adaptation of Faster R-CNN with a DINOv3 context-aware backbone, and a real-time YOLO-based detection pipeline.

## Project Structure

- `DINO_FRCNN/`: Contains the implementation for a two-stage object detector incorporating Faster R-CNN with a modern DINOv3 (ConvNeXt-based) backbone.
  - `train.py`: The PyTorch training loop incorporating mixed data augmentations (via Albumentations), bounding box parsing, and custom metric evaluations.
  - `eval.py`: Evaluation script using COCO metrics tailored for validating the model weights against the KITTI val-split.
- `YOLO/`: Contains the pipeline for a real-time YOLO detector.
  - `kitti.yaml`: Dataset configuration file for mapping KITTI constraints to YOLO framework logic.
  - `YOLO.ipynb`: Interactive notebook for visualizing, building, and training the YOLO model.
  - `eval.ipynb`: Post-processing script executing batch inference over validation images and translating output coordinate matrices precisely into the rigid 16-column KITTI standard for metric calculation.

## Dataset Setup

The KITTI 2D Object Detection dataset must be configured before running these tracks. Annotations consist of cars, pedestrians, and cyclists. Filter out "DontCare" classes for optimal convergence.

For the DINO FRCNN track, place data according to:
`KITTI_ROOT = "/scratch/work/heidarr1/KITTI_ROOT"`
For YOLO Track, place data according to YOLO standards and paths in `kitti.yaml` and `eval.ipynb`.

## Requirements

Ensure you have a recent version of PyTorch.
Dependencies include:

- `ultralytics`
- `albumentations`
- `transformers`
- `torchvision`
- `tqdm`
- `pycocotools`
