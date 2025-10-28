"""utils.py

Small utility functions and lightweight helpers used across the repository.

This module provides:
- geometry/tensor helpers used by model code (autopad, dist2bbox, make_anchors, make_divisible)
- a small IO helper (unzip_file) with a progress bar for archive extraction
- a compact per-class IoU / accuracy tracker (AccuracyIoU) for additional validation reporting

Design goals:
- Keep utilities lightweight and dependency-minimal so they can be imported in varied environments.
- Provide clear docstrings and comments to make their intended behavior obvious to users and reviewers.
"""
from __future__ import annotations
import torch
import pandas as pd
import zipfile
import os
import math
from tqdm import tqdm
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Auto-pad convolution kernels to preserve 'same' spatial output size.

    Args:
        k (int | sequence): kernel size
        p (int | None): padding. If None, computed to preserve output size.
        d (int): dilation factor

    Returns:
        int | list: padding value(s) suitable for given kernel/dilation.
    """
    if d > 1:
        # If dilation > 1, effective kernel size increases.
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        # Compute symmetric padding that preserves spatial dims ("same" conv)
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor center points and matching stride tensors from feature maps.

    Notes:
        - `feats` is expected to be a list of feature-tensors or shape-like entries corresponding
          to detection heads. We extract spatial dims and generate (x,y) center points per cell.
        - grid_cell_offset controls whether anchors are centered (0.5) or aligned to corner (0.0).

    Args:
        feats (list[Tensor] | sequence): list of feature maps produced by backbone/head
        strides (list[int]): stride per feature map relative to input image
        grid_cell_offset (float): offset within each grid cell to place anchor center

    Returns:
        tuple:
            - anchor_points (Tensor[N,2]): concatenated anchor center coordinates (x,y)
            - stride_tensor (Tensor[N,1]): concatenated stride values per anchor
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        # Height and width of the feature map
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # create grid coordinates (shifted by grid_cell_offset)
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        # meshgrid with 'ij' indexing so first dim = y, second dim = x
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Convert predicted distances (left, top, right, bottom) from an anchor point to bounding boxes.

    Args:
        distance (Tensor): predicted distances in format (..., 4) representing [l, t, r, b]
        anchor_points (Tensor): anchor center locations (..., 2)
        xywh (bool): return format; if True returns [cx, cy, w, h], else returns [x1, y1, x2, y2]
        dim (int): dimension along which to concatenate results

    Returns:
        Tensor: boxes in either xywh (center) or xyxy (corner) format.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        # center coordinates and width/height
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_divisible(x, divisor):
    """
    Round `x` up to the nearest value divisible by `divisor`.

    This is commonly used when computing channel counts or input sizes that must align with
    network downsampling strides.

    Args:
        x (int): input value to be rounded up
        divisor (int | Tensor): divisor to enforce divisibility by

    Returns:
        int: smallest integer >= x that is divisible by divisor
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def unzip_file(zip_path: str, target_dir: str) -> None:
    """
    Extract a zip archive to target_dir with a tqdm progress bar.

    This helper is safe to call in scripts and will create `target_dir` if missing.

    Args:
        zip_path (str): path to zip file
        target_dir (str): destination folder for extracted contents
    """
    # Ensure the target directory exists; create it if it doesn't
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Open the zip file and extract while reporting progress
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        info_list = zip_ref.infolist()
        total_size = sum(zinfo.file_size for zinfo in info_list)

        # Create progress bar with byte units and automatic scaling (KB/MB)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Unzipping") as pbar:
            for zinfo in info_list:
                zip_ref.extract(zinfo, target_dir)
                pbar.update(zinfo.file_size)


class AccuracyIoU:
    """
    Lightweight per-class IoU and accuracy tracker.

    Purpose:
        Provide extra per-class reporting complementary to mAP. This class tracks:
        - per-class total IoU (sum of IoU for matched correct-class detections)
        - per-class true positives (TP), false positives (FP), false negatives (FN)
        - per-class counts of ground-truth instances (GT)
        - simple handling of true negative / no-object cases for bookkeeping

    Notes:
        - This is intended as a diagnostic helper (human-readable tables). It is not a
          replacement for DetMetrics / COCO evaluation but provides quick insights.
        - Inputs are expected in native image coordinates (xyxy) and detection format:
          [x1, y1, x2, y2, confidence, class_index].
    """

    def __init__(self, class_names, nc, conf=0.25, iou_thres=0.45):
        """Initialize internal counters and thresholds.

        Args:
            class_names (Sequence[str] | dict): mapping/indexed names for classes
            nc (int): number of classes
            conf (float): confidence threshold for considering a detection
            iou_thres (float): IoU threshold to consider a match between detection and GT
        """
        self.nc = nc  # number of classes
        self.class_names = class_names
        # Default fallback for some ultralytics arg edge-cases; keep behavior stable.
        self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold for matching
        # Per-class accumulators
        self.class_iou = {i: 0.0 for i in range(nc)}  # Sum IoU for matched correct-class detections
        self.class_tp = {i: 0 for i in range(nc)}     # True positives per class
        self.class_fp = {i: 0 for i in range(nc)}     # False positives per class
        self.class_fn = {i: 0 for i in range(nc)}     # False negatives per class
        self.class_gt = {i: 0 for i in range(nc)}     # Ground truth count per class
        # Simple background/no-object bookkeeping
        self.true_negative = 0
        self.false_negative = 0
        self.total_negatives = 0

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update internal counters for a single-image batch.

        Args:
            detections (Tensor|None): detections tensor of shape [N, >=6] or None when no detections.
            gt_bboxes (Tensor[M,4]): ground-truth boxes (xyxy)
            gt_cls (Tensor[M]): ground-truth class indices

        Behavior:
            - Filters detections by self.conf
            - When no GT and no detections: increments true_negative
            - When detections but no GT: increments FP for predicted classes
            - When GT present: computes IoU matrix and matches detections to GT based on iou_thres,
              then updates TP/FP/FN and accumulates IoU for correct-class matches.
        """
        if detections is None:
            # No detections at all
            if gt_cls.shape[0] != 0:
                # Objects present but no detections => false negative
                self.false_negative += 1
            else:
                # No objects and no detections => true negative
                self.true_negative += 1
                self.total_negatives += 1
            return

        # Keep only detections above confidence threshold
        detections = detections[detections[:, 4] > self.conf]

        if gt_cls.shape[0] == 0:
            # No ground truth objects in this image
            if detections.shape[0] == 0:
                # Nothing predicted and nothing present
                self.true_negative += 1
                self.total_negatives += 1
            else:
                # Detections but no objects => false positives for the detected classes
                detection_classes = detections[:, 5].int().cpu().numpy()
                self.total_negatives += 1
                for dc in detection_classes:
                    self.class_fp[dc] += 1
            return

        # Update ground truth counts per class
        gt_classes = gt_cls.int().cpu().numpy()
        for gc in gt_classes:
            self.class_gt[gc] += 1

        # Compute pairwise IoU between GT and detections: shape [M, N]
        iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N]
        detection_classes = detections[:, 5].int().cpu().numpy()

        # Find candidate matches above IoU threshold
        x = torch.where(iou > self.iou_thres)

        if x[0].shape[0]:  # If there are matches
            # Build a matches array: [gt_idx, det_idx, iou_value]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Greedy selection: sort by IoU descending and keep unique detection & GT assignments
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detection matches
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique GT matches

            m0, m1, _ = matches.transpose().astype(int)  # m0: gt indices, m1: detection indices

            # For each GT entry, check whether it was matched and update TP/FP/FN accordingly
            for i, gc in enumerate(gt_classes):
                j = m0 == i
                if sum(j) == 1:  # matched
                    dc = detection_classes[m1[j][0]]  # predicted class for matched detection
                    if dc == gc:
                        # Correct class and matched -> TP and accumulate IoU
                        iou_value = matches[j, 2][0]
                        self.class_iou[gc] += iou_value
                        self.class_tp[gc] += 1
                    else:
                        # Matched but wrong class -> counts as FP for predicted class and FN for GT class
                        self.class_fp[dc] += 1
                        self.class_fn[gc] += 1
                else:
                    # GT was not matched -> false negative
                    self.class_fn[gc] += 1
        else:
            # No matches above IoU threshold: all detections are FP, all GT are FN
            detection_classes = detections[:, 5].int().cpu().numpy()
            for dc in detection_classes:
                self.class_fp[dc] += 1
            for gc in gt_classes:
                self.class_fn[gc] += 1

    def get_metrics(self):
        """
        Compute average IoU and a simple accuracy per class.

        Returns:
            tuple(dict, dict):
                - iou_per_class: average IoU for each class name (IoU sum / TP)
                - acc_per_class: accuracy estimate per class (TP / (TP + FP + FN))
        """
        iou_per_class = {}
        acc_per_class = {}

        total = {i: self.class_tp[i] + self.class_fn[i] + self.class_fp[i] for i in range(self.nc)}
        for cls in range(self.nc):
            iou_per_class[self.class_names[cls]] = (self.class_iou[cls] / self.class_tp[cls]) if self.class_tp[cls] > 0 else 0.0
            acc_per_class[self.class_names[cls]] = (self.class_tp[cls] / total[cls]) if total[cls] > 0 else 0.0

        return iou_per_class, acc_per_class

    def print(self, scores_dict):
        """
        Print a human-readable table combining provided scores_dict (per-class metrics)
        with IoU and accuracy computed by this helper.

        The function augments scores_dict in-place and logs a pandas DataFrame via LOGGER.
        """
        iou_per_class, acc_per_class = self.get_metrics()
        for key, value in iou_per_class.items():
            scores_dict[key]["IoU"] = value
            scores_dict[key]["Accuracy"] = acc_per_class[key]
        # Create a DataFrame from the scores dictionary and append mean row for easy inspection
        df = pd.DataFrame(scores_dict).T
        df.loc['Average'] = df.mean()
        # Ensure integer columns are shown as integers
        df['Images'] = df['Images'].astype(int)
        df['Instances'] = df['Instances'].astype(int)
        LOGGER.info(df.to_string(index=True, justify='left', float_format='%.3f'))
        # Reset counters after printing to prepare for next evaluation run
        self.reset()

    def reset(self):
        """Reset all accumulated counters to zeros."""
        self.class_iou = {i: 0.0 for i in range(self.nc)}
        self.class_tp = {i: 0 for i in range(self.nc)}
        self.class_fp = {i: 0 for i in range(self.nc)}
        self.class_fn = {i: 0 for i in range(self.nc)}
        self.class_gt = {i: 0 for i in range(self.nc)}
        self.true_negative = 0
        self.total_negatives = 0
        self.false_negative = 0