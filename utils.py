from __future__ import annotations
import torch
import zipfile
import os
import math
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou, LOGGER, TryExcept, plt_settings
from sklearn.metrics import roc_curve, auc


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") # Check here if error occurs
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def unzip_file(zip_path: str, target_dir: str) -> None:
    """
    Unzips the specified zip file into the target directory with a progress bar.

    Parameters:
    - zip_path (str): The path to the zip file to be unzipped.
    - target_dir (str): The directory where the unzipped files will be stored.
    """
    # Ensure the target directory exists; create it if it doesn't
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files and directories in the zip
        info_list = zip_ref.infolist()
        # Calculate total uncompressed size for the progress bar
        total_size = sum(zinfo.file_size for zinfo in info_list)
        
        # Create progress bar with total size in bytes, scaled to KB/MB/GB as needed
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Unzipping") as pbar:
            for zinfo in info_list:
                # Extract the file or directory
                zip_ref.extract(zinfo, target_dir)
                # Update progress bar by the file's uncompressed size
                pbar.update(zinfo.file_size)


class AccuracyIoU:
    """
    A class for calculating IoU and classification accuracy for object detection tasks.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        total_iou (float): Running sum of IoU values for matched detections.
        total_matched (int): Total number of matched detections.
        total_correct (int): Total number of matches with correct class predictions.
    """

    def __init__(self, nc, conf=0.50, iou_thres=0.70):
        """
        Initialize attributes for IoU and accuracy calculation.

        Args:
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold for filtering detections. Defaults to 0.50.
            iou_thres (float, optional): IoU threshold for matching. Defaults to 0.70.
        """
        self.nc = nc
        # Set a lower default confidence if conf is None or very small
        self.conf = 0.25 if conf in {None, 0.001} else conf
        self.iou_thres = iou_thres
        self.total_iou = 0.0
        self.total_matched = 0
        self.total_correct = 0

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU and accuracy metrics based on predicted class.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        import torch
        import numpy as np

        # Handle edge cases: empty ground truth or detections
        if gt_cls.shape[0] == 0 or detections is None or detections.shape[0] == 0:
            return

        # Validate detections input
        if not isinstance(detections, torch.Tensor):
            raise ValueError(f"Detections must be a torch.Tensor, got {type(detections)}")
        if detections.shape[1] != 6:
            raise ValueError(f"Detections must have shape [N, 6], got {detections.shape}")

        # Check for None values in confidence scores
        conf_scores = detections[:, 4]
        if any(x is None for x in conf_scores.tolist()):
            raise ValueError("Confidence scores in detections contain None values")

        # Filter detections by confidence threshold
        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            return

        # Compute IoU between ground truth and detections
        iou = box_iou(gt_bboxes, detections[:, :4])
        if iou is None:
            raise ValueError("IoU calculation returned None")

        # Convert class tensors to integers
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()

        # Find matches where IoU exceeds threshold
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            # Create match array with IoU values
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Sort by IoU in descending order
                matches = matches[matches[:, 2].argsort()[::-1]]
                # Keep only the highest IoU match per detection
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                # Keep only the highest IoU match per ground truth
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            # Extract indices and IoU values
            m0, m1, iou_values = matches[:, 0].astype(int), matches[:, 1].astype(int), matches[:, 2]
            gt_classes_np = gt_classes.cpu().numpy().astype(int)
            detection_classes_np = detection_classes.cpu().numpy().astype(int)

            # Check class correctness for matches
            class_matches = gt_classes_np[m0] == detection_classes_np[m1]

            # Update running totals
            self.total_matched += len(m0)
            self.total_correct += np.sum(class_matches)
            self.total_iou += iou_values.sum()

    def get_metrics(self):
        """
        Calculate and return the average IoU and classification accuracy.

        Returns:
            tuple: (mean_iou, classification_accuracy)
                - mean_iou (float): Average IoU across all matched detections.
                - classification_accuracy (float): Proportion of matched detections with correct class.
        """
        mean_iou = self.total_iou / self.total_matched if self.total_matched > 0 else 0.0
        classification_accuracy = self.total_correct / self.total_matched if self.total_matched > 0 else 0.0
        return mean_iou, classification_accuracy

    def print(self):
        """
        Print the calculated IoU and classification accuracy.
        """
        mean_iou, classification_accuracy = self.get_metrics()
        LOGGER.info(f"                      AVG IoU: {mean_iou:.3f} | Classification Accuracy: {classification_accuracy:.3f}")
            

# Assuming box_iou is available from ultralytics.utils.metrics or a similar module
def box_iou(box1, box2):
    """Compute IoU between two sets of boxes."""
    # Placeholder; replace with actual implementation if not provided
    pass


class AccuracyIoUPerClass:
    """
    A class for calculating IoU and classification accuracy per class for object detection tasks,
    where accuracy is based on the predicted class among matched detections.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        class_iou (dict): Sum of IoU for correct predictions per class.
        class_predicted (dict): Number of matched detections predicted as each class.
        class_correct (dict): Number of correct predictions per class (predicted and ground truth class match).
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        Initialize attributes for per-class IoU and accuracy calculation.

        Args:
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold. Defaults to 0.45.
        """
        self.nc = nc
        self.conf = 0.25 if conf in {None, 0.001} else conf  # Ensure conf is a valid float
        self.iou_thres = iou_thres
        self.class_iou = {i: 0.0 for i in range(nc)}       # Sum of IoU for correct predictions
        self.class_predicted = {i: 0 for i in range(nc)}   # Matched detections predicted as class i
        self.class_correct = {i: 0 for i in range(nc)}     # Correct predictions for class i

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU and accuracy metrics per class based on predicted class.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        import torch
        import numpy as np

        # Handle empty ground truth or detections
        if gt_cls.shape[0] == 0 or detections is None or detections.shape[0] == 0:
            return

        # Validate detections is a tensor with the correct shape
        if not isinstance(detections, torch.Tensor):
            raise ValueError(f"Detections must be a torch.Tensor, got {type(detections)}")
        if detections.shape[1] != 6:
            raise ValueError(f"Detections must have shape [N, 6], got {detections.shape}")

        # Check for None values in the confidence column (index 4)
        conf_scores = detections[:, 4]
        if any(x is None for x in conf_scores.tolist()):
            raise ValueError("Confidence scores in detections contain None values")

        # Filter detections by confidence
        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            return

        # Compute IoU between ground truth and detections
        iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N] IoU matrix
        if iou is None:
            raise ValueError("IoU calculation returned None")

        gt_classes = gt_cls.int().cpu().numpy()
        detection_classes = detections[:, 5].int().cpu().numpy()

        # Find matches between detections and ground truth
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:  # If there are matches
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Sort by IoU and ensure unique matches
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detection matches
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique ground truth matches

            # Process each match
            m0, m1 = matches[:, 0].astype(int), matches[:, 1].astype