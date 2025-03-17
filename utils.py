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
    A simplified class for calculating IoU and accuracy for object detection tasks.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        total_iou (float): Running sum of IoU values.
        total_tp (int): Running sum of true positives.
        total_gt (int): Running sum of ground truth instances.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initialize attributes for IoU and accuracy calculation."""
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold
        self.total_iou = 0.0  # Running total IoU
        self.total_tp = 0  # Running total true positives
        self.total_gt = 0  # Running total ground truth instances

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU and accuracy metrics.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        # Handle empty ground truth
        if gt_cls.shape[0] == 0:
            return

        # Handle no detections
        if detections is None:
            self.total_gt += gt_cls.shape[0]  # All ground truth are false negatives
            return

        # Filter detections by confidence
        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            self.total_gt += gt_cls.shape[0]  # No detections, all ground truth are FN
            return

        # Compute IoU between ground truth and detections
        iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N] IoU matrix
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()

        # Find matches between detections and ground truth
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:  # If there are matches
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Sort by IoU in descending order and remove duplicates
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detection matches
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique ground truth matches

            m0, m1, iou_values = matches.transpose().astype(int)
            matched_iou = matches[:, 2]  # IoU values for matched pairs
            self.total_iou += matched_iou.sum()  # Sum IoU for matched detections
            self.total_tp += len(matched_iou)  # Number of true positives
            self.total_gt += gt_bboxes.shape[0]  # Total ground truth instances

            # Log unmatched ground truth as FN (implicitly handled by total_gt - total_tp)
        else:
            self.total_gt += gt_bboxes.shape[0]  # No matches, all ground truth are FN

    def get_metrics(self):
        """
        Calculate and return the average IoU and accuracy across all processed batches.

        Returns:
            tuple: (mean_iou, accuracy) where:
                - mean_iou (float): Average IoU across all matched detections.
                - accuracy (float): TP / (TP + FN) across all ground truth instances.
        """
        mean_iou = self.total_iou / self.total_tp if self.total_tp > 0 else 0.0
        accuracy = self.total_tp / self.total_gt if self.total_gt > 0 else 0.0
        return mean_iou, accuracy

    def print(self):
        """Print the calculated IoU and accuracy."""
        mean_iou, accuracy = self.get_metrics()
        LOGGER.info(f"                      AVG IoU: {mean_iou:.3f} | AVG Accuracy: {accuracy:.3f}")
        

class AccuracyIoUPerClass:
    """
    A class for calculating IoU and accuracy per class for object detection tasks.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        class_iou (dict): Dictionary to store total IoU sums per class.
        class_tp (dict): Dictionary to store true positive counts per class.
        class_gt (dict): Dictionary to store ground truth counts per class.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initialize attributes for per-class IoU and accuracy calculation."""
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold
        self.class_iou = {i: 0.0 for i in range(nc)}  # Total IoU per class
        self.class_tp = {i: 0 for i in range(nc)}    # True positives per class
        self.class_gt = {i: 0 for i in range(nc)}    # Ground truth instances per class

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU and accuracy metrics per class.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        # Handle empty ground truth
        if gt_cls.shape[0] == 0:
            return

        # Update ground truth counts per class
        gt_classes = gt_cls.int().cpu().numpy()
        for gc in gt_classes:
            self.class_gt[gc] += 1

        # Handle no detections
        if detections is None:
            return

        # Filter detections by confidence
        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            return

        # Compute IoU between ground truth and detections
        iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N] IoU matrix
        detection_classes = detections[:, 5].int().cpu().numpy()

        # Find matches between detections and ground truth
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:  # If there are matches
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                # Sort by IoU in descending order and remove duplicates
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detection matches
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique ground truth matches

            m0, m1, _ = matches.transpose().astype(int)  # m0: gt indices, m1: detection indices
            for i, gc in enumerate(gt_classes):
                j = m0 == i
                if sum(j) == 1:  # If this ground truth box is matched
                    dc = detection_classes[m1[j][0]]  # Predicted class
                    if dc == gc:  # Correct class prediction
                        iou_value = matches[j, 2][0]  # IoU for this match
                        self.class_iou[gc] += iou_value
                        self.class_tp[gc] += 1

    def get_metrics(self):
        """
        Calculate and return IoU and accuracy per class.

        Returns:
            tuple: (iou_per_class, acc_per_class) where:
                - iou_per_class (dict): Average IoU for each class.
                - acc_per_class (dict): Accuracy (TP / GT) for each class.
        """
        iou_per_class = {}
        acc_per_class = {}
        for cls in range(self.nc):
            iou_per_class[cls] = self.class_iou[cls] / self.class_tp[cls] if self.class_tp[cls] > 0 else 0.0
            acc_per_class[cls] = self.class_tp[cls] / self.class_gt[cls] if self.class_gt[cls] > 0 else 0.0
        return iou_per_class, acc_per_class

    def print(self, names=None):
        """Print IoU and accuracy for each class."""
        iou_per_class, acc_per_class = self.get_metrics()
        if names is None:
            names = {i: f"Class_{i}" for i in range(self.nc)}
        LOGGER.info("Per-class IoU and Accuracy:")
        for cls in range(self.nc):
            LOGGER.info(f"{names[cls]:<20} | IoU: {iou_per_class[cls]:.3f} | Accuracy: {acc_per_class[cls]:.3f}")
            
            
class ROCPerClass:
    """
    A class for calculating TPR, FPR, AUROC per class, and average AUROC, TPR, FPR for object detection tasks.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        class_scores (dict): Dictionary to store confidence scores per class (for ROC computation).
        class_labels (dict): Dictionary to store binary labels per class (1 for TP, 0 for FP).
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Initialize attributes for per-class ROC calculation."""
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold
        self.class_scores = {i: [] for i in range(nc)}  # Confidence scores per class
        self.class_labels = {i: [] for i in range(nc)}  # Binary labels (1=TP, 0=FP) per class
        self.mean_tpr = 0.0  # Mean TPR at optimal threshold
        self.mean_fpr = 0.0  # Mean FPR at optimal threshold
        self.mean_auroc = 0.0  # Mean AUROC

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update confidence scores and labels for ROC calculation per class.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        if gt_cls.shape[0] == 0:  # No ground truth
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int().cpu().numpy()
                detection_scores = detections[:, 4].cpu().numpy()
                for dc, score in zip(detection_classes, detection_scores):
                    self.class_scores[dc].append(score)
                    self.class_labels[dc].append(0)  # False positives
            return

        if detections is None:  # No detections
            return

        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            return

        gt_classes = gt_cls.int().cpu().numpy()
        detection_classes = detections[:, 5].int().cpu().numpy()
        detection_scores = detections[:, 4].cpu().numpy()

        iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N] IoU matrix

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:  # Matches exist
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # Sort by IoU
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detections
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique ground truth

            m0, m1 = matches[:, 0].astype(int), matches[:, 1].astype(int)
            for i, gc in enumerate(gt_classes):
                j = m0 == i
                if sum(j) == 1:  # Ground truth matched
                    dc = detection_classes[m1[j][0]]
                    score = detection_scores[m1[j][0]]
                    if dc == gc:  # True positive
                        self.class_scores[gc].append(score)
                        self.class_labels[gc].append(1)

        matched_detections = set(m1) if x[0].shape[0] else set()
        for i, (dc, score) in enumerate(zip(detection_classes, detection_scores)):
            if i not in matched_detections:
                self.class_scores[dc].append(score)
                self.class_labels[dc].append(0)  # False positive

    def get_roc_metrics(self):
        """
        Calculate TPR, FPR, and AUROC per class, and average AUROC, TPR, FPR across all classes.

        Returns:
            tuple: (tpr_per_class, fpr_per_class, auroc_per_class, tpr_opt_per_class, fpr_opt_per_class, 
                    mean_auroc, mean_tpr, mean_fpr)
                - tpr_per_class (dict): TPR values per class.
                - fpr_per_class (dict): FPR values per class.
                - auroc_per_class (dict): AUROC score per class.
                - tpr_opt_per_class (dict): TPR at optimal threshold per class.
                - fpr_opt_per_class (dict): FPR at optimal threshold per class.
                - mean_auroc (float): Average AUROC across classes with valid data.
                - mean_tpr (float): Average TPR at optimal threshold across classes with valid data.
                - mean_fpr (float): Average FPR at optimal threshold across classes with valid data.
        """
        tpr_per_class = {}
        fpr_per_class = {}
        auroc_per_class = {}
        tpr_opt_per_class = {}
        fpr_opt_per_class = {}
        valid_aurocs = []
        valid_tpr = []
        valid_fpr = []

        for cls in range(self.nc):
            scores = np.array(self.class_scores[cls])
            labels = np.array(self.class_labels[cls])
            if len(scores) > 0 and len(np.unique(labels)) > 1:  # Valid ROC data
                fpr, tpr, thresholds = roc_curve(labels, scores)
                auroc = auc(fpr, tpr)
                tpr_per_class[cls] = tpr
                fpr_per_class[cls] = fpr
                auroc_per_class[cls] = auroc
                idx = np.argmax(tpr - fpr)  # Optimal threshold (Youden's J)
                tpr_opt_per_class[cls] = tpr[idx]
                fpr_opt_per_class[cls] = fpr[idx]
                valid_aurocs.append(auroc)
                valid_tpr.append(tpr[idx])
                valid_fpr.append(fpr[idx])
            else:
                tpr_per_class[cls] = np.array([0.0, 1.0])
                fpr_per_class[cls] = np.array([0.0, 1.0])
                auroc_per_class[cls] = 0.0
                tpr_opt_per_class[cls] = 0.0
                fpr_opt_per_class[cls] = 0.0

        mean_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
        mean_tpr = np.mean(valid_tpr) if valid_tpr else 0.0
        mean_fpr = np.mean(valid_fpr) if valid_fpr else 0.0
        return tpr_per_class, fpr_per_class, auroc_per_class, tpr_opt_per_class, fpr_opt_per_class, mean_auroc, mean_tpr, mean_fpr
    
    def print_avg(self):
        """Print average TPR, FPR, and AUROC across all classes."""
        LOGGER.info(f"{'Average':<20} | TPR: {self.mean_tpr:.3f} | FPR: {self.mean_fpr:.3f} | AUROC: {self.mean_auroc:.3f}")

    def print(self, names=None):
        """Print TPR, FPR, AUROC per class and their averages at the optimal threshold."""
        tpr_per_class, fpr_per_class, auroc_per_class, tpr_opt_per_class, fpr_opt_per_class, mean_auroc, mean_tpr, mean_fpr = self.get_roc_metrics()
        if names is None:
            names = {i: f"Class_{i}" for i in range(self.nc)}
        self.mean_tpr = mean_tpr
        self.mean_fpr = mean_fpr
        self.mean_auroc = mean_auroc
        LOGGER.info("Per-class TPR, FPR, AUROC (at optimal threshold):")
        for cls in range(self.nc):
            LOGGER.info(f"{names[cls]:<20} | TPR: {tpr_opt_per_class[cls]:.3f} | FPR: {fpr_opt_per_class[cls]:.3f} | AUROC: {auroc_per_class[cls]:.3f}")

    @TryExcept("WARNING ⚠️ ROC Curve plot failure")
    @plt_settings()
    def plot(self, save_dir="", names=None, on_plot=None):
        """
        Plot ROC curves for each class and save them to a file.

        Args:
            save_dir (str): Directory to save the plot.
            names (dict, optional): Class index to name mapping.
            on_plot (callable, optional): Callback for plot path.
        """
        tpr_per_class, fpr_per_class, auroc_per_class, _, _, mean_auroc, _, _ = self.get_roc_metrics()
        if names is None:
            names = {i: f"Class_{i}" for i in range(self.nc)}

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random (0.500)")
        
        for cls in range(self.nc):
            if auroc_per_class[cls] > 0.0:
                ax.plot(fpr_per_class[cls], tpr_per_class[cls], lw=2,
                        label=f"{names[cls]} (AUROC = {auroc_per_class[cls]:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves (Mean AUROC = {mean_auroc:.3f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(True)

        plot_fname = Path(save_dir) / "roc_curves.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)

        if on_plot:
            on_plot(plot_fname)