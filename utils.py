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
from ultralytics.utils.metrics import box_iou

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


class ROCMetrics:
    """
    A class for calculating True Positive Rate (TPR), False Positive Rate (FPR), and Area Under the ROC Curve (AUROC)
    for object detection tasks, modeled after the ConfusionMatrix class in Ultralytics.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for determining true positives.
        data (dict): Dictionary storing confidence scores and labels (1 for TP, 0 for FP) per class.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, save_dir=".", names={}):
        """
        Initialize the ROCMetrics class.

        Args:
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold for matching detections to ground truths. Defaults to 0.45.
        """
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.data = {c: [] for c in range(nc)}  # Stores (confidence, label) pairs for each class
        self.save_dir = Path(save_dir)
        self.names = names

    def process(self, tp, conf, pred_cls, target_cls, on_plot=None):
        """
        Process detection results to collect data for ROC computation.

        Args:
            tp (np.ndarray): Binary array indicating true positives, shape (num_dets,).
            conf (np.ndarray): Confidence scores, shape (num_dets,).
            pred_cls (np.ndarray): Predicted classes, shape (num_dets,).
        """
        # Ensure inputs are NumPy arrays and convert to appropriate types
        tp = np.asarray(tp, dtype=int)
        conf = np.asarray(conf, dtype=float)
        pred_cls = np.asarray(pred_cls, dtype=int)

        # Filter detections by confidence threshold
        mask = conf > self.conf
        tp, conf, pred_cls = tp[mask], conf[mask], pred_cls[mask]

        # Collect confidence scores and labels (1 for TP, 0 for FP) for each class
        for t, c, cls in zip(tp, conf, pred_cls):
            self.data[cls].append((c, t))

    def compute_roc(self, c):
        """
        Compute the ROC curve and AUROC for a specific class.

        Args:
            c (int): Class index.

        Returns:
            tuple: (fpr, tpr, auroc)
                - fpr (np.ndarray): False Positive Rates.
                - tpr (np.ndarray): True Positive Rates.
                - auroc (float): Area Under the ROC Curve.
        """
        if not self.data[c]:
            # Return default values if no data for the class
            return np.array([0, 1]), np.array([0, 1]), 0.0

        # Convert list of (confidence, label) to NumPy array
        conf_labels = np.array(self.data[c], dtype=[('conf', float), ('label', int)])
        
        # Sort by confidence in descending order
        sorted_idx = np.argsort(-conf_labels['conf'])
        conf_labels = conf_labels[sorted_idx]

        # Cumulative sums for TPs and FPs
        tp_cumsum = np.cumsum(conf_labels['label'])
        fp_cumsum = np.cumsum(1 - conf_labels['label'])

        total_p = float(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0.0
        total_n = float(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0.0

        if total_p == 0 or total_n == 0:
            # Return default values if no positives or negatives
            return np.array([0, 1]), np.array([0, 1]), 0.0

        # Calculate TPR and FPR
        tpr = tp_cumsum / total_p
        fpr = fp_cumsum / total_n

        # Add endpoints (0,0) and (1,1) for complete ROC curve
        fpr = np.concatenate([[0], fpr, [1]])
        tpr = np.concatenate([[0], tpr, [1]])

        # Compute AUROC using the trapezoidal rule
        auroc = np.trapz(tpr, fpr)

        return fpr, tpr, auroc

    def get_auroc(self):
        """
        Get the AUROC for each class.

        Returns:
            list: List of AUROC values for each class.
        """
        auroc_per_class = []
        for c in range(self.nc):
            _, _, auroc = self.compute_roc(c)
            auroc_per_class.append(auroc)
        return auroc_per_class

    @property
    def mean_auroc(self):
        """
        Compute the mean AUROC across all classes.

        Returns:
            float: Mean AUROC value.
        """
        auroc_values = self.get_auroc()
        return np.mean(auroc_values) if auroc_values else 0.0

    def plot(self, on_plot=None):
        """
        Plot ROC curves for each class and save the figure.

        Args:
            save_dir (Path, optional): Directory to save the plot. Defaults to current directory.
            names (dict, optional): Dictionary mapping class indices to names. Defaults to empty dict.
            on_plot (callable, optional): Callback function to handle plot file path. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
        for c in range(self.nc):
            fpr, tpr, auroc = self.compute_roc(c)
            label = self.names.get(c, f"Class {c}")
            ax.plot(fpr, tpr, label=f"{label} AUROC={auroc:.3f}")
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        
        # Save the plot
        plot_fname = self.save_dir / "roc_curve.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """
        Print AUROC values for each class.

        Args:
            names (dict, optional): Dictionary mapping class indices to names. Defaults to empty dict.
        """
        auroc_per_class = self.get_auroc()
        for c, auroc in enumerate(auroc_per_class):
            label = self.names.get(c, f"Class {c}")
            LOGGER.info(f"{label}: AUROC = {auroc:.4f}")
        LOGGER.info(f"Mean AUROC: {self.mean_auroc:.4f}")