from __future__ import annotations
import torch
import zipfile
import os
import math
from tqdm import tqdm
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou, LOGGER
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



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


# class AccuracyIoU:
#     """
#     A class for calculating IoU and accuracy per class for object detection tasks.

#     Attributes:
#     nc (int): Number of classes.
#     conf (float): Confidence threshold for detections.
#     iou_thres (float): IoU threshold for matching detections to ground truth.
#     class_iou (dict): Dictionary to store total IoU sums per class.
#     class_tp (dict): Dictionary to store true positive counts per class.
#     class_gt (dict): Dictionary to store ground truth counts per class.
#     """

#     def __init__(self, class_names, nc, conf=0.25, iou_thres=0.45):
#         """Initialize attributes for per-class IoU and accuracy calculation."""
#         self.nc = nc  # number of classes
#         self.class_names = class_names
#         self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
#         self.iou_thres = iou_thres  # IoU threshold
#         self.class_iou = {i: 0.0 for i in range(nc)}  # Total IoU per class
#         self.class_tp = {i: 0 for i in range(nc)}    # True positives per class
#         self.class_fp = {i: 0 for i in range(nc)}    # False positives per class
#         self.class_fn = {i: 0 for i in range(nc)}    # False negatives per class
#         self.class_gt = {i: 0 for i in range(nc)}    # Ground truth instances per class
#         self.tn_predicted_background = 0             # Total true negatives predicted(background)
#         self.fn_predicted_background = 0             # Total false negatives predicted(background)   

#     def process_batch(self, detections, gt_bboxes, gt_cls):
#         """
#         Process a batch to update IoU and accuracy metrics per class.

#         Args:
#             detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
#             gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
#             gt_cls (torch.Tensor[M]): Ground truth class labels.
#         """
        
#         if detections is None:
#             if gt_cls.shape[0] != 0:
#                 self.fn_predicted_background += 1  # No detections, objects present: FN
#             else:
#                 self.tn_predicted_background += 1  # No detections, no objects: TN
#         else:
#             # Filter detections by confidence
#             detections = detections[detections[:, 4] > self.conf]
#             if gt_cls.shape[0] == 0:
#                 if detections.shape[0] == 0:
#                     self.tn_predicted_background += 1  # No detections, no objects: TN
#                 else:
#                     # Detections, no objects: FP
#                     # Update class_fp for unmatched detections
#                     detection_classes = detections[:, 5].int().cpu().numpy()
#                     for dc in detection_classes:
#                         self.class_fp[dc] += 1
#             else:
#                 # Update ground truth counts per class
#                 gt_classes = gt_cls.int().cpu().numpy()
#                 for gc in gt_classes:
#                     self.class_gt[gc] += 1

#                 # Compute IoU between ground truth and detections
#                 iou = box_iou(gt_bboxes, detections[:, :4])  # [M, N] IoU matrix
#                 detection_classes = detections[:, 5].int().cpu().numpy()

#                 # Find matches between detections and ground truth
#                 x = torch.where(iou > self.iou_thres)
                
#                 if x[0].shape[0]:  # If there are matches
#                     matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
#                     if x[0].shape[0] > 1:
#                         # Sort by IoU in descending order and remove duplicates
#                         matches = matches[matches[:, 2].argsort()[::-1]]
#                         matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # Unique detection matches
#                         matches = matches[matches[:, 2].argsort()[::-1]]
#                         matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # Unique ground truth matches

#                     m0, m1, _ = matches.transpose().astype(int)  # m0: gt indices, m1: detection indices
#                     for i, gc in enumerate(gt_classes):
#                         j = m0 == i
#                         if sum(j) == 1:  # If this ground truth box is matched
#                             dc = detection_classes[m1[j][0]]  # Predicted class
#                             if dc == gc:  # Correct class prediction
#                                 iou_value = matches[j, 2][0]  # IoU for this match
#                                 self.class_iou[gc] += iou_value
#                                 self.class_tp[gc] += 1
#                             else:  # Incorrect class prediction
#                                 self.class_fp[dc] += 1
#                                 self.class_fn[gc] += 1
#                         else:  # False negative
#                             self.class_fn[gc] += 1
#                 else:  
#                     # updated Fp and FN for unmatched detections
#                     for dc in detection_classes:
#                         self.class_fp[dc] += 1
#                     for gc in gt_classes:
#                         self.class_fn[gc] += 1

#     def get_metrics(self):
#         """
#         Calculate and return IoU and accuracy per class.

#         Returns:
#             tuple: (iou_per_class, acc_per_class) where:
#                 - iou_per_class (dict): Average IoU for each class.
#                 - acc_per_class (dict): Accuracy (TP / GT) for each class.
#         """
#         iou_per_class = {}
#         acc_per_class = {}
#         acc_per_class["Background"] = self.tn_predicted_background / (self.tn_predicted_background + self.fn_predicted_background) if self.tn_predicted_background + self.fn_predicted_background > 0 else 0.0

#         total = {i: self.class_tp[i] + self.class_fn[i] + self.class_fp[i] for i in range(self.nc)}
        
#         for cls in range(self.nc):
#             iou_per_class[self.class_names[cls]] = (self.class_iou[cls] / self.class_tp[cls]) if self.class_tp[cls] > 0 else 0.0
#             acc_per_class[self.class_names[cls]] = (self.class_tp[cls] / total[cls]) if total[cls] > 0 else 0.0
            
#         return iou_per_class, acc_per_class

#     def print(self):
#         """Print IoU and accuracy for each class."""
#         iou_per_class, acc_per_class = self.get_metrics()
#         LOGGER.info("Per-class IoU and Accuracy:")
#         for key, value in iou_per_class.items():
#             LOGGER.info(f"{key}: IoU: {value:.3f} | Accuracy: {acc_per_class[key]:.3f}")
#         # print background class
#         LOGGER.info(f"Background: Accuracy: {acc_per_class['Background']:.3f}")
#         # reset the values
#         self.reset()
            
#     def print_avg(self):
#         """Print the average IoU and accuracy across all classes."""
#         iou_per_class, acc_per_class = self.get_metrics()
#         avg_iou = sum(iou_per_class.values()) / self.nc if sum(iou_per_class.values()) > 0.0 else 0.0
#         avg_acc = sum(acc_per_class.values()) / self.nc if sum(acc_per_class.values()) > 0.0 else 0.0
#         LOGGER.info(f"Average IoU : {avg_iou:.3f} | Average Accuracy: {avg_acc:.3f}")
        
#     def reset(self):
#         self.class_iou = {i: 0.0 for i in range(self.nc)}
#         self.class_tp = {i: 0 for i in range(self.nc)}
#         self.class_fp = {i: 0 for i in range(self.nc)}
#         self.class_fn = {i: 0 for i in range(self.nc)}
#         self.class_gt = {i: 0 for i in range(self.nc)}
#         self.tn_predicted_background = 0
#         self.fn_predicted_background = 0
        
        
class AccuracyIoU:
    """
    A class for calculating IoU and accuracy per class for object detection tasks,
    and for computing the ROC curve and AUROC for object presence detection.

    Attributes:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching detections to ground truth.
        class_iou (dict): Dictionary to store total IoU sums per class.
        class_tp (dict): Dictionary to store true positive counts per class.
        class_fp (dict): Dictionary to store false positive counts per class.
        class_fn (dict): Dictionary to store false negative counts per class.
        class_gt (dict): Dictionary to store ground truth counts per class.
        tn_predicted_background (int): Total true negatives for background.
        fn_predicted_background (int): Total false negatives for background.
        max_conf_per_image (list): List of maximum confidence scores per image.
        has_object_per_image (list): List indicating if each image has at least one object.
    """

    def __init__(self, class_names, nc, conf=0.25, iou_thres=0.45):
        """
        Initialize attributes for per-class IoU, accuracy, and ROC calculation.
        
        Args:
            class_names (list): List of class names.
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold. Defaults to 0.45.
        """
        self.nc = nc  # Number of classes
        self.class_names = class_names
        self.conf = 0.25 if conf in {None, 0.001} else conf  # Confidence threshold
        self.iou_thres = iou_thres  # IoU threshold
        self.class_iou = {i: 0.0 for i in range(nc)}  # Total IoU per class
        self.class_tp = {i: 0 for i in range(nc)}    # True positives per class
        self.class_fp = {i: 0 for i in range(nc)}    # False positives per class
        self.class_fn = {i: 0 for i in range(nc)}    # False negatives per class
        self.class_gt = {i: 0 for i in range(nc)}    # Ground truth instances per class
        self.tn_predicted_background = 0             # Total true negatives predicted (background)
        self.fn_predicted_background = 0             # Total false negatives predicted (background)
        # For ROC curve
        self.max_conf_per_image = []
        self.has_object_per_image = []

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU, accuracy, and ROC metrics.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        # Compute max confidence for ROC curve using raw detections
        if detections is not None and detections.shape[0] > 0:
            max_conf = detections[:, 4].max().item()
        else:
            max_conf = 0.0
        has_object = gt_cls.shape[0] > 0
        self.max_conf_per_image.append(max_conf)
        self.has_object_per_image.append(has_object)

        # Filter detections by confidence for other metrics
        if detections is not None:
            detections = detections[detections[:, 4] > self.conf]

        if detections is None or detections.shape[0] == 0:
            if gt_cls.shape[0] != 0:
                self.fn_predicted_background += 1  # No detections, objects present: FN
            else:
                self.tn_predicted_background += 1  # No detections, no objects: TN
        else:
            if gt_cls.shape[0] == 0:
                # Detections, no objects: FP
                detection_classes = detections[:, 5].int().cpu().numpy()
                for dc in detection_classes:
                    self.class_fp[dc] += 1
            else:
                # Update ground truth counts per class
                gt_classes = gt_cls.int().cpu().numpy()
                for gc in gt_classes:
                    self.class_gt[gc] += 1

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
                            else:  # Incorrect class prediction
                                self.class_fp[dc] += 1
                                self.class_fn[gc] += 1
                        else:  # False negative
                            self.class_fn[gc] += 1
                else:
                    # No matches: all detections are FP, all GT are FN
                    for dc in detection_classes:
                        self.class_fp[dc] += 1
                    for gc in gt_classes:
                        self.class_fn[gc] += 1

    def get_metrics(self):
        """
        Calculate and return IoU and accuracy per class.

        Returns:
            tuple: (iou_per_class, acc_per_class) where:
                - iou_per_class (dict): Average IoU for each class.
                - acc_per_class (dict): Accuracy (TP / (TP + FP + FN)) for each class.
        """
        iou_per_class = {}
        acc_per_class = {}
        acc_per_class["Background"] = self.tn_predicted_background / (self.tn_predicted_background + self.fn_predicted_background) if self.tn_predicted_background + self.fn_predicted_background > 0 else 0.0

        total = {i: self.class_tp[i] + self.class_fn[i] + self.class_fp[i] for i in range(self.nc)}

        for cls in range(self.nc):
            iou_per_class[self.class_names[cls]] = (self.class_iou[cls] / self.class_tp[cls]) if self.class_tp[cls] > 0 else 0.0
            acc_per_class[self.class_names[cls]] = (self.class_tp[cls] / total[cls]) if total[cls] > 0 else 0.0

        return iou_per_class, acc_per_class

    def print(self):
        """Print IoU and accuracy for each class."""
        iou_per_class, acc_per_class = self.get_metrics()
        print("Per-class IoU and Accuracy:")
        for key, value in iou_per_class.items():
            print(f"{key}: IoU: {value:.3f} | Accuracy: {acc_per_class[key]:.3f}")
        # Print background class
        print(f"Background: Accuracy: {acc_per_class['Background']:.3f}")

    def print_avg(self):
        """Print the average IoU and accuracy across all classes."""
        iou_per_class, acc_per_class = self.get_metrics()
        avg_iou = sum(iou_per_class.values()) / self.nc if sum(iou_per_class.values()) > 0.0 else 0.0
        avg_acc = sum(acc_per_class.values()) / self.nc if sum(acc_per_class.values()) > 0.0 else 0.0
        print(f"Average IoU: {avg_iou:.3f} | Average Accuracy: {avg_acc:.3f}")

    def reset(self):
        """Reset all metrics to zero."""
        self.class_iou = {i: 0.0 for i in range(self.nc)}
        self.class_tp = {i: 0 for i in range(self.nc)}
        self.class_fp = {i: 0 for i in range(self.nc)}
        self.class_fn = {i: 0 for i in range(self.nc)}
        self.class_gt = {i: 0 for i in range(self.nc)}
        self.tn_predicted_background = 0
        self.fn_predicted_background = 0
        self.max_conf_per_image = []
        self.has_object_per_image = []

    def compute_roc_curve(self):
        """
        Compute the ROC curve and AUROC for the binary task of object presence.

        Returns:
            tuple: (fpr, tpr, thresholds, auroc) where:
                - fpr (array): False Positive Rates.
                - tpr (array): True Positive Rates.
                - thresholds (array): Confidence thresholds.
                - auroc (float): Area Under the ROC Curve.
        """
        if not self.max_conf_per_image or not self.has_object_per_image:
            raise ValueError("No data collected for ROC computation.")
        y_true = self.has_object_per_image  # True labels: True if image has object
        y_score = self.max_conf_per_image   # Scores: max confidence per image
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)
        return fpr, tpr, thresholds, auroc

    def plot_roc_curve(self):
        """Plot the ROC curve and display the AUROC."""
        fpr, tpr, _, auroc = self.compute_roc_curve()
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Object Presence')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        # Reset the values
        self.reset()