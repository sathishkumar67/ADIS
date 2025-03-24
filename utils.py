from __future__ import annotations
import torch
import zipfile
import os
import math
from tqdm import tqdm
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import *


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
    A class for calculating IoU and accuracy per class for object detection tasks.

    Attributes:
    nc (int): Number of classes.
    conf (float): Confidence threshold for detections.
    iou_thres (float): IoU threshold for matching detections to ground truth.
    class_iou (dict): Dictionary to store total IoU sums per class.
    class_tp (dict): Dictionary to store true positive counts per class.
    class_gt (dict): Dictionary to store ground truth counts per class.
    """

    def __init__(self, class_names, nc, conf=0.25, iou_thres=0.45):
        """Initialize attributes for per-class IoU and accuracy calculation."""
        self.nc = nc  # number of classes
        self.class_names = class_names
        self.conf = 0.25 if conf in {None, 0.001} else conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold
        self.class_iou = {i: 0.0 for i in range(nc)}  # Total IoU per class
        self.class_tp = {i: 0 for i in range(nc)}    # True positives per class
        self.class_fp = {i: 0 for i in range(nc)}    # False positives per class
        self.class_fn = {i: 0 for i in range(nc)}    # False negatives per class
        self.class_gt = {i: 0 for i in range(nc)}    # Ground truth instances per class
        self.tn_predicted_background = 0             # Total true negatives predicted(background)
        self.fn_predicted_background = 0             # Total false negatives predicted(background)   

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Process a batch to update IoU and accuracy metrics per class.

        Args:
            detections (torch.Tensor[N, 6]): Detected boxes with (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor[M, 4]): Ground truth boxes in xyxy format.
            gt_cls (torch.Tensor[M]): Ground truth class labels.
        """
        
        if detections is None:
            if gt_cls.shape[0] != 0:
                self.fn_predicted_background += 1  # No detections, objects present: FN
            else:
                self.tn_predicted_background += 1  # No detections, no objects: TN
        else:
            # Filter detections by confidence
            detections = detections[detections[:, 4] > self.conf]
            if gt_cls.shape[0] == 0:
                if detections.shape[0] == 0:
                    self.tn_predicted_background += 1  # No detections, no objects: TN
                else:
                    # Detections, no objects: FP
                    # Update class_fp for unmatched detections
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
                    # updated Fp and FN for unmatched detections
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
                - acc_per_class (dict): Accuracy (TP / GT) for each class.
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
        LOGGER.info("Per-class IoU and Accuracy:")
        for key, value in iou_per_class.items():
            LOGGER.info(f"{key}: IoU: {value:.3f} | Accuracy: {acc_per_class[key]:.3f}")
        # print background class
        LOGGER.info(f"Background: Accuracy: {acc_per_class['Background']:.3f}")
        # reset the values
        self.reset()
            
    def print_avg(self):
        """Print the average IoU and accuracy across all classes."""
        iou_per_class, acc_per_class = self.get_metrics()
        avg_iou = sum(iou_per_class.values()) / self.nc if sum(iou_per_class.values()) > 0.0 else 0.0
        avg_acc = sum(acc_per_class.values()) / self.nc if sum(acc_per_class.values()) > 0.0 else 0.0
        LOGGER.info(f"Average IoU : {avg_iou:.3f} | Average Accuracy: {avg_acc:.3f}")
        
    def reset(self):
        self.class_iou = {i: 0.0 for i in range(self.nc)}
        self.class_tp = {i: 0 for i in range(self.nc)}
        self.class_fp = {i: 0 for i in range(self.nc)}
        self.class_fn = {i: 0 for i in range(self.nc)}
        self.class_gt = {i: 0 for i in range(self.nc)}
        self.tn_predicted_background = 0
        self.fn_predicted_background = 0
        
        
class AUROC:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, class_names, iou_thres=0.45, task="detect"):
        """
        Initialize a ConfusionMatrix instance.

        Args:
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold for detections.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth.
            task (str, optional): Type of task, either 'detect' or 'classify'.
        """
        self.task = task
        self.nc = nc  # number of classes
        self.class_names = class_names
        self.iou_thres = iou_thres # IoU threshold
        self.matrix = np.zeros((nc + 1, nc + 1))     # confusion matrix 
        self.overall_results = {}
        self.detections = []
        self.gt_bboxes = []
        self.gt_cls = []
        

    def process_batch(self, detections, gt_bboxes, gt_cls, conf):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # predicted background

    def calculate_metrics(self, matrix):
        # Number of classes (e.g., 12 for a 12x12 confusion matrix)
        N = len(matrix)
        
        # Compute the sum of each row (actual class totals)
        row_sums = [sum(matrix[i]) for i in range(N)]
        
        # Compute the sum of each column (predicted class totals)
        col_sums = [sum(matrix[i][j] for i in range(N)) for j in range(N)]
        
        # Compute the total sum of all elements in the confusion matrix
        total_sum = sum(row_sums)
        
        # Initialize the results dictionary
        results = {}
        
        # Calculate metrics for each class
        for i in range(N):
            # True Positives: correctly predicted instances for class i
            TP = matrix[i][i]
            
            # False Positives: instances predicted as class i but belong to other classes
            FP = col_sums[i] - TP
            
            # False Negatives: instances of class i predicted as other classes
            FN = row_sums[i] - TP
            
            # True Negatives: instances not of class i and not predicted as class i
            TN = total_sum - row_sums[i] - col_sums[i] + TP
            
            # Store the metrics in the dictionary with class i as the key
            results[i] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
        
        # Return the dictionary containing metrics for all classes
        return results
    
    def aggregate_batches(self, detections, gt_bboxes, gt_cls):
        self.detections.append(detections)
        self.gt_bboxes.append(gt_bboxes)
        self.gt_cls.append(gt_cls)
    
    def roc_curve_calculation(self):
        thresholds = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]
        for current_confidence in thresholds:
            self.matrix = np.zeros((self.nc + 1, self.nc + 1))     # confusion matrix
            for detections, gt_bboxes, gt_cls in zip(self.detections, self.gt_bboxes, self.gt_cls):
                self.process_batch(detections, gt_bboxes, gt_cls, current_confidence)
            results = self.calculate_metrics(self.matrix)
            self.overall_results[current_confidence] = results
            
    def plot_roc_curve(self):
        self.roc_curve_calculation()

    
    
    
    
    
    
    
    
    
    
    # @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    # @plt_settings()
    # def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
    #     """
    #     Plot the confusion matrix using seaborn and save it to a file.

    #     Args:
    #         normalize (bool): Whether to normalize the confusion matrix.
    #         save_dir (str): Directory where the plot will be saved.
    #         names (tuple): Names of classes, used as labels on the plot.
    #         on_plot (func): An optional callback to pass plots path and data when they are rendered.
    #     """
    #     import seaborn  # scope for faster 'import ultralytics'

    #     array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
    #     array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    #     nc, nn = self.nc, len(names)  # number of classes, names
    #     seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    #     labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    #     ticklabels = (list(names) + ["background"]) if labels else "auto"
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
    #         seaborn.heatmap(
    #             array,
    #             ax=ax,
    #             annot=nc < 30,
    #             annot_kws={"size": 8},
    #             cmap="Blues",
    #             fmt=".2f" if normalize else ".0f",
    #             square=True,
    #             vmin=0.0,
    #             xticklabels=ticklabels,
    #             yticklabels=ticklabels,
    #         ).set_facecolor((1, 1, 1))
    #     title = "Confusion Matrix" + " Normalized" * normalize
    #     ax.set_xlabel("True")
    #     ax.set_ylabel("Predicted")
    #     ax.set_title(title)
    #     plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
    #     fig.savefig(plot_fname, dpi=250)
    #     plt.close(fig)
    #     if on_plot:
    #         on_plot(plot_fname)