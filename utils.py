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
from ultralytics.utils.metrics import box_iou, LOGGER, TryExcept, plt_settings, batch_probiou
from sklearn.metrics import roc_curve, auc
import warnings



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
            

class ConfusionMatrixwithROC:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks,
    with additional functionality to plot ROC curves and compute AUROC scores.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
        all_scores (list): List of confidence scores for all detections (for ROC).
        all_labels (list): List of binary labels (1 for TP, 0 for FP) for all detections (for ROC).
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """
        Initialize a ConfusionMatrix instance.

        Args:
            nc (int): Number of classes.
            conf (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth. Defaults to 0.45.
            task (str, optional): Type of task, either 'detect' or 'classify'. Defaults to "detect".
        """
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default conf is passed
        self.iou_thres = iou_thres
        self.all_scores = []  # Store confidence scores for ROC
        self.all_labels = []  # Store TP/FP labels for ROC

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix and ROC data for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes with (x1, y1, x2, y2, conf, class)
                                                    or (x1, y1, x2, y2, conf, class, angle) for OBB.
            gt_bboxes (Array[M, 4] | Array[M, 5]): Ground truth bounding boxes in xyxy or xyxyr format.
            gt_cls (Array[M]): Ground truth class labels.
        """
        if gt_cls.shape[0] == 0:  # No ground truths
            if detections is not None:
                detections_conf = detections[detections[:, 4] > self.conf]
                detection_classes = detections_conf[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # False positives
                # Collect ROC data for all detections
                if detections.shape[0] > 0:
                    for i in range(detections.shape[0]):
                        self.all_scores.append(detections[i, 4].item())
                        self.all_labels.append(0)  # FP, no matching ground truth
            return
        if detections is None:  # No detections
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # Background FN
            return

        # For confusion matrix (detections above confidence threshold)
        detections_conf = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections_conf[:, 5].int() if detections_conf.shape[0] > 0 else torch.tensor([], dtype=torch.int32)
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5
        iou_conf = (
            batch_probiou(gt_bboxes, torch.cat([detections_conf[:, :4], detections_conf[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections_conf[:, :4])
        )

        x_conf = torch.where(iou_conf > self.iou_thres)
        if x_conf[0].shape[0]:
            matches_conf = torch.cat((torch.stack(x_conf, 1), iou_conf[x_conf[0], x_conf[1]][:, None]), 1).cpu().numpy()
            if x_conf[0].shape[0] > 1:
                matches_conf = matches_conf[matches_conf[:, 2].argsort()[::-1]]  # Sort by IoU descending
                matches_conf = matches_conf[np.unique(matches_conf[:, 1], return_index=True)[1]]  # Unique detections
                matches_conf = matches_conf[matches_conf[:, 2].argsort()[::-1]]
                matches_conf = matches_conf[np.unique(matches_conf[:, 0], return_index=True)[1]]  # Unique ground truths
        else:
            matches_conf = np.zeros((0, 3))

        n_conf = matches_conf.shape[0] > 0
        m0_conf, m1_conf, _ = matches_conf.transpose().astype(int) if n_conf else ([], [], [])
        for i, gc in enumerate(gt_classes):
            j = m0_conf == i
            if n_conf and sum(j) == 1:
                self.matrix[detection_classes[m1_conf[j][0]], gc] += 1  # Correct prediction
            else:
                self.matrix[self.nc, gc] += 1  # Background FN

        for i, dc in enumerate(detection_classes):
            if not any(m1_conf == i):
                self.matrix[dc, self.nc] += 1  # Predicted background (FP)

        # For ROC data (all detections, regardless of confidence threshold)
        if detections.shape[0] > 0:
            iou_all = (
                batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
                if is_obb
                else box_iou(gt_bboxes, detections[:, :4])
            )
            x_all = torch.where(iou_all > self.iou_thres)
            if x_all[0].shape[0]:
                matches_all = torch.cat((torch.stack(x_all, 1), iou_all[x_all[0], x_all[1]][:, None]), 1).cpu().numpy()
                if x_all[0].shape[0] > 1:
                    matches_all = matches_all[matches_all[:, 2].argsort()[::-1]]
                    matches_all = matches_all[np.unique(matches_all[:, 1], return_index=True)[1]]
                    matches_all = matches_all[matches_all[:, 2].argsort()[::-1]]
                    matches_all = matches_all[np.unique(matches_all[:, 0], return_index=True)[1]]
            else:
                matches_all = np.zeros((0, 3))

            m1_all = matches_all[:, 1].astype(int) if matches_all.shape[0] > 0 else []
            for i in range(detections.shape[0]):
                self.all_scores.append(detections[i, 4].item())
                self.all_labels.append(1 if i in m1_all else 0)  # TP if matched, FP otherwise

    def matrix(self):
        """Return the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """
        Return true positives and false positives.

        Returns:
            (tuple): True positives and false positives.
        """
        tp = self.matrix.diagonal()  # True positives
        fp = self.matrix.sum(1) - tp  # False positives
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # Remove background class if detect

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): Callback to pass plot path when rendered.
        """
        import seaborn  # Lazy import

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)
        labels = (0 < nn < 99) and (nn == nc)
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    @TryExcept("WARNING ⚠️ ROC plot failure")
    @plt_settings()
    def plot_roc(self, save_dir="", on_plot=None):
        """
        Plot the ROC curve and display the AUROC score.

        Args:
            save_dir (str): Directory to save the ROC plot.
            on_plot (func): Callback to pass plot path when rendered.
        """
        if not self.all_scores or not self.all_labels or len(set(self.all_labels)) < 2:
            LOGGER.warning("Insufficient data to plot ROC curve: need both TP and FP samples.")
            return

        fpr, tpr, _ = roc_curve(self.all_labels, self.all_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUROC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        plot_fname = Path(save_dir) / "roc_curve.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

        LOGGER.info(f"AUROC Score: {roc_auc:.4f}")

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))