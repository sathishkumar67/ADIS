import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        # self.roc_metrics = ROCMetrics(nc=len(self.names), save_dir=self.save_dir, names=self.names)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.accuracy_iou = AccuracyIoU(nc=self.nc, conf=self.args.conf)
        self.accuracy_iou_per_class = AccuracyIoUPerClass(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # calculate IoU and accuracy
                self.accuracy_iou.process_batch(predn, bbox, cls)
                self.accuracy_iou_per_class.process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
            # self.roc_metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        self.accuracy_iou.print() # print IoU and accuracy average
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )
            self.accuracy_iou_per_class.print(names=self.names)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
    

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

# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt

# class ROCMetrics:
#     """
#     A class for calculating True Positive Rate (TPR), False Positive Rate (FPR), and Area Under the ROC Curve (AUROC)
#     for object detection tasks, modeled after the ConfusionMatrix class in Ultralytics.

#     Attributes:
#         nc (int): Number of classes.
#         conf (float): Confidence threshold for detections.
#         iou_thres (float): IoU threshold for determining true positives.
#         data (dict): Dictionary storing confidence scores and labels (1 for TP, 0 for FP) per class.
#     """

#     def __init__(self, nc, conf=0.25, iou_thres=0.45, save_dir=".", names={}):
#         """
#         Initialize the ROCMetrics class.

#         Args:
#             nc (int): Number of classes.
#             conf (float, optional): Confidence threshold for detections. Defaults to 0.25.
#             iou_thres (float, optional): IoU threshold for matching detections to ground truths. Defaults to 0.45.
#         """
#         self.nc = nc
#         self.conf = conf
#         self.iou_thres = iou_thres
#         self.data = {c: [] for c in range(nc)}  # Stores (confidence, label) pairs for each class
#         self.save_dir = Path(save_dir)
#         self.names = names

#     def process(self, tp, conf, pred_cls, target_cls, on_plot=None):
#         """
#         Process detection results to collect data for ROC computation.

#         Args:
#             tp (np.ndarray): Binary array indicating true positives, shape (num_dets,).
#             conf (np.ndarray): Confidence scores, shape (num_dets,).
#             pred_cls (np.ndarray): Predicted classes, shape (num_dets,).
#         """
#         # Ensure inputs are NumPy arrays and convert to appropriate types
#         tp = np.asarray(tp, dtype=int)
#         conf = np.asarray(conf, dtype=float)
#         pred_cls = np.asarray(pred_cls, dtype=int)

#         # Filter detections by confidence threshold
#         mask = conf > self.conf
#         tp, conf, pred_cls = tp[mask], conf[mask], pred_cls[mask]

#         # Collect confidence scores and labels (1 for TP, 0 for FP) for each class
#         for t, c, cls in zip(tp, conf, pred_cls):
#             self.data[cls].append((c, t))

#     def compute_roc(self, c):
#         """
#         Compute the ROC curve and AUROC for a specific class.

#         Args:
#             c (int): Class index.

#         Returns:
#             tuple: (fpr, tpr, auroc)
#                 - fpr (np.ndarray): False Positive Rates.
#                 - tpr (np.ndarray): True Positive Rates.
#                 - auroc (float): Area Under the ROC Curve.
#         """
#         if not self.data[c]:
#             # Return default values if no data for the class
#             return np.array([0, 1]), np.array([0, 1]), 0.0

#         # Convert list of (confidence, label) to NumPy array
#         conf_labels = np.array(self.data[c], dtype=[('conf', float), ('label', int)])
        
#         # Sort by confidence in descending order
#         sorted_idx = np.argsort(-conf_labels['conf'])
#         conf_labels = conf_labels[sorted_idx]

#         # Cumulative sums for TPs and FPs
#         tp_cumsum = np.cumsum(conf_labels['label'])
#         fp_cumsum = np.cumsum(1 - conf_labels['label'])

#         total_p = float(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0.0
#         total_n = float(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0.0

#         if total_p == 0 or total_n == 0:
#             # Return default values if no positives or negatives
#             return np.array([0, 1]), np.array([0, 1]), 0.0

#         # Calculate TPR and FPR
#         tpr = tp_cumsum / total_p
#         fpr = fp_cumsum / total_n

#         # Add endpoints (0,0) and (1,1) for complete ROC curve
#         fpr = np.concatenate([[0], fpr, [1]])
#         tpr = np.concatenate([[0], tpr, [1]])

#         # Compute AUROC using the trapezoidal rule
#         auroc = np.trapz(tpr, fpr)

#         return fpr, tpr, auroc

#     def get_auroc(self):
#         """
#         Get the AUROC for each class.

#         Returns:
#             list: List of AUROC values for each class.
#         """
#         auroc_per_class = []
#         for c in range(self.nc):
#             _, _, auroc = self.compute_roc(c)
#             auroc_per_class.append(auroc)
#         return auroc_per_class

#     @property
#     def mean_auroc(self):
#         """
#         Compute the mean AUROC across all classes.

#         Returns:
#             float: Mean AUROC value.
#         """
#         auroc_values = self.get_auroc()
#         return np.mean(auroc_values) if auroc_values else 0.0

#     def plot(self, on_plot=None):
#         """
#         Plot ROC curves for each class and save the figure.

#         Args:
#             save_dir (Path, optional): Directory to save the plot. Defaults to current directory.
#             names (dict, optional): Dictionary mapping class indices to names. Defaults to empty dict.
#             on_plot (callable, optional): Callback function to handle plot file path. Defaults to None.
#         """
#         fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
#         for c in range(self.nc):
#             fpr, tpr, auroc = self.compute_roc(c)
#             label = self.names.get(c, f"Class {c}")
#             ax.plot(fpr, tpr, label=f"{label} AUROC={auroc:.3f}")
        
#         # Plot diagonal line (random classifier)
#         ax.plot([0, 1], [0, 1], "k--", label="Random")
#         ax.set_xlabel("False Positive Rate")
#         ax.set_ylabel("True Positive Rate")
#         ax.set_title("ROC Curve")
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        
#         # Save the plot
#         plot_fname = self.save_dir / "roc_curve.png"
#         fig.savefig(plot_fname, dpi=250)
#         plt.close(fig)
        
#         if on_plot:
#             on_plot(plot_fname)

#     def print(self):
#         """
#         Print AUROC values for each class.

#         Args:
#             names (dict, optional): Dictionary mapping class indices to names. Defaults to empty dict.
#         """
#         from ultralytics.utils import LOGGER
#         auroc_per_class = self.get_auroc()
#         for c, auroc in enumerate(auroc_per_class):
#             label = self.names.get(c, f"Class {c}")
#             LOGGER.info(f"{label}: AUROC = {auroc:.4f}")
#         LOGGER.info(f"Mean AUROC: {self.mean_auroc:.4f}")
        
        
#         save_dir, names