from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils import LOGGER, SimpleClass
from ultralytics.utils.metrics import *

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
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
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
    

# class DetMetrics(SimpleClass):
#     """
#     Utility class for computing detection metrics such as precision, recall, mean average precision (mAP),
#     and accuracy of an object detection model.

#     Args:
#         save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
#         plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
#         names (dict of str): A dict of strings that represents the names of the classes. Defaults to {}.

#     Attributes:
#         save_dir (Path): A path to the directory where the output plots will be saved.
#         plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
#         names (dict of str): A dict of strings that represents the names of the classes.
#         box (Metric): An instance of the Metric class for storing the results of the detection metrics.
#         speed (dict): A dictionary for storing the execution time of different parts of the detection process.
#         task (str): The task type, e.g. 'detect'.
#     """

#     def __init__(self, save_dir=Path("."), plot=False, names={}) -> None:
#         """Initialize a DetMetrics instance with a save directory, plot flag, and class names."""
#         self.save_dir = save_dir
#         self.plot = plot
#         self.names = names
#         self.box = Metric()
#         self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
#         self.task = "detect"

#     def process(self, tp, conf, pred_cls, target_cls, on_plot=None):
#         """
#         Process predicted results for object detection and update metrics.

#         Args:
#             tp (np.array): True positive flags for detections.
#             conf (np.array): Confidence scores for detections.
#             pred_cls (np.array): Predicted class indices.
#             target_cls (np.array): Ground truth class indices.
#             on_plot (optional): Callback for plotting, if any.
#         """
#         # Compute standard metrics: precision, recall, mAP50, mAP50-95, etc.
#         results = ap_per_class(tp, conf, pred_cls, target_cls,
#                                plot=self.plot, save_dir=self.save_dir,
#                                names=self.names, on_plot=on_plot)[2:]
#         # Compute "accuracy" as the fraction of detections that are true positives.
#         # This is a simple definition: accuracy = sum(tp) / total number of predictions.
#         if tp.size > 0:
#             accuracy = float(tp.sum() / tp.size)
#         else:
#             accuracy = 0.0
#         # Append accuracy to the list of computed results.
#         results.append(accuracy)
#         self.box.nc = len(self.names)
#         self.box.update(results)

#     @property
#     def keys(self):
#         """Returns a list of keys for accessing specific metrics."""
#         return [
#             "metrics/precision(B)", 
#             "metrics/recall(B)", 
#             "metrics/mAP50(B)", 
#             "metrics/mAP50-95(B)",
#             "metrics/accuracy(B)"
#         ]

#     def mean_results(self):
#         """Calculate mean of detected objects & return precision, recall, mAP50, mAP50-95, and accuracy."""
#         return self.box.mean_results()

#     def class_result(self, i):
#         """Return the result of evaluating the performance for a specific class."""
#         return self.box.class_result(i)

#     @property
#     def maps(self):
#         """Returns mean Average Precision (mAP) scores per class."""
#         return self.box.maps

#     @property
#     def fitness(self):
#         """Returns the fitness score computed from detection metrics."""
#         return self.box.fitness()

#     @property
#     def ap_class_index(self):
#         """Returns the average precision index per class."""
#         return self.box.ap_class_index

#     @property
#     def results_dict(self):
#         """Returns dictionary of computed performance metrics and statistics."""
#         return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

#     @property
#     def curves(self):
#         """Returns a list of curves for accessing specific metrics curves."""
#         return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

#     @property
#     def curves_results(self):
#         """Returns dictionary of computed curves results."""
#         return self.box.curves_results
    
# import numpy as np
# import torch
# from pathlib import Path
# # Assume compute_ap, smooth, plot_pr_curve, plot_mc_curve are defined elsewhere in the codebase
# from ultralytics.utils.ops import box_iou  # if needed elsewhere

# @plt_settings()
# def plot_accuracy_curve(px, py, ap, save_dir=Path("accuracy.png"), names={}, on_plot=None):
#     """Plots a precision-recall curve."""
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
#     py = np.stack(py, axis=1)

#     if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#         for i, y in enumerate(py.T):
#             ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
#     else:
#         ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

#     ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     ax.set_title("Precision-Recall Curve")
#     fig.savefig(save_dir, dpi=250)
#     plt.close(fig)
#     if on_plot:
#         on_plot(save_dir)



# def ap_per_class(
#     tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
# ):
#     """
#     Computes the average precision per class for object detection evaluation.

#     Args:
#         tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
#         conf (np.ndarray): Array of confidence scores of the detections.
#         pred_cls (np.ndarray): Array of predicted classes of the detections.
#         target_cls (np.ndarray): Array of true classes of the detections.
#         plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
#         on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
#         save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
#         names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
#         prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

#     Returns:
#         tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
#         fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
#         p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
#         r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
#         f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
#         ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
#         unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
#         p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
#         r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
#         f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
#         x (np.ndarray): X-axis values for the curves. Shape: (1000,).
#         prec_values (np.ndarray): Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
#     """
#     # Sort by objectness
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

#     # Find unique classes
#     unique_classes, nt = np.unique(target_cls, return_counts=True)
#     nc = unique_classes.shape[0]  # number of classes, number of detections

#     # Create Precision-Recall curve and compute AP for each class
#     x, prec_values = np.linspace(0, 1, 1000), []

#     # Average precision, precision and recall curves
#     ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         n_l = nt[ci]  # number of labels
#         n_p = i.sum()  # number of predictions
#         if n_p == 0 or n_l == 0:
#             continue

#         # Accumulate FPs and TPs
#         fpc = (1 - tp[i]).cumsum(0)
#         tpc = tp[i].cumsum(0)

#         # Compute Accuracy
#         acc = tpc / (tpc + fpc + eps)  # accuracy curve
        
#         # Recall
#         recall = tpc / (n_l + eps)  # recall curve
#         r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

#         # Precision
#         precision = tpc / (tpc + fpc)  # precision curve
#         p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

#         # AP from recall-precision curve
#         for j in range(tp.shape[1]):
#             ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
#             if j == 0:
#                 prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

#     prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

#     # Compute F1 (harmonic mean of precision and recall)
#     f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
#     names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
#     names = dict(enumerate(names))  # to dict
#     if plot:
#         plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
#         plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
#         plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
#         plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

#     i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
#     p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
#     tp = (r * nt).round()  # true positives
#     fp = (tp / (p + eps) - tp).round()  # false positives
#     return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
    


# class Metric(SimpleClass):
#     """
#     Class for computing evaluation metrics for YOLOv8 model.

#     Attributes:
#         p (list): Precision for each class. Shape: (nc,).
#         r (list): Recall for each class. Shape: (nc,).
#         f1 (list): F1 score for each class. Shape: (nc,).
#         all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
#         ap_class_index (list): Index of class for each AP score. Shape: (nc,).
#         nc (int): Number of classes.

#     Methods:
#         ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
#         ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
#         mp(): Mean precision of all classes. Returns: Float.
#         mr(): Mean recall of all classes. Returns: Float.
#         map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
#         map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
#         map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
#         mean_results(): Mean of results, returns mp, mr, map50, map.
#         class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
#         maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
#         fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
#         update(results): Update metric attributes with new evaluation results.
#     """

#     def __init__(self) -> None:
#         """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
#         self.p = []  # (nc, )
#         self.r = []  # (nc, )
#         self.f1 = []  # (nc, )
#         self.all_ap = []  # (nc, 10)
#         self.ap_class_index = []  # (nc, )
#         self.nc = 0

#     @property
#     def ap50(self):
#         """
#         Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

#         Returns:
#             (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
#         """
#         return self.all_ap[:, 0] if len(self.all_ap) else []

#     @property
#     def ap(self):
#         """
#         Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

#         Returns:
#             (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
#         """
#         return self.all_ap.mean(1) if len(self.all_ap) else []

#     @property
#     def mp(self):
#         """
#         Returns the Mean Precision of all classes.

#         Returns:
#             (float): The mean precision of all classes.
#         """
#         return self.p.mean() if len(self.p) else 0.0

#     @property
#     def mr(self):
#         """
#         Returns the Mean Recall of all classes.

#         Returns:
#             (float): The mean recall of all classes.
#         """
#         return self.r.mean() if len(self.r) else 0.0

#     @property
#     def map50(self):
#         """
#         Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

#         Returns:
#             (float): The mAP at an IoU threshold of 0.5.
#         """
#         return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

#     @property
#     def map75(self):
#         """
#         Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

#         Returns:
#             (float): The mAP at an IoU threshold of 0.75.
#         """
#         return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

#     @property
#     def map(self):
#         """
#         Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

#         Returns:
#             (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
#         """
#         return self.all_ap.mean() if len(self.all_ap) else 0.0

#     def mean_results(self):
#         """Mean of results, return mp, mr, map50, map."""
#         return [self.mp, self.mr, self.map50, self.map]

#     def class_result(self, i):
#         """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
#         return self.p[i], self.r[i], self.ap50[i], self.ap[i]

#     @property
#     def maps(self):
#         """MAP of each class."""
#         maps = np.zeros(self.nc) + self.map
#         for i, c in enumerate(self.ap_class_index):
#             maps[c] = self.ap[i]
#         return maps

#     def fitness(self):
#         """Model fitness as a weighted combination of metrics."""
#         w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
#         return (np.array(self.mean_results()) * w).sum()

#     def update(self, results):
#         """
#         Updates the evaluation metrics of the model with a new set of results.

#         Args:
#             results (tuple): A tuple containing the following evaluation metrics:
#                 - p (list): Precision for each class. Shape: (nc,).
#                 - r (list): Recall for each class. Shape: (nc,).
#                 - f1 (list): F1 score for each class. Shape: (nc,).
#                 - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
#                 - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

#         Side Effects:
#             Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
#             on the values provided in the `results` tuple.
#         """
#         (
#             self.p,
#             self.r,
#             self.f1,
#             self.all_ap,
#             self.ap_class_index,
#             self.p_curve,
#             self.r_curve,
#             self.f1_curve,
#             self.px,
#             self.prec_values,
#         ) = results

#     @property
#     def curves(self):
#         """Returns a list of curves for accessing specific metrics curves."""
#         return []

#     @property
#     def curves_results(self):
#         """Returns a list of curves for accessing specific metrics curves."""
#         return [
#             [self.px, self.prec_values, "Recall", "Precision"],
#             [self.px, self.f1_curve, "Confidence", "F1"],
#             [self.px, self.p_curve, "Confidence", "Precision"],
#             [self.px, self.r_curve, "Confidence", "Recall"],
#         ]


# class v8DetectionLoss:
#     """Criterion class for computing training losses."""

#     def __init__(self, model, tal_topk=10):  # model must be de-paralleled
#         """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
#         device = next(model.parameters()).device  # get model device
#         h = model.args  # hyperparameters

#         m = model.model[-1]  # Detect() module
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")
#         self.hyp = h
#         self.stride = m.stride  # model strides
#         self.nc = m.nc  # number of classes
#         self.no = m.nc + m.reg_max * 4
#         self.reg_max = m.reg_max
#         self.device = device

#         self.use_dfl = m.reg_max > 1

#         self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = BboxLoss(m.reg_max).to(device)
#         self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""
#         nl, ne = targets.shape
#         if nl == 0:
#             out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 if n := matches.sum():
#                     out[j, :n] = targets[matches, 1:]
#             out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
#         return out

#     def bbox_decode(self, anchor_points, pred_dist):
#         """Decode predicted object bounding box coordinates from anchor points and distribution."""
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
#         return dist2bbox(pred_dist, anchor_points, xywh=False)

#     def __call__(self, preds, batch):
#         """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
#         loss = torch.zeros(3, device=self.device)  # box, cls, dfl
#         feats = preds[1] if isinstance(preds, tuple) else preds
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1
#         )

#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         batch_size = pred_scores.shape[0]
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # Targets
#         targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
#         targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
#         # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
#         # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt,
#         )

#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             loss[0], loss[2] = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )

#         loss[0] *= self.hyp.box  # box gain
#         loss[1] *= self.hyp.cls  # cls gain
#         loss[2] *= self.hyp.dfl  # dfl gain

#         return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
