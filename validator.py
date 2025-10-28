"""validator.py

DetectionValidator: a drop-in validator for object detection tasks.

This module extends Ultralytics' BaseValidator to provide detection-specific preprocessing,
postprocessing (NMS), metric accumulation (DetMetrics) and optional JSON export for COCO/LVIS
evaluation. It also integrates a small per-class IoU/accuracy tracker (AccuracyIoU) used for
additional per-class reporting.
"""
import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import  DetMetrics, box_iou, ConfusionMatrix
from ultralytics.utils.plotting import output_to_target, plot_images
from utils import AccuracyIoU


class DetectionValidator(BaseValidator):
    """
    DetectionValidator

    A validator implementation tailored for object detection models (YOLO-style).
    It extends ultralytics.engine.validator.BaseValidator and provides:

    - Batch preprocessing (to device, normalization)
    - Postprocessing (NMS + scaling predictions back to original image space)
    - Metric accumulation (mAP via DetMetrics, confusion matrix, custom AccuracyIoU)
    - Optional COCO/LVIS JSON export and official evaluation

    The public API follows the BaseValidator pattern so it can be used as a drop-in validator.
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize validator internals.

        Args:
            dataloader: optional dataloader to use
            save_dir: directory to save results/plots/json
            pbar: optional progress bar handler
            args: argument namespace (must include options like conf, iou, plots, val, save_json, etc.)
            _callbacks: internal callbacks
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        # Counters filled during / after validation
        self.nt_per_class = None   # number of targets per class across dataset
        self.nt_per_image = None   # number of images per class across dataset

        # Dataset type flags (COCO or LVIS) used for JSON export/evaluation
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None  # mapping for COCO 80->91 id conversion

        # Force detection task for this validator
        self.args.task = "detect"

        # Metric helpers
        self.metrics = DetMetrics(save_dir=self.save_dir)
        # IoU thresholds used for mAP@0.5:0.95 (10 steps)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

        # buffer for autolabelling (labels appended to predictions when save_hybrid is enabled)
        self.lb = []
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Prepare a batch for model input and optionally build hybrid label buffer.

        Operations performed:
        - Move image tensor and annotation tensors to the validator device.
        - Normalize image pixels to [0,1] and cast to float/half depending on args.half.
        - If save_hybrid: scale bboxes to pixel coords and build per-image hybrid labels.

        Args:
            batch (dict): batch returned by dataloader with keys including:
                'img', 'batch_idx', 'cls', 'bboxes', etc.

        Returns:
            Modified batch dict with tensors on self.device and normalized images.
        """
        # Move and normalize image tensor
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255

        # Move annotation tensors to device (batch_idx, cls, bboxes)
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        # If hybrid saving is on, prepare a per-image label list containing GT for autolabelling.
        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            # scale normalized xywh bboxes to pixel coords for each image
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize metric helpers and dataset flags before validation starts.

        This method:
        - Detects whether the provided dataset is COCO or LVIS (to enable official eval).
        - Sets up class id mapping for COCO (80->91) when needed.
        - Initializes DetMetrics, ConfusionMatrix and AccuracyIoU helpers.
        - Prepares accumulators used during update_metrics.
        """
        val = self.data.get(self.args.split, "")  # validation path (string expected)
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        # map class ids appropriately for COCO if needed, otherwise simple 1..nc mapping
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))

        # enable saving JSON for final val on COCO/LVIS when not training
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training

        # model metadata
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)

        # configure metric helpers
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.accuracy_iou = AccuracyIoU(class_names=self.names, nc=self.nc, conf=self.args.conf)

        # runtime accumulators
        self.seen = 0
        self.jdict = []  # will hold COCO/LVIS json entries
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted header string used for per-class metric printing."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-Maximum Suppression and other prediction-level options.

        Uses ultralytics.ops.non_max_suppression with options controlled by args and
        internal flags (e.g., end2end, rotated, agnostic NMS).
        """
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
        """Extract and convert ground-truth annotations for a single image in a batch.

        Steps:
        - Select rows from batch annotation tensors that correspond to image index `si`.
        - Convert normalized xywh boxes to xyxy in model input pixel space, then scale them
          back to the original image space using ops.scale_boxes.

        Args:
            si (int): image index within the batch
            batch (dict): dataloader batch containing annotation fields

        Returns:
            dict: {'cls', 'bbox', 'ori_shape', 'imgsz', 'ratio_pad'}
                - cls: 1D tensor of class indices for this image
                - bbox: xyxy boxes in original image pixel coordinates
                - ori_shape: original image shape (height, width)
                - imgsz: current model input size (h, w)
                - ratio_pad: ratio and pad used for letterbox scaling
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            # Convert normalized xywh -> normalized xyxy then scale to model input pixels
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            # Convert boxes from model input space back to original native image space
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Scale predicted boxes back to the original image space.

        pred: tensor of detections (N, >=6) in model input coordinates
        pbatch: dict returned by _prepare_batch with 'imgsz', 'ori_shape', 'ratio_pad'

        Returns:
            predn: cloned tensor with first 4 cols scaled to native image coordinates
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Update accumulated metrics using per-image predictions and ground truth.

        For each prediction set in preds (one per image in batch):
        - Prepare stat container
        - Convert GT and predictions to native image space
        - Compute TP matrix across IoU thresholds with class-awareness
        - Update DetMetrics/ConfusionMatrix/AccuracyIoU helpers
        - Optionally save detection outputs (JSON/TXT)
        """
        for si, pred in enumerate(preds):
            # count this image as seen
            self.seen += 1
            npr = len(pred)

            # stat placeholders for this image (tp matrix sized by npr x niou)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )

            # prepare ground truth for this image
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)

            # store target class info for metrics aggregation later
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            # if no predictions, append empty stats and optionally update confusion matrix with only GT
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        # confusion matrix expects detections=None when no preds present
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # If evaluating in single-class mode, force predicted class index to 0
            if self.args.single_cls:
                pred[:, 5] = 0

            # scale predictions to native image coordinates
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # compute TP matrix for predictions vs GT (class-aware, across IoU thresholds)
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # update custom per-class IoU/accuracy tracker
                self.accuracy_iou.process_batch(predn, bbox, cls)

            # update confusion matrix for plotting if requested
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            # append per-image stats to global accumulators
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # optionally persist predictions
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
        """Finalize aggregated metric objects prior to result retrieval.

        Typically copies over confusion matrix and speed info into the metrics container.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Aggregate per-image stats into arrays and run DetMetrics processing.

        Returns:
            results_dict produced by DetMetrics (contains mAP, precision, recall, etc.)
        """
        # concatenate lists of per-image tensors into single arrays
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy

        # compute counts per class and per image for reporting
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)

        # target_img is only used for counting, remove before metrics.processing
        stats.pop("target_img", None)
        if len(stats):
            # process metrics (this populates self.metrics.results_dict)
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def print_results(self):
        """Log overall and per-class metrics to the configured LOGGER.

        Also computes a per-class F1 score for convenience and then delegates
        a more detailed per-class IoU/accuracy table to AccuracyIoU.
        """
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class when verbose and not training
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            scores_dict = {}
            for i, c in enumerate(self.metrics.ap_class_index):
                # retrieve precision and recall from per-class results
                precision, recall = self.metrics.class_result(i)[0:2]
                # safe f1 computation with tiny epsilon
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
                scores_dict[self.names[c]] = {
                    "Images": int(self.nt_per_image[c]),
                    "Instances": int(self.nt_per_class[c]),
                    "Precision": precision,
                    "Recall": recall,
                    "mAP50": self.metrics.class_result(i)[2],
                    "mAP50-95": self.metrics.class_result(i)[3],
                    "F1-Score": f1_score,
                }
            # delegate pretty printing of per-class IoU/accuracy stats
            self.accuracy_iou.print(scores_dict=scores_dict)
        # Optionally create confusion matrix plots (normalized and unnormalized)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Compute true-positive matrix for a single image.

        Args:
            detections (torch.Tensor): detections (N, >=6), first 4 cols are xyxy box coords.
            gt_bboxes (torch.Tensor): ground-truth boxes (M, 4) in native image coords.
            gt_cls (torch.Tensor): ground-truth class indices (M,).

        Returns:
            torch.Tensor: boolean matrix of shape (N, niou) indicating which detections are TP at each IoU.
        """
        # compute IoU matrix between each GT and each detection: shape (M, N)
        iou = box_iou(gt_bboxes, detections[:, :4])
        # match_predictions (from BaseValidator) handles class-aware assignment and IoU thresholds
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO dataset for evaluation.

        Args:
            img_path (str): dataset path or image-folder path
            mode (str): 'train' or 'val' to select augmentation pipeline
            batch (int or None): batch size for rectangular datasets (rect)

        Returns:
            Dataset instance created by build_yolo_dataset
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Create a DataLoader for validation.

        Uses build_dataset and build_dataloader from ultralytics.data and ensures no shuffling.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Save a visualization of ground-truth labels for a validation batch.

        Args:
            batch: dataloader batch
            ni: batch index used for filename
        """
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
        """Save a visualization of predictions for a validation batch.

        Uses output_to_target to convert model preds to plotting format.
        """
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save predictions for a single image to a YOLO-style txt file.

        The Results helper expects an image array, so we create a dummy mask with shape.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Append prediction entries to internal jdict in COCO-compatible format.

        Each entry contains:
            - image_id (derived from filename stem when numeric)
            - category_id (mapped via self.class_map)
            - bbox: [x,y,w,h] top-left origin, floats rounded to 3 decimals
            - score: confidence rounded to 5 decimals
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        # convert xyxy -> xywh and transform center->top-left
        box = ops.xyxy2xywh(predn[:, :4])  # xywh (center)
        box[:, :2] -= box[:, 2:] / 2  # center to top-left
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
        """Run official COCO / LVIS evaluation on saved predictions.json when applicable.

        This function:
        - Ensures the prediction and annotation files exist
        - Runs pycocotools COCOeval or LVIS LVISEval depending on dataset
        - Updates the provided stats dict with official mAP values
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # official eval requires files + dependencies
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
                # restrict evaluation to images in the dataset
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # print LVIS-specific results
                # update mAP50-95 and mAP50 in the stats dictionary based on evaluation output
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats