"""trainer.py

Provides training utilities and the `DetectionTrainer` class for running model training loops.

This module encapsulates the entire training workflow for object detection tasks, including:
- Dataset and dataloader setup.
- Optimizer and learning rate scheduler configuration.
- Distributed Data Parallel (DDP) for multi-GPU training.
- Automatic Mixed Precision (AMP) for improved performance.
- Checkpointing for saving model progress and resuming training.
- Logging of metrics and training progress.
- Integration with validation loops for performance evaluation.

The `DetectionTrainer` class is the core component, designed to work seamlessly with
`DetectionModel` and `DetectionValidator` to provide a complete training solution.
"""

import gc
import math
import random
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils import TQDM
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    init_seeds,
    one_cycle,
    select_device,
    torch_distributed_zero_first,
    unset_deterministic,
    de_parallel
)
from validator import DetectionValidator
from detector import DetectionModel


class DetectionTrainer:
    """
    Manages the training process for a YOLO object detection model.

    This class orchestrates the entire training pipeline, including data loading, model setup,
    optimization, and evaluation. It handles both single-GPU and multi-GPU (DDP) training.

    Attributes:
        args (Namespace): A namespace object containing all training configurations.
        validator (DetectionValidator): An instance of the validator for evaluating the model.
        model (nn.Module): The YOLO model being trained.
        callbacks (defaultdict): A dictionary of callback functions triggered on specific events.
        save_dir (Path): The directory where results and checkpoints are saved.
        wdir (Path): The directory specifically for saving model weights.
        last (Path): The path to the last saved checkpoint (`last.pt`).
        best (Path): The path to the best performing checkpoint (`best.pt`).
        save_period (int): The interval (in epochs) at which to save checkpoints.
        batch_size (int): The training batch size.
        epochs (int): The total number of training epochs.
        start_epoch (int): The epoch to start training from, used for resuming.
        device (torch.device): The device (CPU or GPU) used for training.
        amp (bool): A flag to enable Automatic Mixed Precision (AMP).
        scaler (amp.GradScaler): The gradient scaler for AMP.
        data (dict): A dictionary containing dataset information (paths, names, etc.).
        trainset (Dataset): The training dataset.
        testset (Dataset): The validation dataset.
        ema (ModelEMA): An instance for applying Exponential Moving Average to model weights.
        resume (bool): A flag indicating if training is being resumed from a checkpoint.
        lf (callable): The learning rate scheduler function.
        scheduler (LRScheduler): The learning rate scheduler.
        best_fitness (float): The best fitness score (e.g., mAP) achieved so far.
        fitness (float): The current fitness score.
        loss (torch.Tensor): The current total loss.
        tloss (torch.Tensor): The smoothed total loss over the current epoch.
        loss_names (list): A list of names for the different loss components.
        csv (Path): The path to the results CSV file for logging metrics.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, bohb=False, custom_callbacks=None):
        """
        Initializes the DetectionTrainer.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): A dictionary of configuration settings to override the defaults.
            _callbacks (dict, optional): A dictionary of custom callbacks.
            bohb (bool): Flag to enable BOHB (Bayesian Optimization Hyperband) specific logic.
            custom_callbacks (dict, optional): Custom callbacks, particularly for BOHB.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)  # Check for 'resume' argument and adjust settings accordingly
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        self.score = 0.0  # Score for Bayesian optimization hyperband
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Setup directories for saving results and weights
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # Update name for logging
        self.wdir = self.save_dir / "weights"  # Weights directory
        if RANK in {-1, 0}:  # Only on the main process
            self.wdir.mkdir(parents=True, exist_ok=True)  # Create weights directory
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # Save run arguments
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"
        self.save_period = self.args.save_period

        # Initialize training parameters
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        self.start_epoch = 0
        self.bohb = bohb
        if self.bohb and custom_callbacks:
            self.custom_callbacks = custom_callbacks
        
        if RANK == -1:  # Non-DDP mode
            print_args(vars(self.args))

        # Adjust settings for CPU/MPS devices
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # Disable multi-worker data loading for CPU/MPS

        # Initialize model and dataset
        self.model = check_model_file_from_stem(self.args.model)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Initialize optimization utilities
        self.lf = None
        self.scheduler = None

        # Initialize epoch-level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB integration
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Adds a callback function to a specific event."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Sets a new callback for an event, replacing any existing ones."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Executes all callbacks registered for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """
        Starts the training process.

        Determines the world size for distributed training and launches the training
        process either as a subprocess (for DDP) or directly.
        """
        # Determine the number of GPUs to use (world_size)
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:
            world_size = 0
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        # Launch DDP training as a subprocess if world_size > 1
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'"
                )
                self.args.batch = 16

            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            # Run training directly for single-GPU or CPU
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Configures the learning rate scheduler."""
        if self.args.cos_lr:
            # Cosine learning rate scheduler
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            # Linear learning rate scheduler
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes the Distributed Data Parallel (DDP) environment."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # Prevent timeouts
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3-hour timeout
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """
        Prepares all necessary components for the training loop.

        This includes setting up the model, freezing layers, configuring AMP,
        creating dataloaders, and initializing the optimizer and scheduler.
        """
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze specified layers
        freeze_list = self.args.freeze if isinstance(self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + [".dfl"]  # Always freeze DFL
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.info(f"WARNING ⚠️ Unfreezing layer '{k}' that was previously frozen.")
                v.requires_grad = True

        # Configure Automatic Mixed Precision (AMP)
        self.amp = torch.tensor(self.args.amp, device=self.device)
        if self.amp and RANK in {-1, 0}:
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Finalize image size and batch size
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        if self.batch_size < 1 and RANK == -1:  # AutoBatch for single-GPU
            self.args.batch = self.batch_size = self.auto_batch()

        # Create dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode="val")
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Setup optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        self.optimizer = self.build_optimizer(model=self.model, name=self.args.optimizer, lr=self.args.lr0, momentum=self.args.momentum, decay=weight_decay)
        
        # Setup scheduler and early stopping
        self._setup_scheduler()
        self.stopper = EarlyStopping(patience=self.args.patience)
        self.stop = False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Executes the main training loop."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # Number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # Warmup iterations
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs} epochs..."
        )

        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while not self.stop:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.scheduler.step()
            self.model.train()

            if RANK != -1: # DDP
                self.train_loader.sampler.set_epoch(epoch)
            
            # Close mosaic augmentation in the last few epochs
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            self.tloss = None # Reset smoothed loss
            pbar = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(pbar, total=nb) # Progress bar

            # Batch loop
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch # Current iteration

                # Warmup phase
                if ni <= nw:
                    self.accumulate = max(1, int(np.interp(ni, [0, nw], [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(ni, [0, nw], [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, [0, nw], [self.args.warmup_momentum, self.args.momentum])

                # Forward pass
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1: # DDP
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items

                # Backward pass
                self.scaler.scale(self.loss).backward()

                # Optimizer step
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    
                    # Check for timed stopping
                    if self.args.time and (time.time() - self.train_time_start) > (self.args.time * 3600):
                        self.stop = True
                        if RANK != -1: # Sync stop signal in DDP
                            dist.broadcast_object_list([self.stop], 0)
                        if self.stop:
                            break

                # Logging
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + len(self.tloss)))
                        % (f"{epoch + 1}/{self.epochs}", f"{self._get_memory():.3g}G", *self.tloss, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                
                self.run_callbacks("on_train_batch_end")
            
            # End of epoch
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")

            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                    self.score = round(self.metrics.get("metrics/mAP50-95(B)", 0.0), 6)
                    # For BOHB
                    if self.bohb and self.custom_callbacks:
                        self.custom_callbacks["on_train_epoch_end"](self.score, epoch + 1)
                
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                
                # Save model checkpoints
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Sync early stopping signal across DDP processes
            if RANK != -1:
                dist.broadcast_object_list([self.stop], 0)
                if self.stop:
                    break
            
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()
            epoch += 1

        # End of training
        if RANK in {-1, 0}:
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.train_time_start) / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self):
        """Automatically determines the largest batch size that fits in memory."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # Estimate based on the maximum number of objects in an image
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # x4 for mosaic
        return super().auto_batch(max_num_obj)
    
    def _get_memory(self):
        """Returns the current GPU memory usage in GB."""
        if self.device.type == "mps":
            return torch.mps.driver_allocated_memory() / (2**30)
        elif self.device.type == "cuda":
            return torch.cuda.memory_reserved() / (2**30)
        return 0

    def _clear_memory(self):
        """Frees up unused memory on the accelerator."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def read_results_csv(self):
        """Reads the results.csv file into a dictionary."""
        import pandas as pd
        return pd.read_csv(self.csv).to_dict(orient="list") if self.csv.exists() else {}

    def save_model(self):
        """Saves the model checkpoint."""
        import io
        buffer = io.BytesIO()
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": None,  # Model is saved in EMA
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save(ckpt, buffer)
        serialized_ckpt = buffer.getvalue()

        # Save last, best, and periodic checkpoints
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            (self.wdir / f"epoch{self.epoch + 1}.pt").write_bytes(serialized_ckpt)

    def get_dataset(self):
        """Loads and prepares the dataset from the provided data file."""
        try:
            # Check dataset based on task type
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            else: # detect, segment, pose, obb
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        
        self.data = data
        if self.args.single_cls:
            # Override for single-class training
            LOGGER.info("Overriding class names for single-class training.")
            self.data["names"] = {0: "item"}
            self.data["nc"] = 1
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Initializes the model, loading weights if specified."""
        if isinstance(self.model, torch.nn.Module):
            return  # Model is already loaded

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        
        # Create the model instance
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def optimizer_step(self):
        """Performs a single optimization step, including gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Applies preprocessing to a batch of data before the forward pass."""
        # Move to device and normalize
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        
        # Multi-scale training
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride)) // self.stride * self.stride
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def validate(self):
        """Runs the validation loop and returns metrics."""
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Constructs and returns a DetectionModel."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Constructs and returns a DetectionValidator."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Creates and returns a DataLoader for the given dataset."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with shuffle, setting shuffle=False")
            shuffle = False
        
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Builds a YOLO dataset with appropriate augmentations.

        Args:
            img_path (str): Path to the image directory.
            mode (str): 'train' or 'val'.
            batch (int, optional): Batch size, used for rectangular training.

        Returns:
            Dataset: The constructed dataset.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Creates a dictionary of named loss components."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def set_model_attributes(self):
        """Sets model attributes such as number of classes and class names."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def progress_string(self):
        """Returns the header string for the training progress log."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")

    def plot_training_samples(self, batch, ni):
        """Plots a batch of training samples with labels."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Plots the distribution of training labels."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def save_metrics(self, metrics):
        """Appends metrics for the current epoch to the results CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        header = "" if self.csv.exists() else (("%s," * (len(metrics) + 2)) % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n"
        with open(self.csv, "a") as f:
            f.write(header + ("%.6g," * (len(metrics) + 2) % tuple([self.epoch + 1, time.time() - self.train_time_start] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plots the training and validation metrics from the results CSV."""
        plot_results(file=self.csv, on_plot=self.on_plot)

    def on_plot(self, name, data=None):
        """Callback to handle generated plots."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs a final validation using the best saved checkpoint."""
        for f in self.last, self.best:
            if f.exists() and f is self.best:
                LOGGER.info(f"\nValidating with best model {f}...")
                self.validator.args.plots = self.args.plots
                self.metrics = self.validator(model=f)
                self.metrics.pop("fitness", None)
                self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Checks for a resume checkpoint and updates arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                # Find the latest run or a specific checkpoint
                last = Path(check_file(resume) if Path(resume).exists() else get_latest_run())
                ckpt_args = attempt_load_weights(last).args
                # Ensure dataset path is valid
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data
                
                # Update current args with resumed args
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)
                # Allow overriding certain arguments on resume
                for k in ("imgsz", "batch", "device", "close_mosaic"):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
                self.resume = True
            except Exception as e:
                raise FileNotFoundError("Resume checkpoint not found. Please provide a valid path.") from e
        else:
            self.resume = False

    def resume_training(self, ckpt):
        """Resumes training state from a loaded checkpoint."""
        if ckpt is None or not self.resume:
            return
            
        start_epoch = ckpt.get("epoch", -1) + 1
        best_fitness = ckpt.get("best_fitness", 0.0)
        
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]

        assert start_epoch > 0, f"{self.args.model} training is finished, cannot resume."
        LOGGER.info(f"Resuming training from epoch {start_epoch} to {self.epochs} total epochs.")
        
        if self.epochs < start_epoch:
            LOGGER.info(f"Model has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt["epoch"]
        
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Disables mosaic augmentation in the dataloader."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            LOGGER.info("Closing dataloader mosaic augmentation.")
            self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, "close_mosaic"):
                self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        """
        Constructs the optimizer for training.

        Separates model parameters into three groups: weights with decay, weights
        without decay (e.g., BatchNorm), and biases.

        Args:
            model (nn.Module): The model to optimize.
            name (str): The name of the optimizer (currently supports 'AdamW').
            lr (float): The learning rate.
            momentum (float): The momentum for the optimizer.
            decay (float): The weight decay.

        Returns:
            torch.optim.Optimizer: The constructed optimizer.
        """
        g0, g1, g2 = [], [], []  # parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # Normalization layers

        # Separate parameters into groups
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g2.append(param)  # Biases
                elif isinstance(module, bn):
                    g1.append(param)  # BatchNorm weights
                else:
                    g0.append(param)  # Other weights

        # Use AdamW optimizer
        optimizer = optim.AdamW(g2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        optimizer.add_param_group({"params": g0, "weight_decay": decay})
        optimizer.add_param_group({"params": g1, "weight_decay": 0.0})
        
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g1)} weight(decay=0.0), {len(g0)} weight(decay={decay}), {len(g2)} bias(decay=0.0)"
        )
        return optimizer