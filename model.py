"""model.py

High-level model wrapper for YOLOv11 providing a user-friendly API.

`YOLO11Model` acts as a thin façade around lower-level components: it can create models from YAML
configs, load checkpoints, run training/validation/export, and expose convenient helpers such as
`.predict()`, `.train()`, `.val()`, `.tune()` and `.export()`.

The class integrates with the repository's `DetectionModel`, `DetectionTrainer`, and
`DetectionValidator` to run full training/evaluation workflows while keeping a small public API for
experimentation and deployment.
"""
from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

import torch
import torch.nn as nn

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)
from ultralytics.models.yolo.detect import DetectionPredictor
from detector import DetectionModel
from trainer import DetectionTrainer
from validator import DetectionValidator



class YOLO11Model(nn.Module):
    """
    A high-level wrapper for YOLOv11 models, providing a user-friendly API for various tasks.

    This class acts as a façade around lower-level components for model creation, loading,
    training, validation, and exporting. It simplifies common workflows by exposing convenient
    methods like `.predict()`, `.train()`, `.val()`, and `.export()`.

    Attributes:
        callbacks (Dict): A dictionary of callback functions for various events.
        predictor (BasePredictor): The predictor object for making predictions.
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object for training the model.
        ckpt (Dict): Checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The model's configuration if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (Dict): A dictionary of configuration overrides.
        metrics (Dict): The latest training or validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The task type of the model (e.g., 'detect').
        model_name (str): The name of the model.
        score (float): The score achieved after training.

    Methods:
        __call__: Alias for the `predict` method.
        _new: Initializes a new model from a configuration file.
        _load: Loads a model from a checkpoint file.
        _check_is_pytorch_model: Verifies if the model is a PyTorch model.
        reset_weights: Resets the model's weights.
        load: Loads weights from a file into the model.
        save: Saves the model's state to a file.
        info: Logs or returns information about the model.
        fuse: Fuses layers for optimized inference.
        predict: Performs object detection predictions.
        track: Performs object tracking.
        val: Validates the model on a dataset.
        benchmark: Benchmarks the model on various export formats.
        export: Exports the model to different formats.
        train: Trains the model on a dataset.
        tune: Performs hyperparameter tuning.
        _apply: Applies a function to the model's tensors.
        add_callback: Adds a callback function for an event.
        clear_callback: Clears all callbacks for an event.
        reset_callbacks: Resets all callbacks to their default state.

    Examples:
        >>> from ultralytics import YOLO
        >>> # Load a pretrained model
        >>> model = YOLO("yolo11n.pt")
        >>> # Predict on an image
        >>> results = model.predict("image.jpg")
        >>> # Train the model
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> # Validate the model
        >>> metrics = model.val()
        >>> # Export the model to ONNX format
        >>> model.export(format="onnx")
    """
    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = "detect",
        verbose: bool = True) -> None:
        """
        Initializes a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of
        model sources, including local files (*.pt, *.yaml), Ultralytics HUB models, and Triton Server models.

        Args:
            model (Union[str, Path]): Path or name of the model to load or create.
                Can be a local file path, a model name from Ultralytics HUB, or a Triton Server URL.
                Defaults to "yolo11n.pt".
            task (str): The task type for the model, e.g., 'detect'. Defaults to "detect".
            verbose (bool): If True, enables verbose output during initialization and operations. Defaults to True.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            ValueError: If the model file or configuration is invalid.
            ImportError: If required dependencies for a specific model type are not installed.
        """
        super().__init__()
        # Initialize core attributes
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # To be initialized on first prediction
        self.model = None
        self.trainer = None
        self.ckpt = {}  # Checkpoint data from .pt file
        self.cfg = None  # Model configuration from .yaml file
        self.ckpt_path = None
        self.overrides = {}  # User-defined configuration overrides
        self.metrics = None  # Stores validation/training metrics
        self.session = None  # HUB session object
        self.task = task
        self.model_name = None
        self.score = None # Stores score after training

        model = str(model).strip()

        # Check for and handle Ultralytics HUB models
        if self.is_hub_model(model):
            checks.check_requirements("hub-sdk>=0.0.12")
            session = HUBTrainingSession.create_session(model)
            model = session.model_file  # Get the local model file path from the session
            if session.train_args:  # If training arguments are sent from HUB
                self.session = session

        # Check for and handle Triton Server models
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"
            return

        # Set environment variable to prevent deterministic warnings with CUDA
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Load or create a new YOLO model based on file type
        if Path(model).suffix in {".yaml", ".yml"}:
            # Create a new model from a YAML configuration file
            self._new(model, task=task, verbose=verbose)
        else:
            # Load an existing model from a weights file (e.g., .pt)
            self._load(model, task=task)

        # Remove super().training to allow access to self.model.training
        # This is a workaround to avoid conflicts with nn.Module's training attribute
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        An alias for the `predict` method, allowing the model instance to be called directly.

        This simplifies making predictions by calling the model instance as a function.

        Args:
            source (Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor], optional):
                The input source for predictions. Can be a file path, URL, PIL image, numpy array,
                PyTorch tensor, or a list/tuple of these.
            stream (bool): If True, treats the input as a continuous stream for predictions. Defaults to False.
            **kwargs: Additional keyword arguments to configure the prediction process.

        Returns:
            list: A list of prediction results, where each result is a `Results` object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        Checks if the given model string corresponds to a Triton Server URL.

        Args:
            model (str): The model string to check.

        Returns:
            bool: True if the model string is a Triton Server URL, False otherwise.

        Examples:
            >>> YOLO11Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            True
            >>> YOLO11Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit
        url = urlsplit(model)
        # A valid Triton URL must have a network location, path, and a supported scheme
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        Checks if the provided model string is an Ultralytics HUB model URL.

        Args:
            model (str): The model string to check.

        Returns:
            bool: True if the model is a HUB model, False otherwise.

        Examples:
            >>> YOLO11Model.is_hub_model("https://hub.ultralytics.com/models/MODEL_ID")
            True
            >>> YOLO11Model.is_hub_model("yolo11n.pt")
            False
        """
        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initializes a new model from a YAML configuration file.

        This method infers the task type from the model definitions if not explicitly provided.

        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str, optional): The specific task for the model. If None, it's inferred from the config.
            model (torch.nn.Module, optional): A custom model instance. If provided, it's used instead of creating a new one.
            verbose (bool): If True, displays model information during loading.
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        # Use a provided model or load one dynamically based on the task
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Set model args to allow direct export from YAML configurations
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        Args:
            weights (str): Path to the model weights file (e.g., *.pt).
            task (str, optional): The task associated with the model. If None, it's inferred from the model weights.
        """
        # Download weights if the path is a URL
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])
        # Add suffix if missing (e.g., 'yolo11n' -> 'yolo11n.pt')
        weights = checks.check_model_file_from_stem(weights)

        if Path(weights).suffix == ".pt":
            # Load a PyTorch checkpoint
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            # Handle other weight formats (e.g., ONNX, TensorRT)
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        Ensures the model is a PyTorch model and raises a TypeError if not.

        This check is necessary for operations like training, exporting, and fusing,
        which are only supported for PyTorch models.

        Raises:
            TypeError: If the model is not a PyTorch module or a *.pt file.
        """
        is_pt_file = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        is_pt_module = isinstance(self.model, torch.nn.Module)
        if not (is_pt_module or is_pt_file):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model for this operation, but it is a different format. "
                "PyTorch models support training, validation, prediction, and export. "
                "Exported formats like ONNX or TensorRT only support prediction and validation."
            )

    def reset_weights(self) -> YOLO11Model:
        """
        Resets the model's weights to their initial state.

        This iterates through all modules and calls their `reset_parameters` method if available.

        Returns:
            YOLO11Model: The model instance with reset weights.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> YOLO11Model:
        """
        Loads parameters from a weights file into the current model.

        Args:
            weights (Union[str, Path]): Path to the weights file. Defaults to "yolo11n.pt".

        Returns:
            YOLO11Model: The model instance with loaded weights.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Saves the current model state to a file.

        The saved file includes the model weights and additional metadata like the
        save date and Ultralytics version.

        Args:
            filename (Union[str, Path]): The path to save the model to. Defaults to "saved_model.pt".
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__

        # Create a dictionary of updates to be saved in the checkpoint
        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        # Save the combined checkpoint and updates dictionary
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns information about the model architecture.

        Args:
            detailed (bool): If True, shows detailed information about layers and parameters. Defaults to False.
            verbose (bool): If True, prints the information. If False, returns it as a list. Defaults to True.

        Returns:
            list: A list of strings containing model information if `verbose` is False.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers for optimized inference speed.

        This process combines a convolution and batch normalization into a single layer,
        reducing computation and memory overhead during forward passes.
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Generates image embeddings from a given source.

        This is a wrapper around the `predict` method, configured to produce embeddings.
        By default, it uses the output of the second-to-last layer.

        Args:
            source (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional):
                The image source for generating embeddings.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the embedding process.

        Returns:
            list: A list containing the image embeddings as PyTorch tensors.
        """
        if not kwargs.get("embed"):
            # Default to embedding the output of the second-to-last layer
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Performs predictions on a given image source.

        Args:
            source (Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor], optional):
                The input source for predictions. If None, uses a default asset.
            stream (bool): If True, treats the source as a continuous stream. Defaults to False.
            predictor (BasePredictor, optional): A custom predictor instance. If None, a default one is created.
            **kwargs: Additional keyword arguments to configure the prediction process (e.g., `conf`, `imgsz`).

        Returns:
            List[Results]: A list of prediction results.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        # Check if running from the command-line interface (CLI)
        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        # Set default arguments for prediction
        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}
        # Combine overrides, defaults, and user kwargs (kwargs have highest priority)
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)  # For SAM-type models

        # Initialize or update the predictor
        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        
        # Set prompts for SAM-type models if provided
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)
            
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Conducts object tracking on a given input source.

        Args:
            source (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional):
                Input source for tracking (e.g., a video file).
            stream (bool): If True, treats the source as a continuous video stream. Defaults to False.
            persist (bool): If True, persists trackers between calls. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the tracking process.

        Returns:
            List[Results]: A list of tracking results.
        """
        # Register trackers if they haven't been already
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker
            register_tracker(self, persist)
            
        # ByteTrack requires a low confidence threshold to consider low-score detections
        kwargs["conf"] = kwargs.get("conf") or 0.1
        # Tracking is typically done with a batch size of 1 for video streams
        kwargs["batch"] = kwargs.get("batch") or 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        Validates the model on a dataset.

        Args:
            validator (ultralytics.engine.validator.BaseValidator, optional): A custom validator instance.
            **kwargs: Arbitrary keyword arguments for customizing the validation process (e.g., `data`, `imgsz`).

        Returns:
            ultralytics.utils.metrics.DetMetrics: Validation metrics.
        """
        # Set default arguments for validation
        custom = {"rect": True}
        # Combine overrides, defaults, and user kwargs
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        Benchmarks the model's performance across various export formats.

        This evaluates metrics like inference speed and accuracy for formats like
        ONNX, TensorRT, etc.

        Args:
            **kwargs: Arbitrary keyword arguments to customize the benchmarking process
                (e.g., `data`, `imgsz`, `half`, `int8`, `device`).

        Returns:
            Dict: A dictionary containing the benchmark results.
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose", False),
            format=kwargs.get("format", ""),
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment (e.g., ONNX, TensorRT).

        Args:
            **kwargs: Arbitrary keyword arguments to customize the export process
                (e.g., `format`, `imgsz`, `half`, `int8`, `simplify`).

        Returns:
            str: The path to the exported model file.
        """
        self._check_is_pytorch_model()
        from ultralytics.engine.exporter import Exporter

        # Set default arguments for exporting
        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # Reset device to avoid multi-GPU errors
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        bohb = False,
        custom_callbacks = None,
        **kwargs: Any):
        """
        Trains the model on a dataset.

        Args:
            trainer (BaseTrainer, optional): A custom trainer instance. If None, a default one is used.
            bohb (bool): Flag to enable BOHB hyperparameter optimization. Defaults to False.
            custom_callbacks (list, optional): A list of custom callbacks.
            **kwargs: Arbitrary keyword arguments for training configuration
                (e.g., `data`, `epochs`, `batch`, `imgsz`, `optimizer`).

        Returns:
            Dict: Training metrics.
        """
        self._check_is_pytorch_model()
        
        # If using a HUB session, prioritize HUB training arguments
        if hasattr(self.session, "model") and self.session.model.id:
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ Using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args

        checks.check_pip_update_available()

        # Load overrides from a config file if provided
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        # Set method-specific default arguments
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        # Initialize the trainer
        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks, bohb=bohb, custom_callbacks=custom_callbacks)
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.hub_session = self.session
        self.trainer.train()
        self.score = self.trainer.score
        
        # Update model state after training on the main process
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conducts hyperparameter tuning for the model.

        Args:
            use_ray (bool): If True, uses Ray Tune for tuning. Defaults to False.
            iterations (int): The number of tuning iterations. Defaults to 10.
            *args: Additional positional arguments for the tuner.
            **kwargs: Additional keyword arguments for the tuner.

        Returns:
            Dict: A dictionary containing the results of the hyperparameter search.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from ultralytics.utils.tuner import Tuner
            custom = {}
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> YOLO11Model:
        """
        Applies a function to the model's tensors.

        Used for operations like moving the model to a different device (`.to(device)`)
        or changing its precision (`.half()`).

        Args:
            fn (Callable): The function to apply to the model's tensors (e.g., `lambda t: t.cuda()`).

        Returns:
            YOLO11Model: The model instance after applying the function.
        """
        self._check_is_pytorch_model()
        # Apply the function to the model's tensors using the parent method
        self = super()._apply(fn)
        # Reset predictor since the device may have changed
        self.predictor = None
        # Update the device in overrides
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieves the class names associated with the model.

        Returns:
            Dict[int, str]: A dictionary mapping class indices to class names.
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        # If predictor is not initialized (e.g., for exported models), set it up
        if not self.predictor:
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated (e.g., 'cpu', 'cuda:0').

        Returns:
            torch.device: The device of the model.
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the data transformations (preprocessing steps) applied to the model's input.

        Returns:
            object: The transform object of the model if available, otherwise None.
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event (e.g., 'on_train_start').

        Args:
            event (str): The name of the event to attach the callback to.
            func (Callable): The callback function to register.
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        Args:
            event (str): The name of the event for which to clear callbacks.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions, removing any custom ones.
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Resets specific arguments when loading a PyTorch model checkpoint.

        This retains only essential arguments to avoid conflicts with new training sessions.

        Args:
            args (dict): A dictionary of model arguments from a checkpoint.

        Returns:
            dict: A new dictionary containing only the preserved arguments.
        """
        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """
        Dynamically loads the appropriate module (e.g., trainer, validator) based on the model's task.

        Args:
            key (str): The type of module to load ('model', 'trainer', 'validator', 'predictor').

        Returns:
            object: The loaded module class.

        Raises:
            NotImplementedError: If the requested module is not supported for the current task.
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # Get the calling function's name
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        A mapping from task names to their corresponding implementation classes.

        This allows for dynamic loading of modules based on the model's task.

        Returns:
            dict: A dictionary mapping tasks to their model, trainer, validator, and predictor classes.
        """
        return {"detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            }}

    def eval(self):
        """
        Sets the model to evaluation mode.

        This affects layers like Dropout and BatchNorm that behave differently
        during training and inference.

        Returns:
            YOLO11Model: The model instance in evaluation mode.
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        Delegates attribute access to the underlying `self.model`.

        This allows direct access to attributes of the wrapped model, e.g., `model.stride`.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The requested attribute's value.
        """
        # If the attribute is 'model' itself, return it from the nn.Module's dictionary
        if name == "model":
            return self._modules["model"]
        # Otherwise, get the attribute from the wrapped model
        return getattr(self.model, name)