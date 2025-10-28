"""detector.py

Utilities for parsing model YAML definitions and building the DetectionModel used by YOLOv11.

This module provides `parse_model` which converts a model specification (typically loaded from a YAML)
into a torch.nn.Sequential model built from the block primitives in `blocks.py`. It also exposes
`DetectionModel`, a wrapper around the parsed network that provides convenience methods such as
`predict`, `load`, `fuse`, and `info` used throughout training and evaluation workflows.

The implementation aims to be compatible with multiple YOLO family conventions (v3/v5/v8/etc.) and
contains helpers for model introspection and optimization (e.g., fusing conv+bn).
"""
from copy import deepcopy
import contextlib
import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    time_sync,
)
from ultralytics.nn.tasks import yaml_model_load, TorchVision, Index
from blocks import *
try:
    import thop
except ImportError:
    thop = None  # conda support without 'ultralytics-thop' installed



def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """
    Parses a YOLO model dictionary from a YAML file into a PyTorch model.

    Args:
        d (dict): A dictionary representing the model architecture, usually loaded from a YAML file.
        ch (int): The number of input channels for the model.
        verbose (bool, optional): If True, logs information about the model architecture. Defaults to True.

    Returns:
        (nn.Sequential, list): A tuple containing the constructed PyTorch model (as an nn.Sequential)
                               and a sorted list of layer indices to be saved for feature concatenation.
    """
    import ast
    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # Define sets of modules for specific handling
    base_modules = frozenset(
        {
            Conv,
            Bottleneck,
            SPPF,
            C2PSA,
            DWConv,
            C2f,
            C3k2,
            C3,
            torch.nn.ConvTranspose2d
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            C2f,
            C3k2,
            C3,
            C2PSA
        }
    )
    # Iterate through the model definition (backbone and head)
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # Dynamically get the module class from its name
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        # Evaluate string arguments as Python expressions
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # Apply depth gain to the number of repetitions
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        # Handle different module types with specific logic
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect}):
            args.append([ch[x] for x in f])
            if m in {Detect}:
                m.legacy = legacy
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]
        
        # Create the module or a sequence of modules
        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        
        # Attach metadata to the module
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        
        # Add the 'from' index to the save list for feature concatenation
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


class DetectionModel(nn.Module):
    """
    YOLO detection model. This class wraps the parsed model and provides methods for forward pass,
    inference, fusing layers, and loading weights.
    """
    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """
        Initializes the YOLO detection model.

        Args:
            cfg (str or dict, optional): Path to the model configuration YAML file or a dictionary
                                        representing the model architecture. Defaults to "yolo11n.yaml".
            ch (int, optional): Number of input channels. Defaults to 3.
            nc (int, optional): Number of classes. If provided, it overrides the value in the YAML file.
                                Defaults to None.
            verbose (bool, optional): If True, prints model information during initialization.
                                    Defaults to True.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        
        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        
        # Build strides and initialize the detection head
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            def _forward(x):
                """A helper function for a single forward pass to compute strides."""
                return self.forward(x)

            # Calculate stride based on a dummy forward pass
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
    
    def forward(self, x, *args, **kwargs):
        """
        Performs a forward pass of the model. This method routes the input to either
        the loss function for training or the prediction function for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or a dictionary containing
                                     the image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): The loss tensor if `x` is a dict (training), otherwise the
                            network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, embed=None, augment=False):
        """
        Performs a forward pass through the network for inference.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool, optional): If True, prints the computation time and FLOPs of each layer.
                                    Defaults to False.
            visualize (bool, optional): If True, saves feature maps of the model. Defaults to False.
            augment (bool, optional): If True, applies augmentation during prediction. Defaults to False.
            embed (list, optional): A list of layer indices from which to return feature vectors/embeddings.
                                    Defaults to None.

        Returns:
            (torch.Tensor): The output of the model. If `embed` is used, returns concatenated embeddings.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # Get input from specified earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            # Profile layer-wise performance
            if profile:
                # Profile the computation time and FLOPs of a single layer
                c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
                flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
                t = time_sync()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
                LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
                if c:
                    LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
            
            # Run the module
            x = m(x)
            y.append(x if m.i in self.save else None)  # save output if in savelist
            
            # Visualize feature maps if requested
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            
            # Extract embeddings if requested
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def fuse(self, verbose=True):
        """
        Fuses Conv2d and BatchNorm2d layers into a single convolutional layer to improve
        computation efficiency during inference. This operation is performed in-place.

        Args:
            verbose (bool, optional): If True, prints model information after fusing.
                                      Defaults to True.

        Returns:
            (nn.Module): The fused model.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Checks if the model has been fused by counting the number of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. If the count
                                    is below this value, the model is considered fused.
                                    Defaults to 10.

        Returns:
            (bool): True if the number of BatchNorm layers is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints detailed information about the model, including layers, parameters, and FLOPs.

        Args:
            detailed (bool, optional): If True, prints detailed information about each layer.
                                       Defaults to False.
            verbose (bool, optional): If True, prints the model information. Defaults to True.
            imgsz (int, optional): The input image size used for calculating FLOPs. Defaults to 640.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or
        registered buffers, such as moving them to a different device.

        Args:
            fn (function): The function to apply (e.g., `lambda t: t.to(device)`).

        Returns:
            (nn.Module): The model with the function applied.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Loads pre-trained weights into the model from a state dictionary.

        Args:
            weights (dict | torch.nn.Module): A state dictionary or a PyTorch model containing
                                              the pre-trained weights.
            verbose (bool, optional): If True, logs the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Computes the loss for a given batch of data. Initializes the criterion if it doesn't exist.

        Args:
            batch (dict): A dictionary containing the input images and ground truth labels.
            preds (torch.Tensor | List[torch.Tensor], optional): Pre-computed model predictions.
                                                                  If None, a forward pass is performed.
                                                                  Defaults to None.
        
        Returns:
            (torch.Tensor): The calculated loss tensor.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)
    
    def init_criterion(self):
        """
        Initializes the loss criterion for the detection model.

        Returns:
            (nn.Module): The loss criterion module (e.g., v8DetectionLoss).
        """
        return v8DetectionLoss(self)