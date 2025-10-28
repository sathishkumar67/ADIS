"""blocks.py

Collection of neural network building blocks used by YOLOv11 models.

This module provides commonly used modules such as Conv, DWConv, Bottleneck, C2f, C3, C3k, SPPF,
attention-based blocks (Attention, PSABlock, C2PSA), DFL and the Detect head. These components are
designed to be composable when building model architectures defined by YAML files.

The docstrings on the classes explain the purpose and expected tensor shapes where relevant. This file
avoids performing any model I/O and focuses solely on layer implementations.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Conv(nn.Module):
    """
    Standard convolution block with batch normalization and SiLU activation.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int, optional): Kernel size. Defaults to 1.
        s (int, optional): Stride. Defaults to 1.
        p (int, optional): Padding. If None, padding is calculated automatically. Defaults to None.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        d (int, optional): Dilation rate. Defaults to 1.
        act (bool, optional): If True, apply activation function. Defaults to True.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes the Conv layer with a 2D convolution, batch normalization, and SiLU activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """
        Forward pass through the Conv block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out).
        """
        # Apply convolution, batch normalization, and activation
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Forward pass for a fused Conv layer (conv + bn).
        Used after the model has been fused for faster inference.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with activation applied.
        """
        return self.act(self.conv(x))

class DWConv(Conv):
    """
    Depth-wise convolution layer.

    A specific type of grouped convolution where the number of groups is equal to the
    number of input channels. This results in each input channel being convolved with
    its own set of filters.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int, optional): Kernel size. Defaults to 1.
        s (int, optional): Stride. Defaults to 1.
        d (int, optional): Dilation rate. Defaults to 1.
        act (bool, optional): If True, apply activation function. Defaults to True.
    """
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initializes a Depth-wise convolution by setting groups to be the greatest common divisor of c1 and c2."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Bottleneck(nn.Module):
    """
    Standard bottleneck block with a residual (shortcut) connection.

    This block consists of two convolutional layers. The first reduces the channel
    dimension, and the second restores it. A shortcut connection adds the input
    to the output of the convolutions if the dimensions match.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): If True, add a shortcut connection. Defaults to True.
        g (int, optional): Number of groups for the second convolution. Defaults to 1.
        k (tuple[int, int], optional): Kernel sizes for the two convolutions. Defaults to (3, 3).
        e (float, optional): Expansion factor for the hidden channels. Defaults to 0.5.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass through the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor. If shortcut is enabled, it's the sum of input and transformed input.
        """
        # If the shortcut connection is enabled, add the input to the output of the convolutions
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (Faster R-CNN inspired).

    This block splits the input tensor into two parts. One part goes through a series of
    Bottleneck blocks, and the other part is a skip connection. The results are then
    concatenated. This design improves gradient flow and reduces computation.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of Bottleneck blocks. Defaults to 1.
        shortcut (bool, optional): If True, use shortcut connections in Bottlennecks. Defaults to False.
        g (int, optional): Number of groups for convolutions in Bottlenecks. Defaults to 1.
        e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        Forward pass through the C2f layer.

        The input is expanded and split into two chunks. One chunk is processed sequentially
        by Bottleneck blocks, and the results are concatenated before a final convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Expand channels and split into two parts
        y = list(self.cv1(x).chunk(2, 1))
        # Apply Bottleneck blocks sequentially to the second part
        y.extend(m(y[-1]) for m in self.m)
        # Concatenate all parts and apply the final convolution
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """
        Alternative forward pass using `split()` instead of `chunk()`.
        Functionally equivalent to `forward` but may have performance differences.
        """
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.

    This is another variant of the Cross Stage Partial network. It splits the input
    after a convolution and processes one part through a series of Bottleneck blocks.
    The two parts are then concatenated and passed through a final convolution.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of Bottleneck blocks. Defaults to 1.
        shortcut (bool, optional): If True, use shortcut connections in Bottlenecks. Defaults to True.
        g (int, optional): Number of groups for convolutions in Bottlenecks. Defaults to 1.
        e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """
        Forward pass through the C3 block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Process the input in two parallel paths, one through bottlenecks and one direct convolution.
        # Then concatenate the results and apply a final convolution.
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """
    C3 block with a customizable kernel size.

    This class inherits from C3 but allows specifying a custom kernel size `k` for the
    Bottleneck blocks within it.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of Bottleneck blocks. Defaults to 1.
        shortcut (bool, optional): If True, use shortcut connections. Defaults to True.
        g (int, optional): Number of groups for convolutions. Defaults to 1.
        e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
        k (int, optional): Kernel size for the Bottleneck blocks. Defaults to 3.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module, setting a custom kernel size for the internal Bottlenecks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """
    A faster C2f implementation that can use C3k blocks.

    This block is structurally similar to C2f but provides an option to use `C3k` blocks
    instead of standard `Bottleneck` blocks, allowing for customizable kernel sizes.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of inner blocks (C3k or Bottleneck). Defaults to 1.
        c3k (bool, optional): If True, use C3k blocks; otherwise, use Bottleneck blocks. Defaults to False.
        e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for convolutions. Defaults to 1.
        shortcut (bool, optional): If True, use shortcut connections. Defaults to True.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with an option for C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # Use C3k blocks if c3k is True, otherwise default to Bottleneck blocks
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.

    SPPF is a more efficient version of the SPP layer. It uses multiple small max-pooling
    layers sequentially instead of parallel large ones, which is computationally faster
    while producing the same output.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int, optional): Kernel size for the max-pooling layers. Defaults to 5.
    """
    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer. It is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Forward pass through the SPPF block.

        The input is passed through a convolution, then pooled three times sequentially.
        The outputs of the initial convolution and each pooling are concatenated and
        passed through a final convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with aggregated multi-scale features.
        """
        x = self.cv1(x)
        # Sequentially apply max-pooling and collect feature maps
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # Concatenate the original and pooled feature maps and apply final convolution
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    This module computes self-attention on an input tensor by projecting it into query (q),
    key (k), and value (v) representations. It uses scaled dot-product attention,
    leveraging PyTorch's optimized implementation (`F.scaled_dot_product_attention`).

    Args:
        dim (int): The number of input channels (embedding dimension).
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        attn_ratio (float, optional): Ratio to determine the dimension of the key. Defaults to 0.5.
    """
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False) # Positional encoding

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        # Project input to query, key, and value
        qkv = self.qkv(x)
        # Reshape and split into q, k, v for each head
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # Transpose for attention calculation
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Use PyTorch's optimized scaled dot-product attention (e.g., Flash Attention)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)

        # Reshape back to image format, add positional encoding, and apply final projection
        attn = attn.transpose(1, 2).contiguous().view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(attn)


class PSABlock(nn.Module):
    """
    Position-Sensitive Attention (PSA) block.

    This block combines a multi-head self-attention layer with a feed-forward network (FFN).
    Residual connections are used around both the attention and FFN components.

    Args:
        c (int): Number of input and output channels.
        attn_ratio (float, optional): Attention ratio for the `Attention` module. Defaults to 0.5.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        shortcut (bool, optional): If True, use residual connections. Defaults to True.
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with an attention module and a feed-forward network."""
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Forward pass through the PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply attention with a residual connection
        x = x + self.attn(x) if self.add else self.attn(x)
        # Apply feed-forward network with a residual connection
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """
    CSP block with Position-Sensitive Attention.

    This module is a variant of the C2f block that replaces `Bottleneck` blocks with `PSABlock`s.
    It splits the input, processes one part through a sequence of attention blocks, and then
    concatenates the results.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of PSABlock modules. Defaults to 1.
        e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified channels and number of PSABlocks."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Forward pass through the C2PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Split the feature map into two parts
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # Process the second part through the PSA blocks
        b = self.m(b)
        # Concatenate and apply the final convolution
        return self.cv2(torch.cat((a, b), 1))


class Concat(nn.Module):
    """
    Concatenation layer.

    A simple module that concatenates a list of tensors along a specified dimension.

    Args:
        dimension (int, optional): The dimension along which to concatenate. Defaults to 1.
    """
    def __init__(self, dimension=1):
        """Initializes the Concat module with the specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenates the input list of tensors.

        Args:
            x (list[torch.Tensor]): A list of tensors to concatenate.

        Returns:
            torch.Tensor: The concatenated tensor.
        """
        return torch.cat(x, self.d)

class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) Integral module.

    This module is used to convert a discrete probability distribution over box locations
    into a single continuous value. It does this by performing a weighted sum (via a fixed
    1D convolution) of the distribution, where the weights are the indices of the bins.

    Proposed in Generalized Focal Loss: https://ieeexplore.ieee.org/document/9792391

    Args:
        c1 (int, optional): The number of bins in the distribution (e.g., `reg_max`). Defaults to 16.
    """
    def __init__(self, c1=16):
        """Initializes the DFL module with a fixed-weight convolution."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # Initialize weights as a linear sequence from 0 to c1-1
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        Forward pass for DFL.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 4 * c1, anchors) representing
                              the logits of the distributions for top, bottom, left, right.

        Returns:
            torch.Tensor: Output tensor of shape (batch, 4, anchors) with the integrated
                          bounding box coordinates.
        """
        b, _, a = x.shape  # batch, channels, anchors
        # Reshape to (b, 4, c1, a), apply softmax, perform weighted sum with conv, and reshape back
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)



class Detect(nn.Module):
    """
    YOLOv11 Detection Head.

    This module takes feature maps from the backbone/neck and produces final predictions
    for bounding boxes and class probabilities. It includes separate convolutional branches
    for regression (bounding boxes) and classification.

    Attributes:
        nc (int): Number of classes.
        nl (int): Number of detection layers (feature map scales).
        reg_max (int): Number of bins for the Distribution Focal Loss.
        no (int): Number of outputs per anchor (4 * reg_max for box + nc for class).
        stride (torch.Tensor): Strides for each feature map level.
    """
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max detections per image
    shape = None
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """
        Initializes the Detect head.

        Args:
            nc (int, optional): Number of classes. Defaults to 80.
            ch (tuple[int, ...]): A tuple of input channel sizes for each feature map.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # Convolutional layers for box regression
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # Convolutional layers for class prediction
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch))
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def bias_init(self):
        """
        Initializes the biases of the final convolutional layers.
        This can help with training stability, especially at the beginning.
        Requires model strides to be computed first.
        """
        # Initialize biases for the final conv layers of box and class prediction heads
        for a, b, s in zip(self.cv2, self.cv3, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box bias
            # Initialize class bias based on a heuristic for object presence
            b[-1].bias.data[: m.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def forward(self, x):
        """
        Forward pass of the detection head.

        Args:
            x (list[torch.Tensor]): A list of feature maps from the neck.

        Returns:
            torch.Tensor or tuple: During training, returns raw feature maps.
                                   During inference, returns decoded predictions.
        """
        # Apply box and class prediction heads to each feature map
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:  # Training path returns raw logits
            return x
        
        # Inference path
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """
        Performs inference by decoding the raw feature maps.
        
        Args:
            x (list[torch.Tensor]): List of raw feature maps from the prediction heads.

        Returns:
            torch.Tensor: Decoded predictions of shape (batch_size, 4 + nc, num_anchors).
        """
        shape = x[0].shape  # BCHW of the first feature map
        # Flatten and concatenate feature maps from all levels
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # Generate anchors and strides if needed (for dynamic shapes)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # Split concatenated predictions into box and class parts
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # Decode box predictions from distribution to (x, y, w, h)
        # `dist2bbox` converts (top, bottom, left, right) distances to bounding boxes
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # Concatenate decoded boxes and class probabilities (after sigmoid)
        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """
        Decodes bounding box predictions.

        Converts the output of the DFL module (distances to the four sides of the anchor box)
        into actual bounding box coordinates (either xywh or xyxy format).

        Args:
            bboxes (torch.Tensor): The predicted distances from the DFL module.
            anchors (torch.Tensor): The anchor points for each prediction.
            xywh (bool, optional): If True, return boxes in (x, y, w, h) format.
                                   If False, return in (x1, y1, x2, y2) format. Defaults to True.

        Returns:
            torch.Tensor: Decoded bounding boxes.
        """
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions to get final detections.

        This method is typically applied after Non-Maximum Suppression (NMS). It selects
        the top `max_det` detections based on confidence scores.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc).
                                  The last dimension is [x, y, w, h, class_probs...].
            max_det (int): Maximum number of detections to return per image.
            nc (int, optional): Number of classes. Defaults to 80.

        Returns:
            torch.Tensor: Processed predictions with shape (batch_size, max_det, 6).
                          The last dimension is [x, y, w, h, confidence, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16, 8400, 84)
        # Split predictions into boxes and class scores
        boxes, scores = preds.split([4, nc], dim=-1)
        # Find the top k detections based on the max class score
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        # Gather the corresponding boxes and scores
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        # Find the top scores and their class indices across all classes
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        # Combine into the final format [box, score, class_id]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)