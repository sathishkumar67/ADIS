from __future__ import annotations
import torch
import zipfile
import os
from tqdm import tqdm

def unzip_file(zip_path, target_dir):
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

    print("Unzipping completed.....")

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
