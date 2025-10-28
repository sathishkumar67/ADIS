# Animal Intrusion Detection System (ADIS)

This repository contains an implementation of an Animal Intrusion Detection System (ADIS) built on top of
the YOLOv11 family of object detectors. It provides code for model definition, training/validation utilities,
dataset handling, and evaluation (including a `DetectionValidator` class used for detection validation and COCO/LVIS export).

This README explains how to set up the project, run basic validation/evaluation, and use the `DetectionValidator`.

---

## Table of contents
- Project overview
- Repository structure
- Requirements
- Installation (Windows / PowerShell)
- Quick start (validation / evaluation)
- Using `DetectionValidator` (example)
- Notebooks and scripts
- Contributing and license

---

## Project overview

ADIS is designed to detect animals in images and video using YOLOv11 models. It includes utilities for
dataset building, evaluation, saving predictions in COCO/LVIS format, plotting, and a small set of
model variants (nano/small/medium) tuned for different deployment needs.

The code in this repository is intended for researchers and practitioners who want a compact, runnable
baseline for animal detection using the ultralytics YOLOv11 codebase.

---

## Repository structure (important files)

- `model.py`          : model definitions and network blocks
- `blocks.py`         : building blocks used by models
- `detector.py`       : detection utilities / wrappers
- `trainer.py`        : training loop (if present / configured)
- `validator.py`      : validation utilities (contains `DetectionValidator`)
- `utils.py`          : helper functions (accuracy, plotting, etc.)
- `requirements.txt`  : Python dependencies
- `config/`           : model config YAML files (yolo11*.yaml)
- `YOLOv11_evaluate.ipynb`: notebook for running evaluation and visualization

Refer to the file headers for more implementation details.

---

## Requirements

Primary dependency:
- Python 3.8+ (recommended 3.8–3.11)
- PyTorch (CUDA-enabled if using GPU)
- ultralytics (YOLOv11 utilities used by this code)

Other packages are listed in `requirements.txt`. To install them in a PowerShell on Windows:

```powershell
python -m venv .venv; 
.\.venv\Scripts\Activate.ps1; 
python -m pip install --upgrade pip; 
pip install -r requirements.txt
```

If you plan to run COCO/LVIS evaluation (export + official mAP), also install:

```powershell
pip install pycocotools  # for COCO evaluation
# or for LVIS:
pip install lvis
```

Note: installing `pycocotools` on Windows sometimes requires Visual Studio Build Tools or pre-built wheels.

---

## Quick start — run validation/evaluation

1. Activate your virtual environment (see Requirements).
2. Ensure your dataset config is available in `self.data` or use paths in `config/*.yaml`.
3. Use the included notebook `YOLOv11_evaluate.ipynb` to run evaluation interactively, or run a small
   Python snippet to construct and run the `DetectionValidator` (example below).

Example: programmatic validation using the `DetectionValidator` class

```python
from types import SimpleNamespace
from validator import DetectionValidator

# Minimal example args namespace - adapt paths/flags to your setup
args = SimpleNamespace(
    split='val', # key into data dict for validation files
    conf=0.001,
    iou=0.65,
    save_json=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=0.25,
    plots=False,
    val=True,
    workers=4,
    half=False,
    single_cls=False,
    agnostic_nms=False,
    max_det=300,
    plots_dir=None,
    task='detect',
    verbose=True
)

# If you have a model object `model` that exposes `.names` etc. call as follows:
validator = DetectionValidator(dataloader=None, save_dir='runs/val', pbar=None, args=args)
# initialize metrics using a model object - validator.init_metrics(model)
# then run validator on your dataloader (the class follows ultralytics BaseValidator patterns)
```

The repository includes a `YOLOv11_evaluate.ipynb` notebook that demonstrates setting up the dataloader,
running the validator, plotting predictions, and exporting COCO-format JSON for official evaluation.

---

## `DetectionValidator` (summary)

`DetectionValidator` (defined in `validator.py`) is a drop-in validator for detection tasks that extends
`ultralytics.engine.validator.BaseValidator`. Key features:

- Preprocesses batches (moves tensors to the configured device and normalizes images).
- Postprocesses model outputs (applies NMS and rescales boxes to original image coordinates).
- Accumulates metrics required for mAP computation (via `DetMetrics`) and builds confusion matrices.
- Optionally exports predictions to COCO or LVIS JSON and runs official mAP evaluation if those libraries are installed.

Usage notes:
- The class expects an `args` object containing validation options (conf/iou thresholds, save flags, plots, etc.).
- You typically call `validator.init_metrics(model)` before running validation to set up names, class maps, and helpers.
- See `validator.py` docstrings for per-method behavior (preprocess, postprocess, update_metrics, eval_json, ...).

---

## Notebooks and scripts

- `YOLOv11_evaluate.ipynb`: Guided evaluation and visualization — a good starting point to quickly run validation
  on your dataset and visualize results.
- `trainer.py`: (if present) training loop and hyperparameter handling
- `validator.py`: validation helpers and `DetectionValidator`

---

## Configuration

Model and dataset configuration YAMLs are stored under `config/` (e.g. `yolo11n.yaml`, `yolo11s.yaml`, `yolo11m.yaml`).
Edit these files to change model architecture, dataset paths, or training/validation settings.

---

## Troubleshooting

- If you see errors importing `ultralytics` or `pycocotools`, ensure you have installed the packages in the same
  Python environment you're running. On Windows, `pycocotools` installation can be tricky — use pre-built wheels
  or install via conda if possible.
- If GPU memory is exhausted, try lowering the batch size or using a smaller model (e.g., `yolo11n`).
- For image I/O issues, check that the paths inside your dataset config point correctly to the image files.

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Open an issue describing the feature or bug.
2. Create a branch for your change.
3. Submit a PR with clear description and tests where applicable.

---

## License & contact

This project includes a `LICENSE` file in the repository root. Please check the license for permitted uses.

For questions or collaboration, open an issue in this repository or contact the maintainer.

---

## Acknowledgements

This project leverages the Ultralytics YOLO ecosystem and PyTorch. Thanks to upstream authors for their
excellent tooling and reference implementations.