# Animal Intrusion Detection System (ADIS)

ADIS is a compact, end-to-end repository for animal detection built around the YOLOv11 family.
It provides model definitions, training and validation utilities, dataset helpers, evaluation tools
and example notebooks to train, tune and evaluate object detection models on custom datasets.

This README documents how to set up the project, run training/tuning/evaluation, and use the
key components such as `YOLO11Model`, `DetectionTrainer`, and `DetectionValidator`.
---

## Table of contents
- Project overview
- Features
- Repository layout
- Requirements
- Installation (Windows / PowerShell)
- Quick start
  - Train
  - Tune (Optuna)
  - Evaluate / Validate
  - Inference
- Notebooks
- Key components (trainer, validator, utils)
- Configuration and checkpoints
- Troubleshooting
- Contributing
- License

---

## Project overview

ADIS (Animal Intrusion Detection System) aims to provide a practical, reproducible starting point for
building animal detection systems using YOLOv11. It layers model definitions and training workflows
on top of Ultralytics utilities and includes convenience wrappers and notebooks for common tasks:

- Training and checkpointing (see `trainer.py` / `YOLOv11_training.ipynb`)
- Hyperparameter tuning with Optuna (see `YOLOv11_tunning.ipynb`)
- Validation and COCO/LVIS JSON export via `DetectionValidator` (`validator.py`)
- Lightweight utilities for I/O and per-class diagnostics (`utils.py`)

The repository is suitable for experiments (research) and as a starting point for deployment.
---

## Repository layout

Top-level files and folders (high level):

- `model.py`           - High-level `YOLO11Model` wrapper for train/val/export/predict.
- `detector.py`       - Model parsing and `DetectionModel` implementation (network builder).
- `blocks.py`         - Neural network building blocks used by the model definitions.
- `trainer.py`        - `DetectionTrainer` class: training loop, optimizer, scheduler, EMA, DDP support.
- `validator.py`      - `DetectionValidator` class: validation loop, NMS, metrics, COCO export.
- `utils.py`          - Small helpers (autopad, anchors, unzip, AccuracyIoU diagnostics).
- `config/`           - YAML model definitions (e.g. `yolo11n.yaml`, `yolo11m.yaml`, ...).
- `study/`            - saved Optuna study folders (per-model tuning results).
- `requirements.txt`  - Python dependencies used by the notebooks and scripts.
- `YOLOv11_*.ipynb`   - Example notebooks for training, tuning and evaluation.

See inline module docstrings for further implementation details.

---

## Requirements

Minimum environment:

- Python 3.8+
- PyTorch (match to your CUDA/cpu setup)
- Ultralytics package (repository uses Ultralytics utilities)

Install requirements (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (for official COCO/LVIS evaluation):

```powershell
pip install pycocotools  # COCO evaluation
# or
pip install lvis         # LVIS evaluation
```

Note: `pycocotools` can be difficult to build on Windows; use prebuilt wheels or conda where available.
---

## Quick start

Below are short examples to get you started. For interactive workflows prefer the notebooks.

1) Train

```powershell
# activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1
python -c "from model import YOLO11Model; YOLO11Model('').train(data='data.yaml', epochs=10)"
```

2) Tune (Optuna)

Open `YOLOv11_tunning.ipynb` and run the tuning notebook. It uses Optuna to search hyperparameters and
saves studies under `study/`.

3) Evaluate / Validate

Use the `YOLOv11_evaluate.ipynb` notebook for step-by-step evaluation. Programmatically you can call:

```python
from types import SimpleNamespace
from validator import DetectionValidator

args = SimpleNamespace(split='val', conf=0.001, iou=0.65, save_json=False, save_txt=False, save_hybrid=False,
             save_conf=0.25, plots=False, val=True, workers=4, half=False, single_cls=False,
             agnostic_nms=False, max_det=300, task='detect', verbose=True)

validator = DetectionValidator(dataloader=None, save_dir='runs/val', pbar=None, args=args)
# validator.init_metrics(model)  # call with a DetectionModel or path
# validator()  # run validation via BaseValidator API
```

4) Inference

After training, load the best checkpoint and run inference using `YOLO11Model`:

```python
from model import YOLO11Model
model = YOLO11Model('runs/detect/train/weights/best.pt')
results = model.predict('path/to/image.jpg')
results.show()
```
---

## Key components

Below are brief descriptions of the most relevant modules. See each file's docstring for details.

- `model.py` / `YOLO11Model` — high-level user-facing API. Use this to train, validate, tune, export and predict.

- `detector.py` / `DetectionModel` — lower-level model class that builds a PyTorch model from YAML specs (uses `blocks.py`).

- `trainer.py` / `DetectionTrainer` — training loop, optimizer construction, LR scheduling, DDP support, EMA, checkpointing.

- `validator.py` / `DetectionValidator` — validation loop built on `ultralytics.engine.validator.BaseValidator`; handles NMS,
  metric accumulation (mAP via `DetMetrics`), confusion matrices, optional COCO/LVIS JSON export and evaluation.

- `utils.py` / `AccuracyIoU` — small helpers. `AccuracyIoU` provides additional per-class IoU & accuracy diagnostics
  used during verbose validation reporting.
---

## Notebooks and examples

Notebook-driven examples are provided to make common workflows easier:

- `YOLOv11_training.ipynb` — end-to-end training example (download dataset, write data.yaml, train).
- `YOLOv11_tunning.ipynb` — Optuna based hyperparameter tuning example that uses the `YOLO11Model.train` API.
- `YOLOv11_evaluate.ipynb` — model evaluation and quick inference examples (download model, validate, run inference).

Use these notebooks in Colab / Kaggle or locally. They show how to download dataset/model from Hugging Face hub,
prepare data.yaml and run training/tuning/evaluation without wiring up CLI arguments.
---

## Configuration & checkpoints

- Model YAMLs: `config/*.yaml` contain model architecture definitions (width/depth multipliers, heads, anchors).
- Checkpoints saved by training are written to `runs/` with `weights/last.pt` and `weights/best.pt`.
- Optuna studies (when tuning) are saved under `study/` (subfolders per model variant).

When resuming training, `DetectionTrainer` inspects the checkpoint to restore optimizer/EMA state and args.
---

## Troubleshooting

- pycocotools on Windows: prefer pre-built wheels or conda packages; building from source often requires Visual Studio Build Tools.
- GPU OOM: reduce `batch` or `imgsz`, or use smaller model (yolo11n -> yolo11s -> yolo11m).
- Dataset paths: ensure `data.yaml` points to valid `train/val/test` folders and image files' names are numeric if you use COCO eval id mapping.

If you hit a runtime error, include the traceback and environment (python/pytorch versions) when opening an issue.
---

## Contributing

Contributions, bug reports and improvements are very welcome. Suggested flow:

1. Open an issue describing the change or bug.
2. Create a topic branch for your changes.
3. Submit a PR with tests / repro instructions where relevant.

## License & contact

This repository is licensed under the MIT License — see `LICENSE` in the project root.

For questions or collaboration, open an issue or submit a pull request.

## Acknowledgements

This project builds on PyTorch and Ultralytics tooling. Thanks to upstream authors for the reference
implementations and utilities that made this repository possible.

excellent tooling and reference implementations.