from __future__ import annotations
import os
import joblib
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from huggingface_hub import hf_hub_download
from utils import unzip_file
from model import YOLO11Model



# Define the global variables
REPO_ID = "pt-sk/ADIS" 
FILENAME_IN_REPO = "dataset.zip"
LOCAL_DIR = os.getcwd()
TRAIN_PATH = f"{LOCAL_DIR}/dataset/train"
VAL_PATH = f"{LOCAL_DIR}/dataset/val"
TEST_PATH = f"{LOCAL_DIR}/dataset/test"
DATASET_PATH = f"{LOCAL_DIR}/{FILENAME_IN_REPO}"
REPO_TYPE = "dataset"
NUM_CLASSES = 10                                               
CLASSES = ['Cat', 'Cattle', 'Chicken', 'Deer', 'Dog', "Squirrel", 'Eagle', 'Goat', 'Rodents', 'Snake'] 
DATA_YAML_FILE = f"{LOCAL_DIR}/data.yaml"
MODEL_PATH = "yolo11n.pt"


# download the dataset and unzip it
# hf_hub_download(repo_id=REPO_ID, filename=FILENAME_IN_REPO, repo_type=REPO_TYPE, local_dir=LOCAL_DIR)
# unzip_file(DATASET_PATH, LOCAL_DIR)


# Get the number of CPU cores
num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# split paths for model
data_yaml = f"""
train: {TRAIN_PATH}
val: {VAL_PATH}
test: {TEST_PATH}

nc: {NUM_CLASSES}
names: {CLASSES}
"""

# write data yaml file
with open(DATA_YAML_FILE, "w") as file:
    file.write(data_yaml)
    print("data yaml file written!.............")
    
    
# Define the objective function
def objective(trial):
    
    # Define callback to report intermediate results
    def on_train_epoch_end(score, epoch):
        trial.report(score, step=epoch)  
        if trial.should_prune():
            raise optuna.TrialPruned()

    callbacks = {
        "on_train_epoch_end" : on_train_epoch_end
    }
    
    # Define hyperparameters using Optuna suggestions
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-3, log=True)
    lrf = trial.suggest_float("lrf", 0.1, 1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.01, log=True)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.6, 0.9)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    
    CONFIG_DICT = {
    "task": "detect",
    "mode": "train",
    "bohb": True,
    "custom_callbacks": callbacks,
    "data": DATA_YAML_FILE,
    "batch": 64, # 576
    "imgsz": 320,
    "save": True,
    "device": 0,
    "workers": num_cores,
    "pretrained": True,
    "optimizer": "AdamW",
    "seed": 42,
    "epochs": 10,
    "warmup_epochs": 2,
    "patience": 2}

    # Train YOLO model
    model = YOLO11Model(MODEL_PATH)
    model.train(**CONFIG_DICT, lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay, warmup_momentum=warmup_momentum)
    
    # Return validation mAP as the objective value
    return model.score 

# Define the study
NUM_TRIALS = 40
study = optuna.create_study(direction='maximize', 
                            sampler=optuna.samplers.TPESampler(), 
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name="yolo11_tuning",
                            load_if_exists=True,
                            storage="sqlite:///yolo11_tuning.db")

# Optimize with a callback to stop after NUM_TRIALS complete trials
study.optimize(
    objective,
    n_trials=NUM_TRIALS,
    callbacks=[MaxTrialsCallback(NUM_TRIALS, states=(TrialState.COMPLETE,))]
)

joblib.dump(study, "optuna_study.pkl")