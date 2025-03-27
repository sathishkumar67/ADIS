import subprocess
import os
from huggingface_hub import hf_hub_download
from utils import unzip_file



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
hf_hub_download(repo_id=REPO_ID, filename=FILENAME_IN_REPO, repo_type=REPO_TYPE, local_dir=LOCAL_DIR)
unzip_file(DATASET_PATH, LOCAL_DIR)

# Number of GPUs available
n_gpus = 2  # Adjust based on your system

# List to store subprocesses
processes = []

# Launch a process for each GPU
for gpu_id in range(n_gpus):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # Assign a specific GPU to this process
    p = subprocess.Popen(['python', 'optimize_script.py'], env=env)
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.wait()