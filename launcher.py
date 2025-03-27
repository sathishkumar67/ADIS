import subprocess
import os

# Number of GPUs available
n_gpus = 4  # Adjust based on your system

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