import tensorflow as tf
import os
import shutil
import time
from datetime import datetime

def create_log_dir(base_dir, experiment_name, params=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if params:
      param_str = "_".join([f"{key}_{value}" for key, value in params.items()])
      log_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}_{param_str}")
    log_dir = os.path.join(base_dir, f"{timestamp}_{experiment_name}")
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def cleanup_logs(log_dir, days_to_keep=30):
    now = time.time()
    cutoff = now - (days_to_keep * 86400)  # 86400 seconds in a day

    for root, dirs, files in os.walk(log_dir, topdown=False):
        for name in dirs:
            dir_path = os.path.join(root, name)
            if os.path.getmtime(dir_path) < cutoff:
                shutil.rmtree(dir_path)
                print(f"Deleted: {dir_path}")

# Logging setup
base_log_dir = "logs"
experiment_name = "experiment1"
# params = {"lr": 0.01, "batch_size": 32}

log_dir = create_log_dir(base_log_dir, experiment_name)
writer = tf.summary.create_file_writer(log_dir)