import time, os, pandas as pd
from collections import defaultdict
from ekya_update.common import append_to_csv, get_kst_as_string, stop_sys, atomic_to_csv


class LoggerSubstitution:
    def __init__(self, stream_max_len=1000, stream_max_time=30, base_dir=None):
        self.stream_max_time = stream_max_time
        self.stream_max_len = stream_max_len
        self.base_dir = base_dir
        
        # Rename previous results
        if os.path.exists(self.base_dir):
            prev_dir_new_path = f"{self.base_dir}_replaced_on_{get_kst_as_string()}"
            os.rename(self.base_dir, prev_dir_new_path)

    # Below are logging during runtime
    def log_inference_results(self, camera_idx, task_id, chunk_id, log_items):
        inference_result_log_path = os.path.join(self.base_dir, "runtime", "inference", f"camera_{camera_idx}_task_{task_id}_chunk_{chunk_id}.csv")
        df = pd.DataFrame(log_items)
        atomic_to_csv(df, inference_result_log_path)
    
    def log_schedules(self, task_id, inference_resource_weights, training_resource_weights, current_hyperparameters):
        results = defaultdict(list)
    
        # Log Training Schedules
        for camera_name, weight in training_resource_weights.items():
            results["task_id"].append(task_id)
            results["camera_idx"].append(int(camera_name))
            results["job_type"].append("training")
            results["weight"].append(weight)
            results["hp_id"].append(current_hyperparameters[camera_name]["id"])
            results["hp_epochs"].append(current_hyperparameters[camera_name].get("epochs", None))

        # Log Inference Schedules
        for camera_name, weight in inference_resource_weights.items():
            results["task_id"].append(task_id)
            results["camera_idx"].append(int(camera_name))
            results["job_type"].append("inference")
            results["weight"].append(weight)
            results["hp_id"].append(current_hyperparameters[camera_name]["id"])
            results["hp_epochs"].append(current_hyperparameters[camera_name].get("epochs", None))
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=["task_id", "camera_idx"], ascending=True)
        schedule_result_log_path = os.path.join(self.base_dir, "runtime", "schedules", f"task_{task_id}.csv")
        atomic_to_csv(df, schedule_result_log_path)
        
    def log_retraining_results(self, camera_idx, task_id, log_items):
        retraining_result_log_path = os.path.join(self.base_dir, "runtime", "retraining", f"camera_{camera_idx}_task_{task_id}.csv")
        df = pd.DataFrame(log_items)
        atomic_to_csv(df, retraining_result_log_path)
        
    
    # Below are aggregation process for the logs made during runtime
    def aggregate_inference_results(self):
        message="currently aggregate_inference_results is not implemented"
        stop_sys(message)
    
    
    def aggregate_retraining_results(self):
        message="currently aggregate_retraining_results is not implemented"
        stop_sys(message)
                              
