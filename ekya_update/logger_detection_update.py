import os, shutil, pandas as pd
from collections import defaultdict
from ekya_update.common import atomic_to_csv, get_kst_as_string, load_dict_from_temp_file


class LoggerDetectionSubstitution:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir
        self.fin_file_path = os.path.join(self.base_dir, "fin.log")
        
        if os.path.exists(self.base_dir) and os.path.exists(self.fin_file_path):
            prev_dir_new_path = f"{self.base_dir}_replaced_on_{get_kst_as_string()}"
            os.rename(self.base_dir, prev_dir_new_path)
        else:
            pass
    
    def log_inference_results(self, camera_idx, task_id, chunk_id, log_items_path):
        inference_result_log_path = os.path.join(
            self.base_dir,
            "runtime",
            "inference",
            f"camera_{camera_idx}_task_{task_id}_chunk_{chunk_id}.csv",
        )
        prediction_json_path = os.path.join(
            self.base_dir,
            "runtime",
            "inference_predictions",
            f"camera_{camera_idx}_task_{task_id}_chunk_{chunk_id}.json",
        )
        log_items = load_dict_from_temp_file(log_items_path)
        prediction_dataset_temp_path = log_items.get("prediction_dataset_temp_path", [None])[0]
        if prediction_dataset_temp_path is not None and os.path.exists(prediction_dataset_temp_path):
            os.makedirs(os.path.dirname(prediction_json_path), exist_ok=True)
            shutil.copy2(prediction_dataset_temp_path, prediction_json_path)
            os.remove(prediction_dataset_temp_path)
            log_items["prediction_json_path"] = [prediction_json_path]
        else:
            log_items["prediction_json_path"] = [None]
        
        if "prediction_dataset_temp_path" in log_items:
            log_items.pop("prediction_dataset_temp_path")
        else:
            pass
        
        df = pd.DataFrame(log_items)
        atomic_to_csv(df, inference_result_log_path)
    
    def log_schedules(self, task_id, inference_resource_weights, training_resource_weights, current_hyperparameters):
        results = defaultdict(list)
        
        for camera_name, weight in training_resource_weights.items():
            results["task_id"].append(task_id)
            results["camera_idx"].append(int(camera_name))
            results["job_type"].append("training")
            results["weight"].append(weight)
            results["hp_id"].append(current_hyperparameters[camera_name]["id"])
            results["hp_epochs"].append(current_hyperparameters[camera_name].get("epochs", None))
        
        for camera_name, weight in inference_resource_weights.items():
            results["task_id"].append(task_id)
            results["camera_idx"].append(int(camera_name))
            results["job_type"].append("inference")
            results["weight"].append(weight)
            results["hp_id"].append(current_hyperparameters[camera_name]["id"])
            results["hp_epochs"].append(current_hyperparameters[camera_name].get("epochs", None))
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=["task_id", "camera_idx"], ascending=True)
        schedule_dir = os.path.join(self.base_dir, "runtime", "schedules")
        postfix = 1
        schedule_result_log_path = os.path.join(schedule_dir, f"task_{task_id}_{postfix}.csv")
        while os.path.exists(schedule_result_log_path):
            postfix += 1
            schedule_result_log_path = os.path.join(schedule_dir, f"task_{task_id}_{postfix}.csv")
        atomic_to_csv(df, schedule_result_log_path)
    
    def log_retraining_results(self, camera_idx, task_id, log_items):
        retraining_result_log_path = os.path.join(
            self.base_dir,
            "runtime",
            "retraining",
            f"camera_{camera_idx}_task_{task_id}.csv",
        )
        df = pd.DataFrame(log_items)
        atomic_to_csv(df, retraining_result_log_path)
