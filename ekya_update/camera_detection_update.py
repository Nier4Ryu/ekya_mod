import copy, math, os, time, numpy as np, ray
from ekya.CONFIG import RANDOM_SEED
from ekya_update.common import check_mps_is_running, logical_to_physical_gpu, stop_sys
from ekya_update.detection_dataset_update import (
    get_aligned_detection_dataset_dict_from_target_num_samples,
    get_detection_dataset_dict_with_sample_range_from_params,
    get_detection_dataset_dict_with_samples_from_params,
    get_detection_inference_dataloader_from_dataset_dict,
    get_detection_train_dataloader_from_dataset_dict,
    get_empty_detection_dataset_dict_from_source,
    get_padded_detection_dataset_dict_from_params,
    load_detection_dataset_dict_from_path,
)
from ekya_update.model_detection_update import RayMLDetectionModel


class CameraDetectionSubstitution(object):
    def __init__(self,
                 id,
                 dataset_name,
                 train_sample_json_path,
                 test_sample_json_path,
                 fps,
                 slo,
                 num_chunks,
                 start_task,
                 termination_task,
                 label_type,
                 train_split,
                 min_inference_resources_to_meet_slo,
                 training_memory_footprint_value,
                 inference_memory_footprint_value,
                 log_dir,
                 max_detection_exemplar_frames=2000,
                 target_num_samples=None):
        if not check_mps_is_running():
            message = "Ekya detection requires MPS server to be running! stopping sys!"
            stop_sys(message)
        else:
            self.id = id
            self.camera_idx = int(self.id)
            self.dataset_name = dataset_name
            self.fps = fps
            self.slo = slo
            self.num_chunks = num_chunks
            self.start_task = start_task
            self.termination_task = termination_task
            self.label_type = label_type
            self.train_split = train_split
            self.min_inference_resources_to_meet_slo = min_inference_resources_to_meet_slo
            self.training_memory_footprint_value = training_memory_footprint_value
            self.inference_memory_footprint_value = inference_memory_footprint_value
            self.log_dir = os.path.join(log_dir, "logger_generated")
            self.max_detection_exemplar_frames = max_detection_exemplar_frames
            self.current_task = -1
            self.current_inference_path = None
            
            start_t = time.time()
            self.train_dataset_dict = load_detection_dataset_dict_from_path(train_sample_json_path)
            self.test_dataset_dict = load_detection_dataset_dict_from_path(test_sample_json_path)
            self.test_dataset_dict = get_aligned_detection_dataset_dict_from_target_num_samples(
                dataset_dict=self.test_dataset_dict,
                target_num_samples=target_num_samples,
            )
            fin_t = time.time()
            print(f"loading detection jsons took {fin_t-start_t}s for camera:{self.id}")
            
            self.chunk_size = int(self.fps * self.slo)
            self.num_samples_per_task = int(self.chunk_size * self.num_chunks)
            if self.num_samples_per_task > 0:
                pass
            else:
                message = f"Invalid detection task size. fps:{self.fps}, slo:{self.slo}, num_chunks:{self.num_chunks}"
                stop_sys(message, raise_error=True)
            
            self.num_real_test_samples = len(self.test_dataset_dict["samples"])
            self.num_tasks = math.ceil(self.num_real_test_samples / self.num_samples_per_task)
            total_samples_needed = self.num_tasks * self.num_samples_per_task
            self.test_dataset_dict = get_padded_detection_dataset_dict_from_params(
                dataset_dict=self.test_dataset_dict,
                multiple=self.num_samples_per_task,
            )
            print(
                f"Detection camera {self.id}: real_samples={self.num_real_test_samples}, "
                f"total_samples={len(self.test_dataset_dict['samples'])}, tasks={self.num_tasks}"
            )
            
            self.prepared_label_info_by_task = {}
            self.db_dataset_dict = get_empty_detection_dataset_dict_from_source(self.train_dataset_dict)
    
    def get_task_dataset_dict(self, task_id):
        start_idx = self.num_samples_per_task * task_id
        end_idx = self.num_samples_per_task * (task_id+1)
        task_dataset_dict = get_detection_dataset_dict_with_sample_range_from_params(
            dataset_dict=self.test_dataset_dict,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        
        return task_dataset_dict
    
    def get_chunk_dataset_dict(self, task_id, chunk_id):
        task_start_idx = self.num_samples_per_task * task_id
        chunk_start_idx = task_start_idx + chunk_id*self.chunk_size
        chunk_end_idx = chunk_start_idx + self.chunk_size
        chunk_dataset_dict = get_detection_dataset_dict_with_sample_range_from_params(
            dataset_dict=self.test_dataset_dict,
            start_idx=chunk_start_idx,
            end_idx=chunk_end_idx,
        )
        
        return chunk_dataset_dict
    
    def get_selected_stream_dataset_dict_for_task(self, task_id, subsample_rate):
        if task_id > 0:
            previous_task_dataset_dict = self.get_task_dataset_dict(task_id=task_id-1)
            previous_samples = previous_task_dataset_dict["samples"]
            num_samples_to_pick = max(int(len(previous_samples)*subsample_rate), 1)
            num_samples_to_pick = min(num_samples_to_pick, len(previous_samples))
            rng = np.random.default_rng(RANDOM_SEED+task_id+self.camera_idx)
            selected_indexes = rng.choice(len(previous_samples), size=num_samples_to_pick, replace=False).tolist()
            selected_indexes = sorted(selected_indexes)
            selected_samples = [
                copy.deepcopy(previous_samples[index])
                for index in selected_indexes
            ]
            selected_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
                source_dataset_dict=previous_task_dataset_dict,
                samples=selected_samples,
            )
        else:
            selected_dataset_dict = get_empty_detection_dataset_dict_from_source(self.test_dataset_dict)
        
        return selected_dataset_dict
    
    def set_labeled_stream_dataset_dict_for_task(self, task_id, subsample_rate, labeled_stream_dataset_dict):
        if task_id in self.prepared_label_info_by_task:
            pass
        else:
            self.prepared_label_info_by_task[task_id] = {
                "subsample_rate": subsample_rate,
                "labeled_stream_dataset_dict": labeled_stream_dataset_dict,
                "db_snapshot_dataset_dict": copy.deepcopy(self.db_dataset_dict),
            }
            self.update_detection_db(new_dataset_dict=labeled_stream_dataset_dict)
    
    def update_detection_db(self, new_dataset_dict):
        pool_samples = self.db_dataset_dict["samples"] + new_dataset_dict["samples"]
        sample_by_key = {}
        ordered_keys = []
        for sample in pool_samples:
            sample_key = sample.get("img_save_path", sample.get("sample_id", None))
            if sample_key in sample_by_key:
                pass
            else:
                ordered_keys.append(sample_key)
            sample_by_key[sample_key] = copy.deepcopy(sample)
        
        deduped_samples = [
            sample_by_key[sample_key]
            for sample_key in ordered_keys
            if sample_key in sample_by_key
        ]
        if len(deduped_samples)>self.max_detection_exemplar_frames:
            capped_samples = deduped_samples[-self.max_detection_exemplar_frames:]
        else:
            capped_samples = deduped_samples
        self.db_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
            source_dataset_dict=new_dataset_dict,
            samples=capped_samples,
        )
    
    def get_labeled_train_val_dataset_dicts_from_task(self, task_id, subsample_rate):
        if task_id in self.prepared_label_info_by_task:
            label_info = self.prepared_label_info_by_task[task_id]
            labeled_stream_samples = label_info["labeled_stream_dataset_dict"]["samples"]
            db_samples = label_info["db_snapshot_dataset_dict"]["samples"]
            num_samples_to_pick = max(int(self.num_samples_per_task*subsample_rate), 1)
            
            if len(db_samples)>0:
                num_stream_target = min(len(labeled_stream_samples), max(1, num_samples_to_pick//2))
                num_db_target = min(len(db_samples), num_samples_to_pick-num_stream_target)
                shortfall = num_samples_to_pick - (num_stream_target+num_db_target)
                if shortfall > 0:
                    num_stream_target = min(len(labeled_stream_samples), num_stream_target+shortfall)
                else:
                    pass
            else:
                num_stream_target = min(len(labeled_stream_samples), num_samples_to_pick)
                num_db_target = 0
            
            selected_samples = [
                copy.deepcopy(sample)
                for sample in labeled_stream_samples[:num_stream_target]
            ]
            selected_samples.extend([
                copy.deepcopy(sample)
                for sample in db_samples[:num_db_target]
            ])
            rng = np.random.default_rng(RANDOM_SEED+task_id+self.camera_idx+7)
            if len(selected_samples)>1:
                shuffled_indexes = rng.permutation(len(selected_samples)).tolist()
                selected_samples = [
                    selected_samples[index]
                    for index in shuffled_indexes
                ]
            else:
                pass
            
            num_train = int(len(selected_samples)*self.train_split)
            if len(selected_samples)>1 and num_train>=len(selected_samples):
                num_train = len(selected_samples)-1
            elif len(selected_samples)>0 and num_train<1:
                num_train = 1
            else:
                pass
            
            train_samples = selected_samples[:num_train]
            val_samples = selected_samples[num_train:]
            train_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
                source_dataset_dict=label_info["labeled_stream_dataset_dict"],
                samples=train_samples,
            )
            val_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
                source_dataset_dict=label_info["labeled_stream_dataset_dict"],
                samples=val_samples,
            )
        else:
            message = f"Detection labels for camera:{self.id}, task:{task_id} were not prepared before dataloader creation"
            stop_sys(message, raise_error=True)
        
        return train_dataset_dict, val_dataset_dict
    
    def _get_dataloader(self,
                        task_id: int,
                        train_batch_size: int = 1,
                        test_batch_size: int = 1,
                        num_workers: int = 4,
                        subsample_rate: float = 1,
                        **kwargs):
        test_dataset_dict = self.get_task_dataset_dict(task_id=task_id)
        dataloaders_dict = {
            "train": None,
            "val": None,
            "test": get_detection_inference_dataloader_from_dataset_dict(
                dataset_dict=test_dataset_dict,
                batch_size=test_batch_size,
                num_workers=num_workers,
            ),
        }
        
        if task_id > 0:
            train_dataset_dict, val_dataset_dict = self.get_labeled_train_val_dataset_dicts_from_task(
                task_id=task_id,
                subsample_rate=subsample_rate,
            )
            dataloaders_dict["train"] = get_detection_train_dataloader_from_dataset_dict(
                dataset_dict=train_dataset_dict,
                batch_size=train_batch_size,
                num_workers=num_workers,
            )
            dataloaders_dict["val"] = get_detection_inference_dataloader_from_dataset_dict(
                dataset_dict=val_dataset_dict,
                batch_size=test_batch_size,
                num_workers=num_workers,
            )
        else:
            pass
        
        return dataloaders_dict
    
    def update_training_model(self, hyperparameters, gpu_fraction, restore_path=None, blocking=False):
        self.hyperparameters = hyperparameters
        self.training_gpu_fraction = gpu_fraction
        if hasattr(self, "training_model"):
            try:
                ray.kill(self.training_model, no_restart=True)
            except Exception:
                pass
            del self.training_model
        else:
            pass
        
        model_updated = False
        while not model_updated:
            try:
                gpu_idx = 0
                physical_gpu = logical_to_physical_gpu(gpu_idx)
                gpu_percent = int(self.training_gpu_fraction)
                self.training_model = RayMLDetectionModel.options(
                    name="{}_detection_training".format(self.id),
                    num_gpus=0.01,
                    resources={f"GPU{gpu_idx}": self.training_gpu_fraction/100},
                    runtime_env={
                        "env_vars": {
                            "CUDA_VISIBLE_DEVICES": physical_gpu,
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent),
                        },
                    },
                ).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.training_gpu_fraction,
                    restore_path=restore_path,
                    name="{}_detection_training".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
                    label_type=self.label_type,
                )
                model_updated = True
            except ValueError:
                pass
        
        if blocking:
            ray.get(self.training_model.ready.remote())
        else:
            pass
    
    def run_retraining(self,
                       hyperparameters,
                       gpu_fraction,
                       dataloaders_dict={},
                       validation_freq=-1,
                       restore_path=None,
                       profiling_mode=False,
                       task_num=None,
                       window_end_time=None):
        self.update_training_model(hyperparameters, gpu_fraction, restore_path=restore_path, blocking=False)
        if not dataloaders_dict:
            dataloaders_dict = self._get_dataloader(
                self.current_task,
                train_batch_size=hyperparameters["train_batch_size"],
                test_batch_size=hyperparameters["test_batch_size"],
                subsample_rate=hyperparameters["subsample"],
                num_workers=hyperparameters.get("num_workers", 4),
            )
        else:
            pass
        
        task = self.training_model.retrain_model.remote(
            dataloaders_dict["train"],
            dataloaders_dict["val"],
            dataloaders_dict["test"],
            hyperparameters,
            validation_freq,
            profiling_mode,
            task_num=task_num,
            window_end_time=window_end_time,
        )
        metadata = {key:{} for key in dataloaders_dict.keys()}
        
        return task, metadata
    
    def update_inference_model(self, hyperparameters, gpu_fraction, restore_path=None, blocking=False):
        self.hyperparameters = hyperparameters
        self.inference_gpu_fraction = gpu_fraction
        self.current_inference_path = restore_path if restore_path is not None else self.current_inference_path
        
        if hasattr(self, "inference_model"):
            try:
                ray.kill(self.inference_model, no_restart=True)
            except Exception:
                pass
            del self.inference_model
        else:
            pass
        
        model_updated = False
        while not model_updated:
            try:
                gpu_idx = 0
                physical_gpu = logical_to_physical_gpu(gpu_idx)
                gpu_percent = int(self.inference_gpu_fraction)
                self.inference_model = RayMLDetectionModel.options(
                    name="{}_detection_inference".format(self.id),
                    num_gpus=0.01,
                    resources={f"GPU{gpu_idx}": self.inference_gpu_fraction/100},
                    runtime_env={
                        "env_vars": {
                            "CUDA_VISIBLE_DEVICES": physical_gpu,
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent),
                        },
                    },
                ).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.inference_gpu_fraction,
                    restore_path=self.current_inference_path,
                    name="{}_detection_inference".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
                    label_type=self.label_type,
                )
                model_updated = True
            except ValueError:
                pass
        
        if blocking:
            ray.get(self.inference_model.ready.remote())
        else:
            pass
    
    def update_inference_from_retrained_model(self, path=None):
        if path:
            ray.get(self.training_model.save_model.remote(path))
            try:
                ray.kill(self.training_model, no_restart=True)
            except Exception:
                pass
            del self.training_model
            self.current_inference_path = path
            self.update_inference_model(
                self.hyperparameters,
                self.inference_gpu_fraction,
                restore_path=self.current_inference_path,
            )
        else:
            message = "Detection update_inference_from_retrained_model requires an explicit path"
            stop_sys(message, raise_error=True)
    
    def set_current_task(self, new_current_task):
        self.current_task = new_current_task
    
    def training_memory_footprint(self):
        return self.training_memory_footprint_value
    
    def inference_memory_footprint(self):
        return self.inference_memory_footprint_value
