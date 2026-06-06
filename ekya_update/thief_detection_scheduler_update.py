import os, time, numpy as np, pandas as pd, ray
from typing import List
from ekya.simulation.camera import generate_training_job
from ekya.simulation.schedulers import thief_sco_scheduler
from ekya.simulation.jobs import InferenceJob as SimInferenceJob
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.microprofilers.modelling_funcs import get_scaled_optimus_fn, get_linear_fn
from ekya_update.camera_detection_update import CameraDetectionSubstitution
from ekya_update.common import atomic_to_csv, logical_to_physical_gpu, stop_sys
from ekya_update.model_detection_update import RayMLDetectionModel
from ekya_update.simple_detection_microprofiler_update import SimpleDetectionMicroprofilerSubstitution


class ThiefDetectionSchedulerSubstitution(BaseScheduler):
    def __init__(self, scheduler_kwargs, model_load_path, log_dir):
        self.scheduler_kwargs = scheduler_kwargs
        self.inference_profile = pd.read_csv(self.scheduler_kwargs["inference_profile_path"])
        self.microprofile_device = self.scheduler_kwargs["microprofile_device"]
        self.microprofile_resources_per_trial = self.scheduler_kwargs["microprofile_resources_per_trial"]
        self.microprofile_epochs = self.scheduler_kwargs["microprofile_epochs"]
        self.microprofile_subsample_rate = self.scheduler_kwargs["microprofile_subsample_rate"]
        self.profiling_epochs = np.array(self.scheduler_kwargs["profiling_epochs"])
        self.default_hyperparams = self.scheduler_kwargs["hyperparams"]["0"]
        self.hyperparameters = self.scheduler_kwargs["hyperparams"]
        self.predmodel_acc_args = self.scheduler_kwargs["predmodel_acc_args"]
        self.measured_time_per_epoch_for_hyperparams = self.scheduler_kwargs["measured_time_per_epoch_for_hyperparams"]
        self.measured_inittime_for_hyperparams = self.scheduler_kwargs["measured_inittime_for_hyperparams"]
        self.steal_increment = self.scheduler_kwargs["steal_increment"]
        self.teacher_hyperparameters = self.scheduler_kwargs["teacher_hyperparameters"]
        self.teacher_labeling_batch_size = self.scheduler_kwargs.get("teacher_labeling_batch_size", 1)
        self.teacher_num_workers = self.scheduler_kwargs.get("teacher_num_workers", 4)
        self.model_load_path = model_load_path
        self.log_dir = log_dir
    
    def prepare_teacher_labels_for_task(self, cameras, task_id):
        start_time = time.time()
        max_hp_subsample = max(float(hp["subsample"]) for hp in self.hyperparameters.values())
        gpu_resources = self.microprofile_resources_per_trial
        
        for camera in cameras:
            selected_dataset_dict = camera.get_selected_stream_dataset_dict_for_task(
                task_id=task_id,
                subsample_rate=max_hp_subsample,
            )
            if len(selected_dataset_dict["samples"])>0:
                labeled_dataset_dict, metrics = self.run_teacher_inference_for_dataset_dict(
                    camera=camera,
                    task_id=task_id,
                    dataset_dict=selected_dataset_dict,
                    gpu_resources=gpu_resources,
                )
            else:
                labeled_dataset_dict = selected_dataset_dict
            camera.set_labeled_stream_dataset_dict_for_task(
                task_id=task_id,
                subsample_rate=max_hp_subsample,
                labeled_stream_dataset_dict=labeled_dataset_dict,
            )
        
        labeling_time = time.time() - start_time
        return labeling_time
    
    def run_teacher_inference_for_dataset_dict(self, camera, task_id, dataset_dict, gpu_resources):
        gpu_idx = 0
        physical_gpu = logical_to_physical_gpu(gpu_idx)
        gpu_percent = int(gpu_resources*100)
        actor_name = f"{camera.id}_detection_teacher_{task_id}_{time.time_ns()}"
        teacher_actor = RayMLDetectionModel.options(
            name=actor_name,
            num_gpus=0.01,
            resources={f"GPU{gpu_idx}": gpu_resources},
            runtime_env={
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": physical_gpu,
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent),
                },
            },
        ).remote(
            hyperparameters=self.teacher_hyperparameters,
            gpu_allocation_percentage=gpu_percent,
            restore_path=self.teacher_hyperparameters.get("weight_path", None),
            name=actor_name,
            camera_idx=camera.camera_idx,
            log_dir=self.log_dir,
            label_type="ground_truth",
        )
        labeled_dataset_dict, metrics = ray.get(teacher_actor.infer_dataset_dict.remote(
            dataset_dict=dataset_dict,
            batch_size=self.teacher_labeling_batch_size,
            num_workers=self.teacher_num_workers,
            score_threshold=self.teacher_hyperparameters.get("score_threshold", None),
        ))
        ray.kill(teacher_actor, no_restart=True)
        
        return labeled_dataset_dict, metrics
    
    def generate_profiles(self, cameras, microprofile_results, default_inference_maps):
        profiles = {}
        unsuccessful_models = 0
        for camera in cameras:
            camera_profiles = []
            for hp_result, default_map in zip(microprofile_results[camera.id], default_inference_maps[camera.id]):
                test_map = hp_result["test_acc"]
                hyperparameters = hp_result["hyperparameters"]
                try:
                    microprofile_accuracy_model = get_scaled_optimus_fn(
                        microprofile_x=np.array([self.microprofile_epochs]),
                        microprofile_y=np.array([test_map]),
                        start_acc=default_map,
                        **self.predmodel_acc_args,
                    )
                except RuntimeError:
                    unsuccessful_models += 1
                    microprofile_accuracy_model = lambda x: default_map*np.ones_like(x)
                
                time_per_epoch = self.measured_time_per_epoch_for_hyperparams[hyperparameters["id"]]
                init_time = self.measured_inittime_for_hyperparams[hyperparameters["id"]]
                microprofile_runtime_model = get_linear_fn(a=time_per_epoch, b=init_time)
                map_predictions = microprofile_accuracy_model(self.profiling_epochs)
                runtime_predictions = microprofile_runtime_model(self.profiling_epochs)
                for map_prediction, runtime_prediction, epochs in zip(map_predictions, runtime_predictions, self.profiling_epochs):
                    hp_temp = hyperparameters.copy()
                    hp_temp["epochs"] = int(epochs)
                    camera_profiles.append([hp_temp, map_prediction, runtime_prediction, int(epochs), default_map])
            profiles[camera.id] = camera_profiles
        
        if unsuccessful_models:
            print(f"[THIEF DETECTION SCHEDULER][WARN] Failed to generate models for {unsuccessful_models} cameras.")
        else:
            pass
        
        return profiles
    
    def execute_microprofiling(self, cameras, task_id):
        hyp_list = list(self.hyperparameters.values())
        microprofiler = SimpleDetectionMicroprofilerSubstitution(device=self.microprofile_device)
        microprofile_results = {}
        for camera_idx, camera in enumerate(cameras):
            dataloaders = [
                camera._get_dataloader(
                    task_id=task_id,
                    train_batch_size=hp["train_batch_size"],
                    test_batch_size=hp["test_batch_size"],
                    subsample_rate=hp["subsample"],
                    num_workers=hp.get("num_workers", 4),
                )
                for hp in hyp_list
            ]
            pretrained_model_path = self.get_latest_model_path_for_camera(camera_idx=camera_idx, task_id=task_id)
            best_result, results = microprofiler.run_microprofiling(
                candidate_hyperparams=hyp_list,
                dataloaders=dataloaders,
                resources=self.microprofile_resources_per_trial,
                epochs=self.microprofile_epochs,
                pretrained_model_path=pretrained_model_path,
                subsample_rate=self.microprofile_subsample_rate,
                task_num=task_id,
                camera_idx=camera_idx,
                log_dir=self.log_dir,
                model_name=hyp_list[0]["feature_extractor_name"],
            )
            microprofile_results[camera.id] = results
            microprofiler.cleanup()
        
        return microprofile_results
    
    def get_latest_model_path_for_camera(self, camera_idx, task_id):
        if task_id==0:
            pretrained_model_path = self.model_load_path
        else:
            model_train_history_df_path = os.path.join(self.log_dir, str(camera_idx), "model_train_history.csv")
            if not os.path.exists(model_train_history_df_path):
                pretrained_model_path = self.model_load_path
            else:
                model_train_history_df = pd.read_csv(model_train_history_df_path)
                weight_save_path = model_train_history_df["weight_save_path"].iloc[-1]
                prev_task_num = model_train_history_df["task_num"].iloc[-1]
                if prev_task_num > task_id:
                    message = "There was an error in detection model_train_history_logging, fix this issue!"
                    stop_sys(message, raise_error=True)
                else:
                    pretrained_model_path = weight_save_path
        
        return pretrained_model_path
    
    def get_schedule(self, cameras: List[CameraDetectionSubstitution], resources: float, state: dict):
        task_id = state["task_id"]
        retraining_period = state["retraining_period"]
        window_start_time = state["window_start_time"]
        retrain_fin_tracker = state.get("retrain_fin_tracker", None)
        
        if task_id==0:
            inference_resource_weights, hyperparameters = self.get_inference_schedule(cameras, resources, task_id=task_id)
            training_resource_weights = {camera.id:0 for camera in cameras}
            schedule_result = inference_resource_weights, training_resource_weights, hyperparameters
            if retrain_fin_tracker is not None:
                for camera in cameras:
                    ray.get(retrain_fin_tracker.set.remote(camera.id, True))
            else:
                pass
        else:
            pipeline_start_time = time.time()
            teacher_labeling_time = self.prepare_teacher_labels_for_task(cameras=cameras, task_id=task_id)
            microprofile_start_time = time.time()
            microprofile_results = self.execute_microprofiling(cameras=cameras, task_id=task_id)
            default_inference_maps = {
                camera_id:[hp_result["preretrain_test_acc"] for hp_result in result]
                for camera_id, result in microprofile_results.items()
            }
            profiles = self.generate_profiles(cameras, microprofile_results, default_inference_maps)
            microprofile_time_taken = time.time() - microprofile_start_time
            
            SimInferenceJobs = {}
            for camera in cameras:
                SimInferenceJobs[camera.id] = SimInferenceJob(
                    f"{camera.id}_inference",
                    default_inference_maps[camera.id][0],
                    self.default_hyperparams["feature_extractor_name"],
                    self.inference_profile["subsampling"],
                    self.inference_profile["c1"],
                    resource_alloc=0,
                )
            
            SimTrainingCfgs = {}
            for camera in cameras:
                SimTrainingCfgs[camera.id] = []
                for hp, map_prediction, runtime_prediction, epochs, preretrain_map in profiles[camera.id]:
                    if map_prediction > preretrain_map:
                        job = generate_training_job(
                            f"{camera.id}_train_{hp['id']}_{epochs}",
                            map_prediction,
                            runtime_prediction,
                            preretrain_map,
                            model_name=hp["feature_extractor_name"],
                            inference_job=SimInferenceJobs[camera.id],
                            oracle=False,
                        )
                        SimTrainingCfgs[camera.id].append(job)
                    else:
                        pass
            
            sched_job_pairs = [[SimInferenceJobs[camera.id], SimTrainingCfgs[camera.id]] for camera in cameras]
            elapsed_so_far = time.time() - pipeline_start_time
            remaining_time = int(retraining_period - elapsed_so_far)
            if remaining_time > 0:
                schedule = thief_sco_scheduler(
                    sched_job_pairs,
                    resources,
                    remaining_time,
                    iterations=3,
                    steal_increment=self.steal_increment,
                )
                init_schedule = schedule[0]
                print("[THIEF DETECTION SCHEDULER] Schedule from thief scheduler: {}".format(init_schedule))
                schedule_result = self.extract_ekya_schedule(init_schedule, self.hyperparameters)
                self.log_detection_scheduler_outputs(
                    task_id=task_id,
                    window_start_time=window_start_time,
                    teacher_labeling_time=teacher_labeling_time,
                    microprofile_time_taken=microprofile_time_taken,
                    total_pipeline_time=time.time()-pipeline_start_time,
                    retraining_period=retraining_period,
                    microprofile_results=microprofile_results,
                    profiles=profiles,
                    schedule_result=schedule_result,
                    raw_schedule=init_schedule,
                    cameras=cameras,
                )
            else:
                message = (
                    f"Detection teacher labeling + microprofiling consumed the retraining window. "
                    f"Elapsed:{elapsed_so_far:.2f}s, period:{retraining_period}s"
                )
                stop_sys(message, raise_error=True)
        
        return schedule_result
    
    def log_detection_scheduler_outputs(self,
                                        task_id,
                                        window_start_time,
                                        teacher_labeling_time,
                                        microprofile_time_taken,
                                        total_pipeline_time,
                                        retraining_period,
                                        microprofile_results,
                                        profiles,
                                        schedule_result,
                                        raw_schedule,
                                        cameras):
        log_path = os.path.join(self.log_dir, "micro_profiling", f"detection_micro_profiling_fin_task_{task_id}.csv")
        log_row = {
            "task_id": task_id,
            "window_start_time": window_start_time,
            "teacher_labeling_time": teacher_labeling_time,
            "microprofile_time": microprofile_time_taken,
            "total_pipeline_time": total_pipeline_time,
            "remaining_for_retraining": retraining_period-total_pipeline_time,
        }
        atomic_to_csv(pd.DataFrame([log_row]), path=log_path)
        
        microprofile_rows = []
        for camera in cameras:
            for hp_result in microprofile_results[camera.id]:
                microprofile_rows.append({
                    "task_id": task_id,
                    "camera_id": camera.id,
                    "hp_id": hp_result["hyperparameters"]["id"],
                    "preretrain_val_map": hp_result["preretrain_test_acc"],
                    "microprofile_val_map": hp_result["test_acc"],
                    "time_per_epoch": hp_result["time_per_epoch"],
                    "init_time": hp_result["init_time"],
                })
        microprofile_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"detection_microprofiling_task_{task_id}.csv")
        atomic_to_csv(pd.DataFrame(microprofile_rows), path=microprofile_log_path)
        
        profile_rows = []
        for camera in cameras:
            for hp, map_pred, runtime_pred, epochs, preretrain_map in profiles[camera.id]:
                profile_rows.append({
                    "task_id": task_id,
                    "camera_id": camera.id,
                    "hp_id": hp["id"],
                    "epochs": epochs,
                    "predicted_map": map_pred,
                    "predicted_runtime": runtime_pred,
                    "preretrain_map": preretrain_map,
                    "map_gain": map_pred-preretrain_map,
                })
        profile_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"detection_profiles_task_{task_id}.csv")
        atomic_to_csv(pd.DataFrame(profile_rows), path=profile_log_path)
        
        inference_resource_weights, training_resource_weights, chosen_hyperparameters = schedule_result
        decision_rows = []
        for camera in cameras:
            row = {
                "task_id": task_id,
                "camera_id": camera.id,
                "inference_resource_pct": inference_resource_weights.get(camera.id, 0),
                "training_resource_pct": training_resource_weights.get(camera.id, 0),
            }
            if camera.id in chosen_hyperparameters:
                hp = chosen_hyperparameters[camera.id]
                row["chosen_hp_id"] = hp["id"]
                row["chosen_epochs"] = hp["epochs"]
                row["chosen_subsample"] = hp["subsample"]
                row["chosen_lr"] = hp["learning_rate"]
            else:
                row["chosen_hp_id"] = None
                row["chosen_epochs"] = None
                row["chosen_subsample"] = None
                row["chosen_lr"] = None
            decision_rows.append(row)
        decision_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"detection_decisions_task_{task_id}.csv")
        atomic_to_csv(pd.DataFrame(decision_rows), path=decision_log_path)
        
        raw_schedule_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"detection_raw_schedule_task_{task_id}.csv")
        raw_rows = [{"task_id": task_id, "job": key, "resource_weight": value} for key, value in raw_schedule.items()]
        atomic_to_csv(pd.DataFrame(raw_rows), path=raw_schedule_log_path)
    
    @staticmethod
    def extract_ekya_schedule(schedule, hyperparameter_map):
        inference_resource_weights = {}
        training_resource_weights = {}
        hyperparameters = {}
        for job_string, weight in schedule.items():
            components = job_string.split("_")
            if "inference" in job_string:
                camera_id = "_".join(components[0:-1])
                inference_resource_weights[camera_id] = weight*100
            elif "train" in job_string:
                epochs = components[-1]
                hp_id = components[-2]
                camera_id = "_".join(components[0:-3])
                hps = hyperparameter_map[hp_id].copy()
                hps["epochs"] = int(epochs)
                hyperparameters[camera_id] = hps
                training_resource_weights[camera_id] = weight*100
            else:
                message = f"Invalid detection scheduler job string:{job_string}"
                stop_sys(message, raise_error=True)
        
        return inference_resource_weights, training_resource_weights, hyperparameters
    
    def get_inference_schedule(self, cameras, resources, task_id=0):
        if task_id > 0:
            total_min_inference = sum(camera.min_inference_resources_to_meet_slo for camera in cameras)
            total_required = total_min_inference + self.microprofile_resources_per_trial
            if total_required > resources:
                message = (
                    f"Cannot fit detection minimum inference ({total_min_inference}) + "
                    f"microprofiling ({self.microprofile_resources_per_trial}) within resources ({resources})."
                )
                stop_sys(message, raise_error=True)
            else:
                leftover = resources - self.microprofile_resources_per_trial - total_min_inference
                per_camera_bonus = leftover / len(cameras) if leftover > 0 else 0
                inference_resource_weights = {
                    camera.id:(camera.min_inference_resources_to_meet_slo+per_camera_bonus)*100
                    for camera in cameras
                }
        else:
            inference_resource_weights = {camera.id:(resources/len(cameras))*100 for camera in cameras}
        
        hyperparameters = {camera.id:self.default_hyperparams for camera in cameras}
        return inference_resource_weights, hyperparameters
    
    def reallocation_callback(self, completed_camera_name, inference_resource_weights, training_resources_weights):
        reallocated_weights = fair_reallocation(
            completed_camera_name,
            inference_resource_weights,
            training_resources_weights,
        )
        return reallocated_weights
