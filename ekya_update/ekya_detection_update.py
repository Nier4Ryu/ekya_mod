import os, time, pandas as pd, ray
from ekya_update.camera_detection_update import CameraDetectionSubstitution
from ekya_update.common import check_mps_is_running, stop_sys
from ekya_update.detection_dataset_update import get_detection_inference_dataloader_from_dataset_dict
from ekya_update.logger_detection_update import LoggerDetectionSubstitution
from ekya_update.thief_detection_scheduler_update import ThiefDetectionSchedulerSubstitution


@ray.remote
class DetectionRetrainFinTracker:
    def __init__(self):
        self.status = {}
    
    def init(self, camera_ids):
        self.status = {camera_id:False for camera_id in camera_ids}
    
    def set(self, camera_id, value):
        self.status[camera_id] = value
    
    def all_done(self):
        all_status_done = all(self.status.values())
        return all_status_done


@ray.remote
def detection_inference_executor(camera: CameraDetectionSubstitution,
                                 num_chunks: int,
                                 retraining_period: int,
                                 test_batch_size: int,
                                 window_start_time: float,
                                 retrain_fin_tracker: ray.actor.ActorHandle = None,
                                 logger: ray.actor.ActorHandle = None,
                                 camera_idx=None,
                                 task_id=None,
                                 num_workers=4):
    remaining_time = retraining_period - (time.time()-window_start_time)
    time_per_chunk = remaining_time/num_chunks
    print(
        "[DetectionInferenceExecutor] camera {} task {} chunks {} time_per_chunk {:.2f}".format(
            camera.id,
            task_id,
            num_chunks,
            time_per_chunk,
        )
    )
    
    def fetch_actor():
        inference_model_actor = None
        while inference_model_actor is None:
            try:
                inference_model_actor = ray.get_actor("{}_detection_inference".format(camera.id))
            except ValueError:
                print("Got a dead detection inference actor, retrying..")
                time.sleep(0.5)
        return inference_model_actor
    
    chunk_maps = []
    for chunk_id in range(num_chunks):
        start_time = time.time()
        chunk_dataset_dict = camera.get_chunk_dataset_dict(task_id=task_id, chunk_id=chunk_id)
        chunk_loader = get_detection_inference_dataloader_from_dataset_dict(
            dataset_dict=chunk_dataset_dict,
            batch_size=test_batch_size,
            num_workers=num_workers,
        )
        inference_model_actor = fetch_actor()
        retry_count = 0
        chunk_map = None
        log_items_path = None
        while chunk_map is None and retry_count < 5:
            try:
                chunk_map, log_items_path = ray.get(inference_model_actor.test_map.remote(
                    test_loader=chunk_loader,
                    task_num=task_id,
                    chunk_id=chunk_id,
                ))
            except Exception as error:
                retry_count += 1
                print(
                    "[DetectionInferenceExecutor][WARNING] test_map failed with {}: {}. Retrying {}.".format(
                        type(error).__name__,
                        str(error)[:200],
                        retry_count,
                    )
                )
                inference_model_actor = fetch_actor()
                chunk_map = None
        
        print("[DetectionInferenceExecutor] camera {} chunk {} mAP {}".format(camera.id, chunk_id, chunk_map))
        chunk_maps.append(chunk_map)
        if logger and log_items_path:
            logger.log_inference_results.remote(
                camera_idx=camera_idx,
                task_id=task_id,
                chunk_id=chunk_id,
                log_items_path=log_items_path,
            )
        else:
            pass
        
        end_time = time.time()
        chunk_remaining_time = time_per_chunk - (end_time-start_time)
        if chunk_remaining_time > 0:
            all_retrain_done = retrain_fin_tracker is not None and ray.get(retrain_fin_tracker.all_done.remote())
            if all_retrain_done:
                time.sleep(0.1)
            else:
                time.sleep(chunk_remaining_time)
        else:
            print("Warning: Detection inference xput is less than line rate in chunk {}, camera {}".format(chunk_id, camera.id))
    
    return chunk_maps


class EkyaDetectionSubstitution(object):
    def __init__(self,
                 cameras,
                 start_task,
                 termination_task,
                 num_chunks,
                 model_load_path,
                 scheduler_name,
                 scheduler_kwargs,
                 num_gpus,
                 window_size,
                 log_dir,
                 resume_from_task=None):
        if not check_mps_is_running():
            message = "Ekya detection requires MPS server to be running! stopping sys!"
            stop_sys(message)
        else:
            self.cameras = cameras
            self.scheduler_name = scheduler_name
            self.model_load_path = model_load_path
            self.log_dir = os.path.join(log_dir, "logger_generated")
            self.logger = ray.remote(LoggerDetectionSubstitution).options(num_cpus=0).remote(base_dir=self.log_dir)
            
            if self.scheduler_name=="thief":
                self.scheduler = ThiefDetectionSchedulerSubstitution(
                    scheduler_kwargs=scheduler_kwargs,
                    model_load_path=self.model_load_path,
                    log_dir=self.log_dir,
                )
            else:
                message = f"Currently EkyaDetectionSubstitution does not support:{self.scheduler_name}"
                stop_sys(message, raise_error=True)
            
            self.retraining_period = window_size
            num_tasks_per_camera = [camera.num_tasks for camera in cameras]
            if len(set(num_tasks_per_camera))!=1:
                message = f"All detection cameras must have the same num_tasks, got:{num_tasks_per_camera}"
                stop_sys(message, raise_error=True)
            else:
                self.num_tasks = num_tasks_per_camera[0]
            
            self.num_inference_chunks = num_chunks
            if resume_from_task is not None:
                self.current_task = resume_from_task - 1
            else:
                self.current_task = start_task - 1
            
            if termination_task == -1:
                self.termination_task = self.num_tasks - 1
            else:
                self.termination_task = termination_task
            
            self.last_retraining_start_time = 0
            self.retrain_fin_tracker = DetectionRetrainFinTracker.options(num_cpus=0).remote()
            self.num_resources = num_gpus
    
    def get_latest_model_path_for_camera(self, camera_idx):
        if self.current_task==0:
            pretrained_model_path = self.model_load_path
        else:
            model_train_history_df_path = os.path.join(self.log_dir, str(camera_idx), "model_train_history.csv")
            if not os.path.exists(model_train_history_df_path):
                pretrained_model_path = self.model_load_path
            else:
                model_train_history_df = pd.read_csv(model_train_history_df_path)
                weight_save_path = model_train_history_df["weight_save_path"].iloc[-1]
                prev_task_num = model_train_history_df["task_num"].iloc[-1]
                if prev_task_num > self.current_task:
                    message = "There was an error in detection model_train_history_logging, fix this issue!"
                    stop_sys(message, raise_error=True)
                else:
                    pretrained_model_path = weight_save_path
        
        return pretrained_model_path
    
    def update_inference_jobs(self, hyperparameters, gpu_inference_fractions, blocking=False):
        for camera_idx, camera in enumerate(self.cameras):
            this_gpu_fraction = gpu_inference_fractions.get(camera.id, 0)
            if this_gpu_fraction > 0:
                pretrained_model_path = self.get_latest_model_path_for_camera(camera_idx=camera_idx)
                camera.update_inference_model(
                    hyperparameters[camera.id],
                    this_gpu_fraction,
                    restore_path=pretrained_model_path,
                    blocking=blocking,
                )
                if camera.id not in self.inference_tasks:
                    self.inference_tasks[camera.id] = detection_inference_executor.remote(
                        camera,
                        num_chunks=self.num_inference_chunks,
                        retraining_period=self.retraining_period,
                        test_batch_size=hyperparameters[camera.id]["test_batch_size"],
                        window_start_time=self.last_retraining_start_time,
                        retrain_fin_tracker=self.retrain_fin_tracker,
                        logger=self.logger,
                        camera_idx=camera_idx,
                        task_id=self.current_task,
                        num_workers=hyperparameters[camera.id].get("num_workers", 4),
                    )
                else:
                    pass
            else:
                print("[WARN][Detection Task {}] Camera {} has no inference resources.".format(camera.current_task, camera.id))
    
    def launch_training_jobs(self, hyperparameters, gpu_training_fractions):
        if self.current_task > 0:
            for camera_idx, camera in enumerate(self.cameras):
                this_gpu_fraction = gpu_training_fractions.get(camera.id, 0)
                if this_gpu_fraction > 0:
                    pretrained_model_path = self.get_latest_model_path_for_camera(camera_idx=camera_idx)
                    window_end_time = self.last_retraining_start_time + self.retraining_period
                    self.retraining_tasks[camera.id], self.retraining_metadata[camera.id] = camera.run_retraining(
                        hyperparameters[camera.id],
                        this_gpu_fraction,
                        dataloaders_dict={},
                        validation_freq=hyperparameters[camera.id].get("validation_freq", -1),
                        restore_path=pretrained_model_path,
                        profiling_mode=False,
                        task_num=self.current_task,
                        window_end_time=window_end_time,
                    )
                    ray.get(self.retrain_fin_tracker.set.remote(camera.id, False))
                    print("[DETECTION TRAINING START][Task {}] Camera {} hp_id={}, epochs={}".format(
                        self.current_task,
                        camera.id,
                        hyperparameters[camera.id]["id"],
                        hyperparameters[camera.id]["epochs"],
                    ))
                else:
                    ray.get(self.retrain_fin_tracker.set.remote(camera.id, True))
                    print("[Detection Task {}] Camera {} was assigned no retraining resources.".format(camera.current_task, camera.id))
        else:
            for camera in self.cameras:
                ray.get(self.retrain_fin_tracker.set.remote(camera.id, True))
    
    def check_task_loop(self):
        print("Starting detection check task loop.")
        running_retraining_tasks = list(self.retraining_tasks.values())
        while True:
            remaining_time = self.retraining_period - (time.time()-self.last_retraining_start_time)
            if remaining_time > 0:
                if len(running_retraining_tasks)>0:
                    done_tasks, running_retraining_tasks = ray.wait(running_retraining_tasks, timeout=0)
                    retraining_results = ray.get(done_tasks)
                else:
                    done_tasks = []
                    retraining_results = []
                
                done_camera_ids = [
                    next(camera_id for camera_id, task_id in self.retraining_tasks.items() if task_id==done_task)
                    for done_task in done_tasks
                ]
                done_cameras = [next(camera for camera in self.cameras if camera.id==camera_id) for camera_id in done_camera_ids]
                
                for done_camera_idx, done_camera in enumerate(done_cameras):
                    done_best_val_map = retraining_results[done_camera_idx][0]
                    ray.get(self.retrain_fin_tracker.set.remote(done_camera.id, True))
                    print("[DETECTION TRAINING FIN][Task {}] Camera {} best_val_map={:.4f}".format(
                        self.current_task,
                        done_camera.id,
                        done_best_val_map,
                    ))
                
                if self.logger:
                    log_time = time.time()
                    for result_idx, retraining_result in enumerate(retraining_results):
                        best_val_map, profile, subprofile_test_results, profile_preretrain_test_map, profile_test_map, misc_results = retraining_result
                        retraining_time_taken = log_time - self.last_retraining_start_time
                        camera = done_cameras[result_idx]
                        hp_id = self.current_hyperparameters[camera.id]["id"]
                        hp_epochs = self.current_hyperparameters[camera.id]["epochs"]
                        camera_idx = int(camera.id)
                        log_items = {
                            "camera_idx": [camera_idx],
                            "task_id": [self.current_task],
                            "retraining_time_taken": [retraining_time_taken],
                            "time_left_after_retraining": [self.retraining_period-retraining_time_taken],
                            "actual_training_time": [misc_results["total_time"]],
                            "per_epoch_avg_time": [misc_results["per_epoch_avg_time"]],
                            "best_val_map": [best_val_map],
                            "profile_preretrain_test_map": [profile_preretrain_test_map],
                            "profile_test_map": [profile_test_map],
                            "hp_id": [hp_id],
                            "hp_epochs": [hp_epochs],
                            "log_time": [log_time],
                        }
                        self.logger.log_retraining_results.remote(
                            camera_idx=camera_idx,
                            task_id=self.current_task,
                            log_items=log_items,
                        )
                else:
                    pass
                
                for done_camera in done_cameras:
                    self.inference_resource_weights, self.training_resource_weights = self.scheduler.reallocation_callback(
                        done_camera.id,
                        self.inference_resource_weights,
                        self.training_resource_weights,
                    )
                    gpu_inference_fractions = self.inference_resource_weights
                    if getattr(done_camera, "inference_gpu_fraction", 0) > 0:
                        pretrained_model_path = self.get_latest_model_path_for_camera(camera_idx=int(done_camera.id))
                        done_camera.update_inference_from_retrained_model(path=pretrained_model_path)
                    else:
                        pass
                    
                    for camera in self.cameras:
                        if camera.id != done_camera.id and gpu_inference_fractions.get(camera.id, 0)>0:
                            camera.update_inference_model(
                                self.current_hyperparameters[camera.id],
                                gpu_inference_fractions[camera.id],
                                blocking=False,
                            )
                        else:
                            pass
                
                if len(self.inference_tasks)>0:
                    done_inference, remaining_inference = ray.wait(
                        list(self.inference_tasks.values()),
                        num_returns=len(self.inference_tasks),
                        timeout=0,
                    )
                    if len(done_inference)==len(self.inference_tasks):
                        print("[DETECTION CHECK TASK LOOP] All inference tasks completed. Breaking early.")
                        return
                    else:
                        pass
                else:
                    pass
                time.sleep(1)
            else:
                return
    
    def run(self):
        try:
            done = False
            while not done:
                self.last_retraining_start_time = time.time()
                self.new_task_callback()
                self.check_task_loop()
                self.stop_current_jobs()
                if self.current_task == self.termination_task:
                    done = True
                else:
                    pass
        finally:
            self._emergency_cleanup()
    
    def new_task_callback(self):
        self.current_task += 1
        print("Detection new task callback called. Now on task {}/{}".format(self.current_task, self.num_tasks))
        for camera in self.cameras:
            camera.set_current_task(self.current_task)
        
        self.inference_tasks = {}
        self.retraining_tasks = {}
        self.retraining_metadata = {}
        self.inference_resource_weights = {}
        self.training_resource_weights = {}
        ray.get(self.retrain_fin_tracker.init.remote([camera.id for camera in self.cameras]))
        
        self.inference_resource_weights, self.current_hyperparameters = self.scheduler.get_inference_schedule(
            self.cameras,
            self.num_resources,
            task_id=self.current_task,
        )
        self.update_inference_jobs(self.current_hyperparameters, self.inference_resource_weights)
        
        custom_state = {
            "task_id": self.current_task,
            "retraining_period": self.retraining_period,
            "window_start_time": self.last_retraining_start_time,
            "retrain_fin_tracker": self.retrain_fin_tracker,
        }
        prev_hyperparameters = self.current_hyperparameters
        schedule_result = self.scheduler.get_schedule(self.cameras, self.num_resources, custom_state)
        self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters = schedule_result
        for camera in self.cameras:
            if camera.id not in self.current_hyperparameters:
                self.current_hyperparameters[camera.id] = prev_hyperparameters[camera.id]
            else:
                pass
        
        self.log_schedules(
            self.current_task,
            self.inference_resource_weights,
            self.training_resource_weights,
            self.current_hyperparameters,
        )
        self.update_inference_jobs(self.current_hyperparameters, self.inference_resource_weights)
        self.launch_training_jobs(self.current_hyperparameters, self.training_resource_weights)
    
    def log_schedules(self, task_id, inference_resource_weights, training_resource_weights, current_hyperparameters):
        self.logger.log_schedules.remote(
            task_id=task_id,
            inference_resource_weights=inference_resource_weights,
            training_resource_weights=training_resource_weights,
            current_hyperparameters=current_hyperparameters,
        )
    
    def stop_current_jobs(self):
        print("Stopping all detection jobs.")
        for task in self.inference_tasks.values():
            ray.cancel(task, force=True)
        for camera in self.cameras:
            if hasattr(camera, "training_model"):
                ray.kill(camera.training_model)
                del camera.training_model
            else:
                pass
            if hasattr(camera, "inference_model"):
                ray.kill(camera.inference_model)
                del camera.inference_model
            else:
                pass
    
    def _emergency_cleanup(self):
        print("[EkyaDetection] Running emergency cleanup.")
        try:
            self.stop_current_jobs()
        except Exception as error:
            print(f"[EkyaDetection] stop_current_jobs failed during cleanup: {error}")
        try:
            ray.kill(self.logger)
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception as error:
            print(f"[EkyaDetection] ray.shutdown failed: {error}")
    
    def shutdown(self, fin_file_path):
        with open(fin_file_path, "w") as file:
            file.write("done\n")
