import ray, os, time, pandas as pd
from typing import List
from torch.utils.data import DataLoader
from ekya_update.camera_update import CameraSubstitution
from ekya_update.logger_update import LoggerSubstitution
from ekya_update.thief_scheduler_update import ThiefSchedulerSubstitution
from ekya_update.common import stop_sys, check_mps_is_running, logical_to_physical_gpu


@ray.remote
class RetrainFinTracker:
    def __init__(self):
        self.status = {}

    def init(self, camera_ids):
        self.status = {cid: False for cid in camera_ids}

    def set(self, camera_id, value):
        self.status[camera_id] = value

    def all_done(self):
        return all(self.status.values())


@ray.remote
def inference_executor(camera: CameraSubstitution,
                       num_chunks: int,
                       retraining_period: int,
                       test_batch_size: int,
                       window_start_time: float,
                       model_name: str = None,
                       last_layer_only = None,
                       retrain_fin_tracker: ray.actor.ActorHandle = None,
                       logger: ray.actor.ActorHandle = None,
                       camera_idx=None,
                       task_id=None,
                       ) -> None:
    # WARNING: Camera object is frozen copy at method invocation. Updates to camera wont be reflected in the remote function
    # Parse dataloader and break it into num_chunks
    test_loader = camera._get_dataloader(camera.current_task, test_batch_size=test_batch_size, model_name=model_name, last_layer_only=last_layer_only)["test"]
    test_dataset = test_loader.dataset
    index = test_dataset.get_indexes()
    chunk_size = len(test_dataset)//num_chunks
    remaining_time = retraining_period - (time.time() - window_start_time)
    time_per_chunk = remaining_time/num_chunks
    print("[InferenceExecutor] Initializing inference executor for camera {}. Total samples in task: {}. Chunk size: "
          "{}. Time per chunk: {:.2f} (remaining: {:.2f}s of {}s window)".format(
              camera.id, len(test_dataset), chunk_size, time_per_chunk, remaining_time, retraining_period))

    test_accs = []

    def fetch_actor():
        inference_model_actor = None
        while inference_model_actor is None:
            try:
                inference_model_actor = ray.get_actor("{}_inference".format(camera.id))
            except ValueError:
                print("Got a dead actor, retrying..")
                time.sleep(0.5)
        return inference_model_actor

    #Iterate over each chunk and run it's inference
    for chunk_id in range(num_chunks):
        print(f"Inference for camera:{camera.id} {chunk_id}")
        start_time = time.time()
        chunk_idxs = index[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
        chunk_dataset = test_dataset.get_filtered_dataset(chunk_idxs)
        chunk_loader = DataLoader(chunk_dataset,
                                              batch_size=test_loader.batch_size,
                                              shuffle=False,
                                              num_workers=0)
        # Try to fetch the latest actor handle from ray until successful
        inference_model_actor = fetch_actor()
        retry_count = 0
        test_acc = None
        log_items_path = None
        while test_acc is None and retry_count < 5:
            try:
                test_acc, log_items_path = ray.get(inference_model_actor.test_acc.remote(test_loader=chunk_loader))
            except Exception as e:
                retry_count+=1
                print("[InferenceExecutor][WARNING] Test_acc task failed with {}: {}. Retrying. Attempt: {}".format(type(e).__name__, str(e)[:200], retry_count))
                inference_model_actor = fetch_actor()
                test_acc = None
        print("[InferenceExecutor] {}: Got test acc: {}".format(camera.id, test_acc))
        test_accs.append(test_acc)

        # Result logging
        if logger and log_items_path:
            logger.log_inference_results.remote(
                camera_idx=camera_idx,
                task_id=task_id,
                chunk_id=chunk_id,
                log_items_path=log_items_path,
            )
        print(f"Inference for camera:{camera.id} {chunk_id} done + loggign should be done")

        end_time = time.time()
        chunk_remaining_time = time_per_chunk - (end_time - start_time)
        if chunk_remaining_time > 0:
            all_retrain_done = retrain_fin_tracker is not None and ray.get(retrain_fin_tracker.all_done.remote())
            if all_retrain_done:
                print("Chunk {}/{} done. All retraining finished, sleeping 0.1s".format(chunk_id, num_chunks))
                time.sleep(0.1)
            else:
                print("Chunk {}/{} done. Sleeping for {:.2f}".format(chunk_id, num_chunks, chunk_remaining_time))
                time.sleep(chunk_remaining_time)
        else:
            # TODO: Log warning: Throughput is less than line rate.
            print("Warning: Inference xput is less than line rate in chunk {}, camera {}".format(chunk_id, camera.id))
    return test_accs

class EkyaSubstitution(object):
    # Updated some part at least
    def __init__(self,
                #  cameras: List[CameraSubstitution],
                #  scheduler_name: str,
                #  retraining_period: int,
                #  num_tasks: int,
                #  num_inference_chunks: int,
                #  num_resources: float,
                #  pretrained_model_dir:str,
                #  log_dir: str = "",
                #  scheduler_kwargs: dict = {},
                #  starting_task: int = 0,
                #  termination_task: int = -1,
                #  profiling_mode: bool = False,
                #  profile_write_path: str = '',
                #  gpu_memory: float = None
                
                # Data Related
                cameras,
                start_task,
                termination_task,
                num_chunks,

                # Model Related
                model_load_path,
                
                # Training Related
                
                # Scheduler Related
                scheduler_name,
                scheduler_kwargs,
                
                # Resource Related
                num_gpus,
                window_size,
                log_dir,

                # Resume Related
                resume_from_task=None,
        ):
        if not check_mps_is_running():
            message = "Ekya requires MPS server to be running! stopping sys!"
            stop_sys(message)
        else:
            # Default Params setup
            self.cameras = cameras
            self.scheduler_name = scheduler_name

            # New params setup
            self.model_load_path = model_load_path

            # Logger Setup
            self.log_dir = os.path.join(log_dir, "logger_generated")
            self.logger = ray.remote(LoggerSubstitution).options(num_cpus=0).remote(base_dir=self.log_dir)

            if self.scheduler_name=="thief":
                self.scheduler = ThiefSchedulerSubstitution(
                    scheduler_kwargs=scheduler_kwargs,
                    model_load_path = self.model_load_path,
                    log_dir = self.log_dir,
                )
            else:
                message = f"Currently EkyaSubstitution does not support:{self.scheduler_name}"
                stop_sys(message, raise_error=True)

            self.retraining_period = window_size
            # Derive num_tasks from cameras and verify all streams agree
            num_tasks_per_camera = [camera.num_tasks for camera in cameras]
            if len(set(num_tasks_per_camera)) != 1:
                message = (f"All cameras must have the same num_tasks derived from fps/slo/df_length, "
                           f"but got: {num_tasks_per_camera}")
                stop_sys(message, raise_error=True)
            else:
                self.num_tasks = num_tasks_per_camera[0]
            self.num_inference_chunks = num_chunks
            if resume_from_task is not None:
                self.current_task = resume_from_task - 1
                print(f"[RESUME] EkyaSubstitution initialized with resume_from_task={resume_from_task}, "
                      f"current_task set to {self.current_task}")
            else:
                self.current_task = start_task - 1
            if termination_task == -1:
                self.termination_task = self.num_tasks - 1
            else:
                self.termination_task = termination_task
            self.last_retraining_start_time = 0
            self.retrain_fin_tracker = RetrainFinTracker.options(num_cpus=0).remote()
            self.num_resources = num_gpus
            
        
        
    def update_inference_jobs(self,
                              hyperparameters: dict,
                              gpu_inference_fractions: dict,
                              blocking: bool = False):
        # Updates inference weights and launches inference executor if not already running
        for camera_idx, camera in enumerate(self.cameras):
            this_gpu_fraction = gpu_inference_fractions.get(camera.id, 0)
            if this_gpu_fraction > 0:
                # Set the weights on the inference job and load the pretrained default model
                # pretrained_model_path = get_pretrained_model_format(camera.dataset_name, self.pretrained_model_dir).format(
                #                                                   hyperparameters[camera.id]["model_name"],
                #                                                   hyperparameters[camera.id]["num_hidden"])
                
                pretrained_model_path = None
                # Use initial model
                if self.current_task==0:
                    pretrained_model_path = self.model_load_path
                # Use updated model
                else:
                    # Look into the model train history, find the newest model and use that models path!
                    # Case 1: No model was pretrained before -> just use initial model!
                    model_train_history_df_path = os.path.join(self.log_dir, str(camera_idx), "model_train_history.csv")
                    if not os.path.exists(model_train_history_df_path):
                        pretrained_model_path = self.model_load_path
                    # Case 2: Load the most recent one's vals
                    else:
                        model_train_history_df = pd.read_csv(model_train_history_df_path)
                        weight_save_path = model_train_history_df['weight_save_path'].iloc[-1]
                        prev_task_num = model_train_history_df['task_num'].iloc[-1]
                        # chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                        # hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                        # epoch  = model_train_history_df['epoch'].iloc[-1]

                        # Check for conflict issues
                        print(prev_task_num, self.current_task)
                        if prev_task_num > self.current_task:
                            message = "There was an error in model_train_history_logging, fix this issue!"
                            stop_sys(message, raise_error=True)
                        else:
                            pretrained_model_path = weight_save_path
                            
                
                camera.update_inference_model(hyperparameters[camera.id],
                                              this_gpu_fraction,
                                              restore_path=pretrained_model_path,
                                              blocking=blocking)
                if camera.id not in self.inference_tasks:
                    # Run inference executor if not already running.
                    self.inference_tasks[camera.id] = inference_executor.remote(
                        camera,
                        num_chunks=self.num_inference_chunks,
                        retraining_period=self.retraining_period,
                        test_batch_size=hyperparameters[camera.id]["test_batch_size"],
                        window_start_time=self.last_retraining_start_time,
                        model_name=hyperparameters[camera.id]["model_name"],
                        last_layer_only=hyperparameters[camera.id]["last_layer_only"],
                        retrain_fin_tracker=self.retrain_fin_tracker,
                        logger=self.logger,
                        camera_idx=camera_idx,
                        task_id=self.current_task,
                    )
            elif this_gpu_fraction == 0:
                print("[WARN][Task {}] Camera {} was assigned 0 or no resources for inference. Not running inference".format(
                    camera.current_task, camera.id))
        
    def retraining_completion_callback(self, camera_id):
        pass

    def launch_training_jobs(self,
                             hyperparameters: dict,
                             gpu_training_fractions: dict):
        if self.current_task > 0:
            for camera_idx, camera in enumerate(self.cameras):
                this_gpu_fraction = gpu_training_fractions.get(camera.id, 0)
                # Run retraining only if some resource weight is allocated
                if this_gpu_fraction > 0:
                    # pretrained_model_path = get_pretrained_model_format(camera.dataset_name, self.pretrained_model_dir).format(
                    #                                               hyperparameters[camera.id]["model_name"],
                    #                                               hyperparameters[camera.id]["num_hidden"])
                    
                    pretrained_model_path = None
                    # Use initial model
                    if self.current_task==0:
                        pretrained_model_path = self.model_load_path
                    # Use updated model
                    else:
                        model_train_history_df_path = os.path.join(self.log_dir, str(camera_idx), "model_train_history.csv")
                        if not os.path.exists(model_train_history_df_path):
                            pretrained_model_path = self.model_load_path
                        else:
                            model_train_history_df = pd.read_csv(model_train_history_df_path)
                            weight_save_path = model_train_history_df['weight_save_path'].iloc[-1]
                            prev_task_num = model_train_history_df['task_num'].iloc[-1]
                            # chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                            # hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                            # epoch  = model_train_history_df['epoch'].iloc[-1]

                            # Check for conflict issues
                            if prev_task_num > self.current_task:
                                message = "There was an error in model_train_history_logging, fix this issue!"
                                stop_sys(message, raise_error=True)
                            else:
                                pretrained_model_path = weight_save_path
                        
                        
                    # window_end_time: absolute timestamp when this window closes
                    # self.last_retraining_start_time: wall-clock time at the start of this window (set in run())
                    # self.retraining_period: fixed total window budget in seconds
                    window_end_time = self.last_retraining_start_time + self.retraining_period
                    (self.retraining_tasks[camera.id],
                     self.retraining_metadata[camera.id]) = camera.run_retraining(hyperparameters[camera.id],
                                                                         this_gpu_fraction,
                                                                         dataloaders_dict={},
                                                                         restore_path=pretrained_model_path,
                                                                         profiling_mode=False,
                                                                         task_num=self.current_task,
                                                                         window_end_time=window_end_time,
                                                                         )
                    ray.get(self.retrain_fin_tracker.set.remote(camera.id, False))
                    print("[TRAINING START][Task {}] Camera {} training started. hp_id={}, epochs={}, restore_path={}".format(
                        self.current_task, camera.id, hyperparameters[camera.id]["id"], hyperparameters[camera.id]["epochs"], pretrained_model_path))
                else:
                    ray.get(self.retrain_fin_tracker.set.remote(camera.id, True))
                    print("[Task {}] Camera {} was assigned 0 or no resources for retraining. Not retraining.".format(camera.current_task, camera.id))

    def check_task_loop(self):
        print("Starting check task loop.")
        running_retraining_tasks = list(self.retraining_tasks.values())
        while True:
            remaining_time = self.retraining_period - (time.time() - self.last_retraining_start_time)
            if remaining_time > 0:
                done_tasks, running_retraining_tasks = ray.wait(running_retraining_tasks, timeout=0)
                retraining_results = ray.get(done_tasks)
                done_camera_ids = [next(camera_id for camera_id, task_id in self.retraining_tasks.items() if task_id == t) for t in done_tasks]
                done_cameras = [next(c for c in self.cameras if c.id == i) for i in done_camera_ids]

                for done_camera_idx, done_camera in enumerate(done_cameras):
                    done_best_val_acc = retraining_results[done_camera_idx][0]
                    ray.get(self.retrain_fin_tracker.set.remote(done_camera.id, True))
                    print("[TRAINING FIN][Task {}] Camera {} training finished. best_val_acc={:.4f}".format(
                        self.current_task, done_camera.id, done_best_val_acc))

                # Retrained accuracy logging
                if self.logger:
                    log_time = time.time()
                    for i, [best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_results] in enumerate(retraining_results):
                        retraining_time_taken = log_time - self.last_retraining_start_time
                        camera = done_cameras[i]
                        hp_id = self.current_hyperparameters[camera.id]["id"]
                        hp_epochs = self.current_hyperparameters[camera.id]["epochs"]
                        # data = [log_time, camera.current_task, retraining_time_taken, best_val_acc, profile_preretrain_test_acc,
                        #         profile_test_acc, hp_id, hp_epochs]
                        # self.logger.append.remote("retraining_{}".format(camera.id), data)
                        
                        camera_idx = int(camera.id)
                        time_left_after_retraining = self.retraining_period - retraining_time_taken
                        actual_training_time = misc_results["total_time"]
                        per_epoch_avg_time = misc_results["per_epoch_avg_time"]
                        log_items = {
                            "camera_idx":[camera_idx],
                            "task_id":[self.current_task],
                            "retraining_time_taken":[retraining_time_taken],
                            "time_left_after_retraining":[time_left_after_retraining],
                            "actual_training_time":[actual_training_time],
                            "per_epoch_avg_time":[per_epoch_avg_time],
                            "best_val_acc":[best_val_acc],
                            "profile_preretrain_test_acc":[profile_preretrain_test_acc],
                            "profile_test_acc":[profile_test_acc],
                            "hp_id":[hp_id],
                            "hp_epochs":[hp_epochs],
                            "log_time":[log_time],
                        }
                        
                        self.logger.log_retraining_results.remote(
                            camera_idx=camera_idx,
                            task_id=self.current_task,
                            log_items=log_items,
                        )

                
                for c in done_cameras:
                    if self.current_task > 0:
                        self.retraining_completion_callback(c.id)
                        self.inference_resource_weights, self.training_resource_weights = \
                            self.scheduler.reallocation_callback(c.id,
                                                                    self.inference_resource_weights,
                                                                    self.training_resource_weights)
                        gpu_inference_fractions = self.inference_resource_weights
                        # Update the inference model with new retrained weights if inference is running
                        if c.inference_gpu_fraction > 0:
                            # Previous
                            pretrained_model_path = None

                            # Use initial model
                            if self.current_task==0:
                                pretrained_model_path = self.model_load_path
                            # Use updated model
                            else:
                                # Look into the model train history, find the newest model and use that models path!
                                # Case 1: No model was pretrained before -> just use initial model!
                                model_train_history_df_path = os.path.join(self.log_dir, str(int(c.id)), "model_train_history.csv")
                                if not os.path.exists(model_train_history_df_path):
                                    pretrained_model_path = self.model_load_path
                                # Case 2: Load the most recent one's vals
                                else:
                                    model_train_history_df = pd.read_csv(model_train_history_df_path)
                                    weight_save_path = model_train_history_df['weight_save_path'].iloc[-1]
                                    prev_task_num = model_train_history_df['task_num'].iloc[-1]
                                    # chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                                    # hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                                    # epoch  = model_train_history_df['epoch'].iloc[-1]

                                    # Check for conflict issues
                                    if prev_task_num > self.current_task:
                                        message = "There was an error in model_train_history_logging, fix this issue!"
                                        stop_sys(message, raise_error=True)
                                    else:
                                        pretrained_model_path = weight_save_path
                            
                            
                            c.update_inference_from_retrained_model(path=pretrained_model_path)
                            print("[MODEL UPDATE][Task {}] Camera {} inference model updated with retrained weights. path={}".format(
                                self.current_task, c.id, pretrained_model_path))

                        # Update remaining cameras' inference weights
                        # NOTE(ROMILB): Updating each inference job's weights through restarts might be expensive
                        # NOTE(ROMILB): Consider parallelizing this.
                        for x in self.cameras:
                            # Update only if inference was running (wt > 0)
                            if x.id != c.id and gpu_inference_fractions[x.id] > 0:
                                x.update_inference_model(self.current_hyperparameters[x.id],
                                                            gpu_inference_fractions[x.id],
                                                            blocking=False)
                
                # Check if all inference tasks have completed
                done_inference, _ = ray.wait(list(self.inference_tasks.values()), num_returns=len(self.inference_tasks), timeout=0)
                if len(done_inference) == len(self.inference_tasks):
                    print("[CHECK TASK LOOP] All inference tasks completed. Breaking early.")
                    return

                print("Check task loop: remaining time: {}".format(remaining_time))
                time.sleep(1)   # Sleep before checking again. WARN: This might cause timing leaks.
            if remaining_time <= 0:
                return

    def _emergency_cleanup(self):
        """Kill all actors and shut down Ray. Safe to call multiple times."""
        print("[Ekya] Running emergency cleanup.")
        try:
            self.stop_current_jobs()
        except Exception as e:
            print(f"[Ekya] stop_current_jobs failed during cleanup: {e}")
        try:
            ray.kill(self.logger)
        except Exception:
            pass
        try:
            ray.shutdown()
            print("[Ekya] ray.shutdown() done.")
        except Exception as e:
            print(f"[Ekya] ray.shutdown failed: {e}")

    def run(self):
        '''
        Main loop for Ekya.
        :return:
        '''
        try:
            done = False # This is set to true to exit the loop. Happens when current_task > num_tasks
            while not done:
                print("Started loop")
                this_loop_start_time = time.time()
                self.last_retraining_start_time = this_loop_start_time
                # TODO: Make this async:
                self.new_task_callback()
                # Check running retraining tasks and load their models into inference when they complete.
                self.check_task_loop()  # This loop terminates when the time period for the task is done.

                # ray.get([self.inference_tasks[c.id] for c in self.cameras])
                # Stop all jobs. Required to stop any rogue jobs running
                self.stop_current_jobs()
                if self.current_task == self.termination_task:   # We have completed the termination task so end here.
                    done = True
        finally:
            self._emergency_cleanup()

    def new_task_callback(self):
        '''
        This method increments tasks and computes a new schedule and runs tasks for execution with the right resource allocations.
        :return:
        '''
        # Increment tasks
        self.current_task += 1
        print("New task callback called. Now on task {}/{}".format(self.current_task, self.num_tasks))
        for camera in self.cameras:
            camera.set_current_task(self.current_task)

        self.inference_tasks = {}
        self.retraining_tasks = {}
        self.retraining_metadata = {}
        self.inference_resource_weights, self.training_resource_weights = {}, {}
        ray.get(self.retrain_fin_tracker.init.remote([c.id for c in self.cameras]))

        # Launch inference jobs as the clock starts ticking even before the scheduler is done
        self.inference_resource_weights, self.current_hyperparameters = self.scheduler.get_inference_schedule(self.cameras, self.num_resources, task_id=self.current_task)

        # Use GPU resource weights directly (no memory-based ray demand conversion)
        gpu_inference_fractions = self.inference_resource_weights
        print("[Ekya] Inference Scheduler allocation: Inference: {}. GPU fractions: {}\n".format(self.inference_resource_weights, gpu_inference_fractions))
        self.update_inference_jobs(self.current_hyperparameters, gpu_inference_fractions)

        # Future work: These resource allocations should be over time. Consquently, resource allocations must be updated over time.
        custom_state = {"task_id": self.current_task,
                        "retraining_period": self.retraining_period,
                        "window_start_time": self.last_retraining_start_time,
                        "retrain_fin_tracker": self.retrain_fin_tracker}
        prev_hyperparams = self.current_hyperparameters
        self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters = self.scheduler.get_schedule(self.cameras, self.num_resources, custom_state)    # {camera.id: schedule/hyperparameters}

        # If resources were not allocated for training, preserve previous hyperparameters
        for camera in self.cameras:
            if camera.id not in self.current_hyperparameters:
                self.current_hyperparameters[camera.id] = prev_hyperparams[camera.id]

        # Use GPU resource weights directly
        gpu_inference_fractions = self.inference_resource_weights
        gpu_training_fractions = self.training_resource_weights
        print("[Ekya] Training+Inference Scheduler allocation: Training: {}\nInference: {}\n GPU Training: {}\n GPU Inference: {}".format(self.training_resource_weights, self.inference_resource_weights, gpu_training_fractions, gpu_inference_fractions))
        self.log_schedules(self.current_task, self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters)

        # Update inference jobs with new weights from the scheduler and launch retraining jobs
        self.update_inference_jobs(self.current_hyperparameters, gpu_inference_fractions)
        self.launch_training_jobs(self.current_hyperparameters, gpu_training_fractions)

    def log_schedules(self, task_id, inference_resource_weights, training_resource_weights, current_hyperparameters):
        self.logger.log_schedules.remote(
            task_id=task_id,
            inference_resource_weights=inference_resource_weights,
            training_resource_weights=training_resource_weights,
            current_hyperparameters=current_hyperparameters,
        )
        
    # Original Logic Kept
    def stop_current_jobs(self):
        '''
        Stops all current jobs in the system. Used at the start of a retraining.
        :return:
        '''
        # TODO: Log results if any are still running.
        print("Stopping all jobs.")
        for task in self.inference_tasks.values():
            ray.cancel(task, force=True)
        for camera in self.cameras:
            if hasattr(camera, "training_model"):
                ray.kill(camera.training_model)
                del camera.training_model
            if hasattr(camera, "inference_model"):
                ray.kill(camera.inference_model)
                del camera.inference_model
            # WARN: This will result in RayActorErrors for some of the get tasks - they must be handled in the calling functions.


    
    def shutdown(self, fin_file_path):
        # message = f"This is the shutdown pahse for Ekya, make any post processings here on the runtime results"
        # stop_sys(message)
        
        self.fin_file_path = fin_file_path
        with open(self.fin_file_path, "w") as f:
            f.write("done\n")
        
    # Not touched...
    
    # Deprecated... -> Need to replace logging logics
    

    def accumulate_profiles(self, camera_id, task_id):
        pass

    def flush_profiles(self):
        pass

    

    