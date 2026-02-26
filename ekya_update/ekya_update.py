import ray, os, time, pandas as pd
from typing import List
from torch.utils.data import DataLoader
from ekya.schedulers.utils import convert_to_ray_demands, quantize_demands
from ekya_update.camera_update import CameraSubstitution
from ekya_update.logger_update import LoggerSubstitution
from ekya_update.thief_scheduler_update import ThiefSchedulerSubstitution
from ekya_update.common import stop_sys, check_mps_is_running


@ray.remote
def inference_executor(camera: CameraSubstitution,
                       num_chunks: int,
                       retraining_period: int,
                       test_batch_size: int,
                       logger: ray.actor.ActorHandle = None,
                       camera_idx=None,
                       task_id=None,
                       ) -> None:
    # WARNING: Camera object is frozen copy at method invocation. Updates to camera wont be reflected in the remote function
    # Parse dataloader and break it into num_chunks
    test_loader = camera._get_dataloader(camera.current_task, test_batch_size=test_batch_size)["test"]
    test_dataset = test_loader.dataset
    index = test_dataset.get_indexes()
    chunk_size = len(test_dataset)//num_chunks
    time_per_chunk = retraining_period//num_chunks
    print("[InferenceExecutor] Initializing inference executor for camera {}. Total samples in task: {}. Chunk size: "
          "{}. Time per chunk: {}".format(camera.id, len(test_dataset), chunk_size, time_per_chunk))

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
        start_time = time.time()
        chunk_idxs = index[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
        chunk_dataset = test_dataset.get_filtered_dataset(chunk_idxs)
        chunk_loader = DataLoader(chunk_dataset,
                                              batch_size=test_loader.batch_size,
                                              shuffle=False,
                                              num_workers=test_loader.num_workers)
        # Try to fetch the latest actor handle from ray until successful
        inference_model_actor = fetch_actor()
        retry_count = 0
        test_acc = None
        log_items = None
        while test_acc is None and retry_count < 5:
            try:
                test_acc, log_items = ray.get(inference_model_actor.test_acc.remote(test_loader=chunk_loader))
            except:
                retry_count+=1
                print("[InferenceExecutor][WARNING] Test_acc task failed. Retrying. Attempt: {}".format(retry_count))
                inference_model_actor = fetch_actor()
                test_acc = None
        print("[InferenceExecutor] {}: Got test acc: {}".format(camera.id, test_acc))
        test_accs.append(test_acc)

        # Result logging
        if logger and log_items:
            logger.log_inference_results.remote(
                camera_idx=camera_idx,
                task_id=task_id,
                chunk_id=chunk_id,
                log_items=log_items,
            )

        end_time = time.time()
        chunk_remaining_time = time_per_chunk - (end_time - start_time)
        if chunk_remaining_time > 0:
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
                num_tasks,
                num_chunks,

                # Model Related
                model_load_path,
                
                # Training Related
                
                # Scheduler Related
                scheduler_name,
                scheduler_kwargs,
                
                # Resource Related
                num_gpus,
                gpu_memory,
                window_size,
                log_dir,
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
            self.num_tasks = num_tasks
            self.num_inference_chunks = num_chunks
            self.current_task = start_task-1
            self.termination_task = termination_task
            self.last_retraining_start_time = 0
            self.num_resources = num_gpus
            self.gpu_memory = gpu_memory
            
        
        
    def update_inference_jobs(self,
                              inference_resource_weights: dict,
                              hyperparameters: dict,
                              ray_inference_resource_demands: dict,
                              blocking: bool = False):
        # Updates inference weights and launches inference exectuoir if not already running
        for camera_idx, camera in enumerate(self.cameras):
            this_inference_weight = inference_resource_weights.get(camera.id, 0)
            camera.inference_gpu_weight = this_inference_weight
            if this_inference_weight > 0:
                this_inference_ray_demand = ray_inference_resource_demands[camera.id]
                camera.inference_ray_demand = this_inference_ray_demand
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
                        chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                        hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                        epoch  = model_train_history_df['epoch'].iloc[-1]

                        # Check for conflict issues
                        print(prev_task_num, self.current_task)
                        if prev_task_num > self.current_task:
                            message = "There was an error in model_train_history_logging, fix this issue!"
                            stop_sys(message, raise_error=True)
                        else:
                            pretrained_model_path = weight_save_path
                            
                
                camera.update_inference_model(hyperparameters[camera.id],
                                              this_inference_weight,
                                              this_inference_ray_demand,
                                              restore_path=pretrained_model_path,
                                              blocking=blocking)
                if camera.id not in self.inference_tasks:
                    # Run inference executor if not already running.
                    self.inference_tasks[camera.id] = inference_executor.remote(camera,
                                                                                num_chunks=self.num_inference_chunks,
                                                                                retraining_period=self.retraining_period,
                                                                                test_batch_size=hyperparameters[camera.id]["test_batch_size"],
                                                                                logger=self.logger,
                                                                                camera_idx=camera_idx,
                                                                                task_id=self.current_task,
                                                                                )
            elif this_inference_weight == 0:
                print("[WARN][Task {}] Camera {} was assigned 0 or no resources for inference. Not running inference".format(
                    camera.current_task, camera.id))
        
    def retraining_completion_callback(self, camera_id):
        pass

    def launch_training_jobs(self,
                             training_resource_weights: dict,
                             hyperparameters: dict,
                             ray_training_resource_demands: dict):
        if self.current_task > 0:
            for camera_idx, camera in enumerate(self.cameras):
                this_train_weight = training_resource_weights.get(camera.id, 0)
                camera.training_gpu_weight = this_train_weight
                # Run retraining only if some resource weight is allocated
                if this_train_weight > 0:
                    this_training_ray_demand = ray_training_resource_demands[camera.id]
                    camera.training_ray_demand = this_training_ray_demand
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
                            chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                            hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                            epoch  = model_train_history_df['epoch'].iloc[-1]

                            # Check for conflict issues
                            if prev_task_num > self.current_task:
                                message = "There was an error in model_train_history_logging, fix this issue!"
                                stop_sys(message, raise_error=True)
                            else:
                                pretrained_model_path = weight_save_path
                        
                        
                    (self.retraining_tasks[camera.id],
                     self.retraining_metadata[camera.id]) = camera.run_retraining(hyperparameters[camera.id],
                                                                         training_resource_weights[camera.id],
                                                                         this_training_ray_demand,
                                                                         dataloaders_dict={},
                                                                         restore_path=pretrained_model_path,
                                                                         profiling_mode=False,
                                                                         task_num=self.current_task,
                                                                         )
                else:
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
                        log_items = {
                            "camera_idx":[camera_idx],
                            "task_id":[self.current_task],
                            "retraining_time_taken":[retraining_time_taken],
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
                        ray_inference_resource_demands, _ = convert_to_ray_demands(self.inference_memory_demand,
                                                                                    self.inference_resource_weights,
                                                                                    self.training_memory_demand,
                                                                                    self.training_resource_weights)
                        ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
                        # Update the inference model with new retrained weights if inference is running
                        if c.inference_gpu_weight > 0:
                            # Previous
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
                                    chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                                    hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                                    epoch  = model_train_history_df['epoch'].iloc[-1]

                                    # Check for conflict issues
                                    if prev_task_num > self.current_task:
                                        message = "There was an error in model_train_history_logging, fix this issue!"
                                        stop_sys(message, raise_error=True)
                                    else:
                                        pretrained_model_path = weight_save_path
                            
                            
                            c.update_inference_from_retrained_model(path=pretrained_model_path)
                            
                        # Update remaining cameras' inference weights
                        # NOTE(ROMILB): Updating each inference job's weights through restarts might be expensive
                        # NOTE(ROMILB): Consider parallelizing this.
                        for x in self.cameras:
                            # Update only if inference was running (wt > 0)
                            if x.id != c.id and self.inference_resource_weights[x.id] > 0:
                                x.update_inference_model(self.current_hyperparameters[x.id],
                                                            self.inference_resource_weights[x.id],
                                                            ray_inference_resource_demands[x.id],
                                                            blocking=False)
                
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
                    self.logger.flush.remote()
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

        # Launch inference jobs as the clock starts ticking even before the scheduler is done
        self.inference_resource_weights, self.current_hyperparameters = self.scheduler.get_inference_schedule(self.cameras, self.num_resources)
        self.inference_memory_demand, self.training_memory_demand = self.get_memory_demands(self.cameras)

        # Get ray resource demands to launch inference jobs and quantize for packing
        ray_inference_resource_demands, _ = convert_to_ray_demands(self.inference_memory_demand, self.inference_memory_demand,
                                                                                               self.training_memory_demand, self.training_memory_demand)    # HACK: Reduced demand to let micrprofiling run
        ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
        print("[Ekya] Inference Scheduler allocation: Inference: {}. Ray demands: {}\n".format(self.inference_resource_weights, ray_inference_resource_demands))
        self.update_inference_jobs(self.inference_resource_weights, self.current_hyperparameters, ray_inference_resource_demands)

        # Future work: These resource allocations should be over time. Consquently, resource allocations must be updated over time.
        custom_state = {"task_id": self.current_task,
                        "retraining_period": self.retraining_period}
        prev_hyperparams = self.current_hyperparameters
        self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters = self.scheduler.get_schedule(self.cameras, self.num_resources, custom_state)    # {camera.id: schedule/hyperparameters}

        # If resources were not allocated for training, preserve previous hyperparameters
        for camera in self.cameras:
            if camera.id not in self.current_hyperparameters:
                self.current_hyperparameters[camera.id] = prev_hyperparams[camera.id]

        # Get ray resource demands and quantize for packing
        ray_inference_resource_demands, ray_training_resource_demands = convert_to_ray_demands(self.inference_memory_demand, self.inference_resource_weights,
                                                                   self.training_memory_demand, self.training_resource_weights)
        ray_inference_resource_demands = quantize_demands(ray_inference_resource_demands)
        ray_training_resource_demands = quantize_demands(ray_training_resource_demands)
        print("[Ekya] Training+Inference Scheduler allocation: Training: {}\nInference: {}\n Ray Training: {}\n Ray Inference: {}".format(self.training_resource_weights, self.inference_resource_weights, ray_training_resource_demands, ray_inference_resource_demands))
        self.log_schedules(self.current_task, self.inference_resource_weights, self.training_resource_weights, self.current_hyperparameters)

        # Update inference jobs with new weights from the scheduler and launch retraining jobs
        self.update_inference_jobs(self.inference_resource_weights, self.current_hyperparameters, ray_inference_resource_demands)
        self.launch_training_jobs(self.training_resource_weights, self.current_hyperparameters, ray_training_resource_demands)

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

    def get_memory_demands(self, cameras=None):
        # Returns memory demands as a fraction of per GPU memory
        if not cameras:
            cameras = self.cameras
        inference_memory_demands = {}
        training_memory_demands = {}
        for camera in cameras:
            inference_memory_demands[camera.id] = camera.inference_memory_footprint()/self.gpu_memory
            training_memory_demands[camera.id] = camera.training_memory_footprint()/self.gpu_memory
        return inference_memory_demands, training_memory_demands

    
    
    # Not touched...
    
    # Deprecated... -> Need to replace logging logics
    

    def accumulate_profiles(self, camera_id, task_id):
        pass

    def flush_profiles(self):
        pass

    

    