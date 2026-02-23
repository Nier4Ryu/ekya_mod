import time, numpy as np, ray, pandas as pd, os
from typing import List
from ekya.simulation.camera import generate_training_job
from ekya.simulation.schedulers import thief_sco_scheduler
from ekya.simulation.jobs import InferenceJob as SimInferenceJob
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.microprofilers.modelling_funcs import get_scaled_optimus_fn, get_linear_fn
from ekya_update.camera_update import CameraSubstitution
from ekya_update.simple_microprofiler_update import SimpleMicroprofilerSubstitution, subsample_dataloader
from ekya_update.common import stop_sys

class ThiefSchedulerSubstitution(BaseScheduler):
    # Updated some part at least
    def __init__(self,
                 scheduler_kwargs,
                 model_load_path,
                 log_dir,
                 ):
        self.scheduler_kwargs = scheduler_kwargs
        self.inference_profile = pd.read_csv(self.scheduler_kwargs["inference_profile_path"])
        self.microprofile_device = self.scheduler_kwargs["microprofile_device"]
        self.microprofile_resources_per_trial = self.scheduler_kwargs["microprofile_resources_per_trial"]
        self.microprofile_epochs = self.scheduler_kwargs["microprofile_epochs"]
        self.microprofile_subsample_rate = self.scheduler_kwargs["microprofile_subsample_rate"]
        self.profiling_epochs = np.array(self.scheduler_kwargs["profiling_epochs"])
        self.default_hyperparams = self.scheduler_kwargs["hyperparams"]["0"] # Use hyperparameters 0 as default
        self.hyperparameters = self.scheduler_kwargs["hyperparams"]
        self.predmodel_acc_args = self.scheduler_kwargs["predmodel_acc_args"]
        self.measured_time_per_epoch_for_hyperparams=self.scheduler_kwargs["measured_time_per_epoch_for_hyperparams"]
        self.measured_inittime_for_hyperparams=self.scheduler_kwargs["measured_inittime_for_hyperparams"]
        
        self.model_load_path = model_load_path
        self.log_dir = log_dir
        
    def generate_profiles(self, cameras, microprofile_results, default_inference_accs):
        profiles = {}
        unsuccesful_models = 0
        for camera in cameras:
            camera_profiles = []
            for hp_result, default_acc in zip(microprofile_results[camera.id], default_inference_accs[camera.id]):
                test_acc, hyperparameters, init_time, time_per_epoch = hp_result['test_acc'], hp_result[
                    'hyperparameters'], hp_result['init_time'], hp_result['time_per_epoch']
                try:
                    microprofile_accuracy_model = get_scaled_optimus_fn(
                        microprofile_x=np.array([self.microprofile_epochs]),
                        microprofile_y=np.array([test_acc]),
                        start_acc=default_acc,
                        **self.predmodel_acc_args)
                except RuntimeError:
                    unsuccesful_models += 1
                    # Simply return the start accuracy
                    microprofile_accuracy_model = lambda x: default_acc * np.ones_like(x)

                # Get runtime model
                print(f"For debug:{self.measured_time_per_epoch_for_hyperparams}")
                time_per_epoch = self.measured_time_per_epoch_for_hyperparams[hyperparameters['id']]
                init_time = self.measured_inittime_for_hyperparams[hyperparameters['id']]
                microprofile_runtime_model = get_linear_fn(a=time_per_epoch,
                                                           b=init_time)

                acc_predictions = microprofile_accuracy_model(self.profiling_epochs)
                runtime_predictions = microprofile_runtime_model(self.profiling_epochs)
                for acc_prediction, runtime_prediction, epochs in zip(acc_predictions, runtime_predictions,
                                                                      self.profiling_epochs):
                    hp_temp = hyperparameters.copy()
                    hp_temp['epochs'] = int(epochs)
                    camera_profiles.append([hp_temp, acc_prediction, runtime_prediction, int(epochs), default_acc])
            profiles[camera.id] = camera_profiles
        if unsuccesful_models:
            print(
                "[THIEF SCHEDULER][WARN] Failed to generate models for {} cameras. Using default inference accuracy.".format(
                    unsuccesful_models))
        return profiles
    
    def execute_microprofiling(self, cameras, task_id):
        hyp_list = list(self.hyperparameters.values())
        ray_microprof = ray.remote(SimpleMicroprofilerSubstitution)
        microprofs = {}
        microprof_tasks = {}
        for camera_idx, camera in enumerate(cameras):
            this_microprof = ray_microprof.options(num_cpus=0).remote(self.microprofile_device)
            microprofs[camera.id] = this_microprof
            dataloaders = [camera._get_dataloader(task_id=task_id, train_batch_size=hp["train_batch_size"],
                                                  test_batch_size=hp["test_batch_size"], subsample_rate=hp["subsample"])
                           for hp in hyp_list]
            
            pretrained_model_path = None
            # Use initial model
            if task_id==0:
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
                    if prev_task_num > task_id:
                        message = "There was an error in model_train_history_logging, fix this issue!"
                        stop_sys(message, raise_error=True)
                    else:
                        pretrained_model_path = weight_save_path
            
            
            microprof_task = microprofs[camera.id].run_microprofiling.remote(candidate_hyperparams=hyp_list,
                                                                             dataloaders=dataloaders,
                                                                             resources=self.microprofile_resources_per_trial,
                                                                             epochs=self.microprofile_epochs,
                                                                             pretrained_model_path=pretrained_model_path,
                                                                             subsample_rate=self.microprofile_subsample_rate,
                                                                             task_num=task_id,
                                                                             camera_idx=camera_idx, 
                                                                             log_dir=self.log_dir,
                                                                             )
            microprof_tasks[camera.id] = microprof_task
        micrprofile_results = {}
        for camera in cameras:
            best_result, results = ray.get(microprof_tasks[camera.id])
            micrprofile_results[
                camera.id] = results  # List of [{best_val_acc, hyperparameters, init_time, time_per_epoch, profile_preretrain_test_acc, profile_test_acc}]
            ray.kill(microprofs[camera.id])  # Kill to free up GPU explicitly
        del microprofs
        return micrprofile_results
    
    
    # Original Logic Kept
    def get_schedule(self,
                     cameras: CameraSubstitution,
                     resources: float,
                     state: dict):
        task_id = state["task_id"]
        retraining_period = state["retraining_period"]

        # No training data available at task 0 - inference only
        if task_id == 0:
            inference_resource_weights, hyperparameters = self.get_inference_schedule(cameras, resources)
            training_resource_weights = {c.id: 0 for c in cameras}
            schedule_result = inference_resource_weights, training_resource_weights, hyperparameters
        else:
            microprofile_start_time = time.time()
            # Run microprofiling for each camera - both training and inference
            microprofile_results = self.execute_microprofiling(cameras, task_id)
            # Get default inference accuracy from microprofile results
            default_inference_accs = {i: [hp_result['preretrain_test_acc'] for hp_result in result] for i, result in
                                      microprofile_results.items()}

            # Generate more profiles by interpolation from micro profiles
            profiles = self.generate_profiles(cameras, microprofile_results, default_inference_accs)

            # Generate SimInferenceJobs
            SimInferenceJobs = {}
            for camera in cameras:
                SimInferenceJobs[camera.id] = SimInferenceJob(
                    f"{camera.id}_inference", default_inference_accs[camera.id][0], self.default_hyperparams['model_name'],
                    # TODO(romilb): Get inference accuracy from current inference config rather than 0th hyperparam.
                    self.inference_profile['subsampling'], self.inference_profile['c1'],
                    resource_alloc=0)  # Start with 0 alloc because the scheduler will modify this

            # Generate SimTrainingJobs
            SimTrainingCfgs = {}
            for camera in cameras:
                SimTrainingCfgs[camera.id] = [
                    generate_training_job(f"{camera.id}_train_{hp['id']}_{epochs}", acc_prediction, runtime_prediction,
                                          preretrain_acc, model_name=hp['model_name'], oracle=False)
                    for [hp, acc_prediction, runtime_prediction, epochs, preretrain_acc] in profiles[camera.id] if
                    acc_prediction > preretrain_acc]

            sched_job_pairs = [[SimInferenceJobs[camera.id], SimTrainingCfgs[camera.id]] for camera in cameras]

            microprofile_time_taken = time.time() - microprofile_start_time
            remaining_time = int(retraining_period - microprofile_time_taken)
            assert remaining_time > 0, "Microprofiling took all the time in the retraining period and no retraining " \
                                       "happened. Microprofiling time = {}, Retraining period = {}".format(
                microprofile_time_taken, retraining_period)

            schedule = thief_sco_scheduler(sched_job_pairs,
                                           resources,
                                           remaining_time,
                                           iterations=3,
                                           steal_increment=0.1)
            init_schedule = schedule[0]
            print("[THIEF SCHEDULER] Schedule from thief scheduler: {}".format(init_schedule))
            schedule_result = self.extract_ekya_schedule(init_schedule, self.hyperparameters)

        return schedule_result
    
    @staticmethod
    def extract_ekya_schedule(schedule: dict,
                              hyperparameter_map: dict) -> list[dict, dict, dict]:
        '''
        Given a schedule from the thief scheduler, extracts inference_resource_weights, training_resource_weights, hyperparameters to be consumed by ekya.
        :param schedule: From thief scheduler
        :param hyperparameter_map: Map of hp_id to hps
        :return: inference_resource_weights, training_resource_weights, hyperparameters, each dict mapping camera.id to value
        '''
        inference_resource_weights = {}
        training_resource_weights = {}
        hyperparameters = {}
        for job_string, weight in schedule.items():
            components = job_string.split('_')
            if "inference" in job_string:
                camera_id = '_'.join(components[0:-1])
                inference_resource_weights[camera_id] = weight * 100
            elif "train" in job_string:
                epochs = components[-1]
                hp_id = components[-2]
                camera_id = '_'.join(components[0:-3])
                hps = hyperparameter_map[hp_id].copy()
                hps['epochs'] = int(epochs)
                hyperparameters[camera_id] = hps
                training_resource_weights[camera_id] = weight * 100
            else:
                raise Exception("Invalid job string {}. Does not contain inference or train".format(job_string))
        return inference_resource_weights, training_resource_weights, hyperparameters
    
    def get_inference_schedule(self,
                               cameras: List[CameraSubstitution],
                               resources: float):
        '''
        Returns the schedule when inference only jobs must be run. This must be super fast since this is the schedule
        used before the get_schedule actual schedule is obtained.
        :param cameras: list of cameras
        :param resources: total resources in the system to be split across tasks
        :return: inference resource weights, hyperparameters
        '''
        # Return fair schedule for inference
        inference_resource_weights = {c.id: (resources / (len(cameras))) * 100 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, hyperparameters

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict):
        return fair_reallocation(completed_camera_name,
                                 inference_resource_weights,
                                 training_resources_weights)
    
    def get_default_inference_accs(self, cameras, task_id, subsample_rate):
        tasks = {}
        for camera in cameras:
            # Get dataloader and subsample it
            dataloaders = camera._get_dataloader(task_id=task_id,
                                                 train_batch_size=self.default_hyperparams["train_batch_size"],
                                                 test_batch_size=self.default_hyperparams["test_batch_size"],
                                                 subsample_rate=subsample_rate)
            subsampled_test_dataloader = subsample_dataloader(dataloaders['test'], subsample_rate)
            inference_model_actor = ray.get_actor("{}_inference".format(camera.id))
            tasks[camera.id] = inference_model_actor.test_acc.remote(test_loader=subsampled_test_dataloader,
                                                                     resource_scaled=False)
        default_inference_accs = {}
        for cid, task in tasks.items():
            default_inference_accs[cid] = ray.get(task)
        return default_inference_accs
    
    # Not touched...
    
    
    
    # Deprecated... -> Need to replace logging logics