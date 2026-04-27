import time, numpy as np, ray, pandas as pd, os
from typing import List
from ekya.simulation.camera import generate_training_job
from ekya.simulation.schedulers import thief_sco_scheduler
from ekya.simulation.jobs import InferenceJob as SimInferenceJob
from ekya.schedulers.scheduler import BaseScheduler, fair_reallocation
from ekya.microprofilers.modelling_funcs import get_scaled_optimus_fn, get_linear_fn
from ekya_update.camera_update import CameraSubstitution
from ekya_update.model_update import RayMLModel
from ekya_update.simple_microprofiler_update import SimpleMicroprofilerSubstitution, subsample_dataloader
from ekya_update.common import stop_sys, atomic_to_csv, logical_to_physical_gpu

# Labeling time simulation (linear scaling model, same as AdaInf):
# Reference profiling data (allocation required for 60 fps labeling throughput):
#   ViT_Large_timm teacher: 42% GPU allocation -> 60 frames/second
#   resnext101_64x4d teacher: 12% GPU allocation -> 60 frames/second
LABELING_REFERENCE_FPS = 60
LABELING_REFERENCE_ALLOCATION_BY_TEACHER = {
    "ViT_Large_timm": 0.42,
    "ViT_Large_Dino_v3": 0.42,
    "resnext101_64x4d": 0.12,
}

# Model load time (seconds) per teacher model — one-time cost before labeling inference begins
LABELING_MODEL_LOAD_TIME_BY_TEACHER = {
    "ViT_Large_timm": 0,
    "ViT_Large_Dino_v3": 0,
    "resnext101_64x4d": 0,
}


def get_simulated_labeling_time(teacher_model_name, gpu_allocation_percent, num_samples_to_label):
    """Simulate labeling time = model_load_time + inference_time.
    Inference time uses linear scaling from reference profiling data.
    Only selected samples need labeling (sample selection happens before labeling)."""
    gpu_allocation_fraction = gpu_allocation_percent / 100

    if teacher_model_name not in LABELING_REFERENCE_ALLOCATION_BY_TEACHER:
        message = f"No labeling reference data for teacher model: {teacher_model_name}. Known teachers: {list(LABELING_REFERENCE_ALLOCATION_BY_TEACHER.keys())}"
        stop_sys(message, raise_error=True)
    else:
        model_load_time = LABELING_MODEL_LOAD_TIME_BY_TEACHER[teacher_model_name]
        reference_allocation = LABELING_REFERENCE_ALLOCATION_BY_TEACHER[teacher_model_name]
        simulated_fps = (gpu_allocation_fraction / reference_allocation) * LABELING_REFERENCE_FPS
        inference_time = num_samples_to_label / simulated_fps
        labeling_time = model_load_time + inference_time

    return labeling_time

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
        self.steal_increment = self.scheduler_kwargs["steal_increment"]
        
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
                    # microprofile_accuracy_model = lambda x: default_acc * np.ones_like(x)
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
                for acc_prediction, runtime_prediction, epochs in zip(acc_predictions, runtime_predictions, self.profiling_epochs):
                    hp_temp = hyperparameters.copy()
                    hp_temp['epochs'] = int(epochs)
                    camera_profiles.append([hp_temp, acc_prediction, runtime_prediction, int(epochs), default_acc])
            profiles[camera.id] = camera_profiles
        if unsuccesful_models:
            print(
                "[THIEF SCHEDULER][WARN] Failed to generate models for {} cameras. Using default inference accuracy.".format(
                    unsuccesful_models))
        return profiles
    
    def run_teacher_labeling(self, cameras, num_frames_per_camera, gpu_resources):
        """
        Simulate teacher labeling cost without actually spawning a teacher model.
        Uses AdaInf's linear scaling model: labeling_time = num_samples / simulated_fps
        where simulated_fps = (gpu_allocation / reference_allocation) * 60fps.

        :param cameras: list of CameraSubstitution
        :param num_frames_per_camera: dict {camera.id: num_frames_to_label}.
                                      Cameras with 0 frames are skipped.
        :param gpu_resources: GPU fraction (0-1 scale) allocated per teacher
        :return: labeling_time (wall-clock seconds consumed)
        """
        start_time = time.time()
        simulated_budget = 0.0

        for camera in cameras:
            num_frames = num_frames_per_camera[camera.id]
            if num_frames <= 0:
                continue
            simulated_budget += get_simulated_labeling_time(
                teacher_model_name=camera.model_name_for_golden,
                gpu_allocation_percent=gpu_resources * 100,
                num_samples_to_label=num_frames,
            )

        if simulated_budget > 0:
            print(f"[TEACHER LABELING] Simulating {simulated_budget:.2f}s labeling time")
            time.sleep(simulated_budget)

        labeling_time = time.time() - start_time
        total_frames = sum(num_frames_per_camera.values())
        print(f"[TEACHER LABELING] Labeling took {labeling_time:.2f}s, "
              f"total_frames={total_frames}, per_camera={num_frames_per_camera}")
        return labeling_time

    def execute_microprofiling(self, cameras, task_id, version=2):
        if version == 1:
            return self._execute_microprofiling_v1(cameras, task_id)
        else:
            return self._execute_microprofiling_v2(cameras, task_id)

    def _execute_microprofiling_v1(self, cameras, task_id):
        """Ver 1: Original Ray-based microprofiling (ray actors + remote tasks)"""
        hyp_list = list(self.hyperparameters.values())
        ray_microprof = ray.remote(SimpleMicroprofilerSubstitution)
        microprofs = {}
        microprof_tasks = {}
        for camera_idx, camera in enumerate(cameras):
            this_microprof = ray_microprof.options(num_cpus=0).remote(self.microprofile_device)
            microprofs[camera.id] = this_microprof
            dataloaders = [camera._get_dataloader(task_id=task_id, train_batch_size=hp["train_batch_size"],
                                                  test_batch_size=hp["test_batch_size"], subsample_rate=hp["subsample"],
                                                  model_name=hp["model_name"], last_layer_only=hp["last_layer_only"],
                                                  save_csv_path=os.path.join(self.log_dir, "runtime_logs_micro_profile", f"hp_{hp['id']}"))
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
                    # chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                    # hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                    # epoch  = model_train_history_df['epoch'].iloc[-1]

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
                                                                             model_name=hyp_list[0]["model_name"],
                                                                             )
            microprof_tasks[camera.id] = microprof_task
        micrprofile_results = {}
        for camera in cameras:
            best_result, results = ray.get(microprof_tasks[camera.id])
            micrprofile_results[camera.id] = results
            ray.get(microprofs[camera.id].cleanup.remote())
        del microprofs
        return micrprofile_results

    def _execute_microprofiling_v2(self, cameras, task_id):
        """Ver 2: Direct microprofiling without Ray actors (sequential, no GPU leak)"""
        hyp_list = list(self.hyperparameters.values())
        microprofiler = SimpleMicroprofilerSubstitution(device=self.microprofile_device)
        micrprofile_results = {}
        for camera_idx, camera in enumerate(cameras):
            dataloaders = [camera._get_dataloader(task_id=task_id, train_batch_size=hp["train_batch_size"],
                                                  test_batch_size=hp["test_batch_size"], subsample_rate=hp["subsample"],
                                                  model_name=hp["model_name"], last_layer_only=hp["last_layer_only"],
                                                  save_csv_path=os.path.join(self.log_dir, "runtime_logs_micro_profile", f"hp_{hp['id']}"))
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
                    # chunk_num  = model_train_history_df['chunk_num'].iloc[-1]
                    # hyperparameter_id  = model_train_history_df['hyperparameter_id'].iloc[-1]
                    # epoch  = model_train_history_df['epoch'].iloc[-1]

                    # Check for conflict issues
                    if prev_task_num > task_id:
                        message = "There was an error in model_train_history_logging, fix this issue!"
                        stop_sys(message, raise_error=True)
                    else:
                        pretrained_model_path = weight_save_path

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
                model_name=hyp_list[0]["model_name"],
            )
            micrprofile_results[camera.id] = results
            microprofiler.cleanup()
        return micrprofile_results
    
    
    # Original Logic Kept
    def get_schedule(self,
                     cameras: CameraSubstitution,
                     resources: float,
                     state: dict):
        task_id = state["task_id"]
        retraining_period = state["retraining_period"]
        window_start_time = state["window_start_time"]
        retrain_fin_tracker = state.get("retrain_fin_tracker", None)

        # No training data available at task 0 - inference only
        if task_id == 0:
            inference_resource_weights, hyperparameters = self.get_inference_schedule(cameras, resources, task_id=task_id)
            training_resource_weights = {c.id: 0 for c in cameras}
            schedule_result = inference_resource_weights, training_resource_weights, hyperparameters
            # Mark all cameras as retrain-done so inference_executor reduces idle sleep
            if retrain_fin_tracker is not None:
                for c in cameras:
                    ray.get(retrain_fin_tracker.set.remote(c.id, True))
        else:
            pipeline_start_time = time.time()
            labeling_gpu_resources = self.microprofile_resources_per_trial

            # ---- Phase 1: Teacher labeling for microprofiling ----
            # Label enough samples for microprofiling train/val + test evaluation
            # Train/val: max(hp["subsample"]) * microprofile_subsample_rate * num_samples_per_task
            # Test eval: microprofile_subsample_rate * num_samples_per_task (same for all HPs)
            max_hp_subsample = max(float(hp["subsample"]) for hp in self.hyperparameters.values())
            phase1_train_val_rate = max_hp_subsample * self.microprofile_subsample_rate
            phase1_test_rate = self.microprofile_subsample_rate
            phase1_frame_counts = {c.id: int(c.num_samples_per_task*(phase1_train_val_rate+phase1_test_rate)) for c in cameras}
            print(f"[GET_SCHEDULE] Phase 1: Teacher labeling for microprofiling "
                  f"(train_val_rate={phase1_train_val_rate:.4f}, test_rate={phase1_test_rate:.4f}, frames={phase1_frame_counts})")
            phase1_labeling_time = self.run_teacher_labeling(
                cameras,
                num_frames_per_camera=phase1_frame_counts,
                gpu_resources=labeling_gpu_resources,
            )

            # ---- Phase 2: Microprofiling ----
            print(f"[GET_SCHEDULE] Phase 2: Microprofiling")
            microprofile_start_time = time.time()
            microprofile_results = self.execute_microprofiling(cameras, task_id)
            default_inference_accs = {i: [hp_result['preretrain_test_acc'] for hp_result in result] for i, result in
                                      microprofile_results.items()}
            profiles = self.generate_profiles(cameras, microprofile_results, default_inference_accs)
            microprofile_time_taken = time.time() - microprofile_start_time

            # ---- Estimate phase 3 labeling time for scheduler ----
            # Compute per-frame labeling rate from phase 1
            total_phase1_frames = sum(phase1_frame_counts.values())
            if total_phase1_frames > 0 and phase1_labeling_time > 0:
                time_per_frame = phase1_labeling_time / total_phase1_frames
            else:
                time_per_frame = 0

            # ---- Generate SimInferenceJobs ----
            SimInferenceJobs = {}
            for camera in cameras:
                SimInferenceJobs[camera.id] = SimInferenceJob(
                    f"{camera.id}_inference", default_inference_accs[camera.id][0], self.default_hyperparams['model_name'],
                    self.inference_profile['subsampling'], self.inference_profile['c1'],
                    resource_alloc=0)

            # ---- Generate SimTrainingJobs (with estimated labeling overhead in runtime) ----
            SimTrainingCfgs = {}
            for camera in cameras:
                SimTrainingCfgs[camera.id] = []
                total_train_frames = camera.num_samples_per_task
                phase1_frames_this_camera = phase1_frame_counts[camera.id]
                for [hp, acc_prediction, runtime_prediction, epochs, preretrain_acc] in profiles[camera.id]:
                    if acc_prediction > preretrain_acc:
                        # Estimate additional labeling frames beyond phase 1
                        hp_subsample = float(hp["subsample"])
                        needed_frames = int(hp_subsample * total_train_frames)
                        additional_frames = max(0, needed_frames - phase1_frames_this_camera)
                        estimated_phase3_time = additional_frames * time_per_frame
                        adjusted_runtime = runtime_prediction + estimated_phase3_time
                        job = generate_training_job(
                            f"{camera.id}_train_{hp['id']}_{epochs}",
                            acc_prediction, adjusted_runtime,
                            preretrain_acc, model_name=hp['model_name'],
                            inference_job=SimInferenceJobs[camera.id], oracle=False)
                        SimTrainingCfgs[camera.id].append(job)

            sched_job_pairs = [[SimInferenceJobs[camera.id], SimTrainingCfgs[camera.id]] for camera in cameras]

            # ---- Run thief scheduler ----
            elapsed_so_far = time.time() - pipeline_start_time
            remaining_time = int(retraining_period - elapsed_so_far)
            assert remaining_time > 0, (
                f"Labeling + microprofiling consumed the entire retraining window. "
                f"Phase1: {phase1_labeling_time:.1f}s, Microprofiling: {microprofile_time_taken:.1f}s, "
                f"Elapsed: {elapsed_so_far:.1f}s, Retraining period: {retraining_period}s"
            )

            schedule = thief_sco_scheduler(sched_job_pairs,
                                           resources,
                                           remaining_time,
                                           iterations=3,
                                           steal_increment=self.steal_increment)
            init_schedule = schedule[0]
            print("[THIEF SCHEDULER] Schedule from thief scheduler: {}".format(init_schedule))
            schedule_result = self.extract_ekya_schedule(init_schedule, self.hyperparameters)

            # ---- Phase 3: Teacher labeling for retraining (actual) ----
            # Only label for cameras that actually got training resources (weight > 0)
            inference_resource_weights, training_resource_weights, chosen_hyperparameters = schedule_result
            phase3_frame_counts = {}
            for camera in cameras:
                if camera.id in chosen_hyperparameters and training_resource_weights.get(camera.id, 0) > 0:
                    chosen_subsample = float(chosen_hyperparameters[camera.id]["subsample"])
                    needed_frames = int(chosen_subsample*camera.num_samples_per_task)
                    already_labeled = phase1_frame_counts[camera.id]
                    phase3_frame_counts[camera.id] = max(0, needed_frames-already_labeled)
                else:
                    phase3_frame_counts[camera.id] = 0

            has_phase3_work = any(count > 0 for count in phase3_frame_counts.values())
            if has_phase3_work:
                print(f"[GET_SCHEDULE] Phase 3: Teacher labeling for retraining (frames={phase3_frame_counts})")
                phase3_labeling_time = self.run_teacher_labeling(
                    cameras,
                    num_frames_per_camera=phase3_frame_counts,
                    gpu_resources=labeling_gpu_resources,
                )
            else:
                phase3_labeling_time = 0
                print(f"[GET_SCHEDULE] Phase 3: Skipped (phase 1 covers all retraining needs)")

            # ---- Log timing ----
            pipeline_finish_time = time.time()
            total_pipeline_time = pipeline_finish_time - pipeline_start_time
            log_path = os.path.join(self.log_dir, "micro_profiling", f"micro_profiling_fin_task_{task_id}.csv")
            log_row = {
                "task_id": task_id,
                "window_start_time": window_start_time,
                "labeling_gpu_resources": labeling_gpu_resources,
                "microprofile_gpu_resources": self.microprofile_resources_per_trial,
                "phase1_labeling_time": phase1_labeling_time,
                "microprofile_time": microprofile_time_taken,
                "phase3_labeling_time": phase3_labeling_time,
                "total_pipeline_time": total_pipeline_time,
                "remaining_for_retraining": retraining_period - total_pipeline_time,
            }
            for camera in cameras:
                log_row[f"phase1_frames_{camera.id}"] = phase1_frame_counts[camera.id]
                log_row[f"phase3_frames_{camera.id}"] = phase3_frame_counts.get(camera.id, 0)
            atomic_to_csv(pd.DataFrame([log_row]), path=log_path)

            print(f"[GET_SCHEDULE] Pipeline: Phase1={phase1_labeling_time:.1f}s, "
                  f"Microprofile={microprofile_time_taken:.1f}s, Phase3={phase3_labeling_time:.1f}s, "
                  f"Total={total_pipeline_time:.1f}s, Remaining={retraining_period - total_pipeline_time:.1f}s")

            # ---- Log scheduler decisions ----
            inference_resource_weights, training_resource_weights, chosen_hyperparameters_log = schedule_result

            # Per-camera microprofiling raw results
            microprofile_rows = []
            for camera in cameras:
                for hp_result in microprofile_results[camera.id]:
                    microprofile_rows.append({
                        "task_id": task_id,
                        "camera_id": camera.id,
                        "hp_id": hp_result["hyperparameters"]["id"],
                        "preretrain_test_acc": hp_result["preretrain_test_acc"],
                        "microprofile_test_acc": hp_result["test_acc"],
                        "time_per_epoch": hp_result["time_per_epoch"],
                        "init_time": hp_result["init_time"],
                    })
            microprofile_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"microprofiling_task_{task_id}.csv")
            atomic_to_csv(pd.DataFrame(microprofile_rows), path=microprofile_log_path)

            # Per-camera profile predictions (extrapolated accuracy/runtime at each profiling epoch)
            profile_rows = []
            for camera in cameras:
                for [hp, acc_pred, runtime_pred, epochs, preretrain_acc] in profiles[camera.id]:
                    profile_rows.append({
                        "task_id": task_id,
                        "camera_id": camera.id,
                        "hp_id": hp["id"],
                        "epochs": epochs,
                        "predicted_acc": acc_pred,
                        "predicted_runtime": runtime_pred,
                        "preretrain_acc": preretrain_acc,
                        "acc_gain": acc_pred - preretrain_acc,
                    })
            profile_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"profiles_task_{task_id}.csv")
            atomic_to_csv(pd.DataFrame(profile_rows), path=profile_log_path)

            # Final scheduler allocation
            decision_rows = []
            for camera in cameras:
                row = {
                    "task_id": task_id,
                    "camera_id": camera.id,
                    "inference_resource_pct": inference_resource_weights.get(camera.id, 0),
                    "training_resource_pct": training_resource_weights.get(camera.id, 0),
                }
                if camera.id in chosen_hyperparameters_log:
                    hp = chosen_hyperparameters_log[camera.id]
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
            decision_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"decisions_task_{task_id}.csv")
            atomic_to_csv(pd.DataFrame(decision_rows), path=decision_log_path)

            # Raw thief scheduler output
            raw_schedule_log_path = os.path.join(self.log_dir, "scheduler_decisions", f"raw_schedule_task_{task_id}.csv")
            raw_rows = [{"task_id": task_id, "job": k, "resource_weight": v} for k, v in init_schedule.items()]
            atomic_to_csv(pd.DataFrame(raw_rows), path=raw_schedule_log_path)

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
                               resources: float,
                               task_id: int = 0):
        '''
        Returns the schedule when inference only jobs must be run.
        For task_id > 0, ensures minimum inference resources per camera to meet SLO,
        reserves microprofiling resources, and distributes any leftover to inference.
        If minimum inference + microprofiling exceeds total resources, raises an error.
        :param cameras: list of cameras
        :param resources: total resources in the system to be split across tasks
        :param task_id: current task id (0 = no retraining needed)
        :return: inference resource weights, hyperparameters
        '''
        if task_id > 0:
            total_min_inference = sum(c.min_inference_resources_to_meet_slo for c in cameras)
            total_required = total_min_inference + self.microprofile_resources_per_trial
            if total_required > resources:
                message = (f"Cannot fit minimum inference ({total_min_inference}) + "
                           f"microprofiling ({self.microprofile_resources_per_trial}) "
                           f"within total resources ({resources}). "
                           f"Total required: {total_required}")
                stop_sys(message, raise_error=True)
            # Distribute leftover resources (beyond microprofiling reserve) to inference
            leftover = resources - self.microprofile_resources_per_trial - total_min_inference
            per_camera_bonus = leftover / len(cameras) if leftover > 0 else 0
            inference_resource_weights = {
                c.id: (c.min_inference_resources_to_meet_slo + per_camera_bonus) * 100
                for c in cameras
            }
        else:
            inference_resource_weights = {c.id: (resources / len(cameras)) * 100 for c in cameras}
        hyperparameters = {c.id: self.default_hyperparams for c in cameras}
        return inference_resource_weights, hyperparameters

    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict):
        return fair_reallocation(completed_camera_name,
                                 inference_resource_weights,
                                 training_resources_weights)
    
    # def get_default_inference_accs(self, cameras, task_id, subsample_rate):
    #     tasks = {}
    #     for camera in cameras:
    #         # Get dataloader and subsample it
    #         dataloaders = camera._get_dataloader(task_id=task_id,
    #                                              train_batch_size=self.default_hyperparams["train_batch_size"],
    #                                              test_batch_size=self.default_hyperparams["test_batch_size"],
    #                                              subsample_rate=subsample_rate)
    #         subsampled_test_dataloader = subsample_dataloader(dataloaders['test'], subsample_rate)
    #         inference_model_actor = ray.get_actor("{}_inference".format(camera.id))
    #         tasks[camera.id] = inference_model_actor.test_acc.remote(test_loader=subsampled_test_dataloader,
    #                                                                  resource_scaled=False)
    #     default_inference_accs = {}
    #     for cid, task in tasks.items():
    #         default_inference_accs[cid] = ray.get(task)
    #     return default_inference_accs
    
    # Not touched...
    
    
    
    # Deprecated... -> Need to replace logging logics