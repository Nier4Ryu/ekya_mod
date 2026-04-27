"""
This is a substitution of 
'baselines/ekya/ekya/classes/camera.py' 
consider the previous version as deprecated

As trying to follow the previous version, some names may sound strange
(ex: test samples comes from train_sample_list / train is called pre-train)
"""
import numpy as np, pandas as pd, ray, os, torch, time, math
from lite.src.utils.common import warn_user
from ray.exceptions import RayActorError
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms_v2
from torchvision.transforms.functional import InterpolationMode
from ekya.CONFIG import RANDOM_SEED
from ekya_update.model_update import RayMLModel
from ekya_update.common import atomic_to_csv, stop_sys, apply_ast_on_col, check_mps_is_running, logical_to_physical_gpu
from ekya.utils.dataset_utils import get_normalize_params_from_model_name

IMAGENET_NORMALIZE_PARAMS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
GOOGLE_VIT_NORMALIZE_PARAMS = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

def get_input_size_from_model_name(model_name):
    if model_name=="ViT_Tiny_EVA02_336":
        input_size=(336, 336)
    elif model_name=="ViT_Small_EVA02_336":
        input_size=(336, 336)
    elif model_name=="ViT_Base_EVA02_448":
        input_size=(448, 448)
    elif model_name=="ViT_Large_EVA02_448":
        input_size=(448, 448)
    else:
        input_size=(224, 224)
    return input_size

class CameraSubstitution(object):
    # Updated some part at least
    def __init__(self, 
                # Identification
                id, 

                # Dataset Related
                dataset_name,
                train_sample_list_path,
                train_golden_sample_list_path,
                test_sample_list_path,
                test_golden_sample_list_path,

                # Dataset Split related
                fps,
                slo,
                num_chunks,
                start_task,
                termination_task,
                
                # Golden model Related
                label_type,
                model_name_for_golden,
                num_hidden_for_golden,
                last_layer_only_for_golden,
                model_load_path_for_golden,
                
                # Ekya System Related
                train_split,
                inference_profile_path,
                min_inference_resources_to_meet_slo, # Originally max_inference_resources in Ekya's original code (ekya/simulation/jobs.py)

                training_memory_footprint_value,
                inference_memory_footprint_value,

                # Logging related
                log_dir,

                # Multi-stream alignment: when num_streams > 1, align df length to target_num_samples
                target_num_samples=None,
                 ):
        if not check_mps_is_running():
            message = "Ekya requires MPS server to be running! stopping sys!"
            stop_sys(message)
        else:
            # Setup camera ID
            self.id = id
            self.camera_idx = int(self.id) # I prefer int idx as id, just here for safety
            
            # Load dfs
            start_t = time.time()
            self.train_df = pd.read_csv(train_sample_list_path)
            self.train_df["is_dummy"] = False
            self.train_golden_df = pd.read_csv(train_golden_sample_list_path)
            self.test_df = pd.read_csv(test_sample_list_path)
            self.test_golden_df = pd.read_csv(test_golden_sample_list_path)
            fin_t = time.time()
            print(f"loading dfs took {fin_t-start_t}s for camera:{self.id}")


            # Save additional infos
            self.dataset_name = dataset_name
            self.fps = fps
            self.slo = slo
            self.num_chunks = num_chunks
            self.start_task = start_task
            self.termination_task = termination_task

            self.train_split = train_split
            self.inference_profile_path = inference_profile_path
            self.min_inference_resources_to_meet_slo = min_inference_resources_to_meet_slo

            self.training_memory_footprint_value = training_memory_footprint_value
            self.inference_memory_footprint_value = inference_memory_footprint_value

            # Golden model infos
            self.label_type = label_type
            self.model_name_for_golden = model_name_for_golden
            self.num_hidden_for_golden = num_hidden_for_golden
            self.last_layer_only_for_golden = last_layer_only_for_golden
            self.model_load_path_for_golden = model_load_path_for_golden

            # Log dir infos
            self.log_dir = os.path.join(log_dir, "logger_generated")

            # Derive task parameters from fps and slo
            self.chunk_size = self.fps * self.slo
            self.num_samples_per_task = self.chunk_size * self.num_chunks

            # Multi-stream alignment: truncate or elongate df to match target stream length
            # Only used when num_streams > 1
            if target_num_samples is not None:
                original_len = len(self.test_df)
                if original_len > target_num_samples:
                    self.test_df = self.test_df.iloc[:target_num_samples].reset_index(drop=True)
                    self.test_golden_df = self.test_golden_df.iloc[:target_num_samples].reset_index(drop=True)
                    warn_user(f"Camera {self.id}: Truncated from {original_len} to {target_num_samples} samples", warning_time=0)
                elif original_len < target_num_samples:
                    num_extra = target_num_samples - original_len
                    num_full_repeats = num_extra // original_len
                    num_remainder = num_extra % original_len

                    extra_test_dfs = [self.test_df] * num_full_repeats
                    if num_remainder > 0:
                        extra_test_dfs.append(self.test_df.iloc[:num_remainder])
                    extra_golden_dfs = [self.test_golden_df] * num_full_repeats
                    if num_remainder > 0:
                        extra_golden_dfs.append(self.test_golden_df.iloc[:num_remainder])

                    self.test_df = pd.concat([self.test_df] + extra_test_dfs, ignore_index=True)
                    self.test_golden_df = pd.concat([self.test_golden_df] + extra_golden_dfs, ignore_index=True)
                    warn_user(f"Camera {self.id}: Elongated from {original_len} to {target_num_samples} samples", warning_time=0)
                else:
                    print(f"Camera {self.id}: Already at target {target_num_samples} samples")

            self.num_real_test_samples = len(self.test_df)
            self.num_tasks = math.ceil(self.num_real_test_samples / self.num_samples_per_task)

            # Mark existing samples as real, then pad for clean divisibility
            self.test_df["is_dummy"] = False
            self.test_golden_df["is_dummy"] = False
            total_samples_needed = self.num_tasks * self.num_samples_per_task
            num_padding_samples = total_samples_needed - self.num_real_test_samples
            if num_padding_samples > 0:
                last_row_test = self.test_df.iloc[[-1]].copy()
                last_row_test["is_dummy"] = True
                padding_test_df = pd.concat([last_row_test] * num_padding_samples, ignore_index=True)
                self.test_df = pd.concat([self.test_df, padding_test_df], ignore_index=True)

                last_row_golden = self.test_golden_df.iloc[[-1]].copy()
                last_row_golden["is_dummy"] = True
                padding_golden_df = pd.concat([last_row_golden] * num_padding_samples, ignore_index=True)
                self.test_golden_df = pd.concat([self.test_golden_df, padding_golden_df], ignore_index=True)

                print(f"Camera {self.id}: Padded {num_padding_samples} dummy samples "
                      f"(real={self.num_real_test_samples}, total={total_samples_needed})")
            else:
                print(f"Camera {self.id}: No padding needed "
                      f"(real={self.num_real_test_samples}, total={total_samples_needed})")

            # Add golden_label column from golden model predictions
            self.test_df["golden_label"] = self.test_golden_df["pred"].values
            self.train_df["golden_label"] = self.train_golden_df["pred"].values

            # Internal params to keep track of where to load...
            self.current_task = -1

            # Balanced DB: replay buffer that accumulates exemplars across windows (AdaInf-style)
            self.db_max_per_class = 500
            self.db = pd.DataFrame()
            self.stream_follow_factor = 0.95  # 0.0 = uniform (iCaRL), 1.0 = fully stream-proportional

            print("Warning: As realtime inference scaling function is not aquireable without runtime profiling, we use the default ver - labmda x:1")
            self.inference_scaling_function = lambda x: 1

    def update_training_model(self,
                              hyperparameters: dict,
                              gpu_fraction: float,
                              restore_path: str = "",
                              blocking: bool = False):
        self.hyperparameters = hyperparameters
        self.training_gpu_fraction = gpu_fraction

        # Kill existing training actor immediately (force-kill, no graceful shutdown)
        # Previous approach: graceful terminate then force-kill (could delay resource release)
        # if hasattr(self, "training_model"):
        #     try:
        #         ray.get(self.training_model.__ray_terminate__.remote())
        #     except RayActorError:
        #         pass
        #     finally:
        #         try:
        #             ray.kill(self.training_model, no_restart=True)
        #         except Exception:
        #             pass
        if hasattr(self, "training_model"):
            try:
                ray.kill(self.training_model, no_restart=True)
            except Exception:
                pass
            del self.training_model
        model_updated = False
        while not model_updated:
            try:
                gpu_idx=0
                physical_gpu=logical_to_physical_gpu(gpu_idx)
                gpu_percent = int(self.training_gpu_fraction)
                print(f"[Training] for {self.id} GPU:{gpu_percent}%  Allocated")
                self.training_model = RayMLModel.options(
                    name="{}_training".format(self.id),
                    num_gpus=0.01,
                    resources={f"GPU{gpu_idx}": self.training_gpu_fraction / 100},
                    runtime_env={
                        "env_vars":{
                            "CUDA_VISIBLE_DEVICES": physical_gpu,
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent)
                        }
                    }
                ).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.training_gpu_fraction,
                    restore_path=restore_path,
                    name="{}_training".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
                    label_type=self.label_type,
                )
                model_updated = True
            except ValueError as e:
                # print("Got value error {}. Retrying..".format(e))
                pass  # Retrying because of actor name failures
        if blocking:
            print("WARNING: Training model init is blocking=True. This may cause training jobs to start before the clock timer starts.")
            ray.get(self.training_model.ready.remote())

    def run_retraining(self,
                       hyperparameters: dict,
                       gpu_fraction: float,
                       dataloaders_dict: dict = {},
                       validation_freq: int = -1,
                       restore_path: str = "",
                       profiling_mode: bool = False,
                       task_num=None,
                       window_end_time=None,
                       ):
        self.update_training_model(hyperparameters, gpu_fraction, restore_path=restore_path, blocking=False)
        if not dataloaders_dict:
            dataloaders_dict = self._get_dataloader(self.current_task,
                                                    train_batch_size=hyperparameters["train_batch_size"],
                                                    subsample_rate=hyperparameters["subsample"],
                                                    model_name=hyperparameters["model_name"],
                                                    last_layer_only=hyperparameters["last_layer_only"],
                                                    save_csv_path=os.path.join(self.log_dir, "runtime_logs"))
        
        if profiling_mode:
            message = f"Currently profiling mode is de-activated for camera_update, revert back to ekya or please implement it"
            stop_sys(message)
        
        metadata = {}
        for k, dataloader in dataloaders_dict.items():
            metadata[k] = {}
        
        task = self.training_model.retrain_model.remote(
            dataloaders_dict['train'], dataloaders_dict['val'],
            dataloaders_dict['test'], hyperparameters,
            validation_freq, profiling_mode,
            task_num=task_num,
            window_end_time=window_end_time,
            )
        
        return task, metadata

    
    # Update some parts, but keep calling conventions
    
    # -> Updated such that golden labels are retrieved here right away
    def _get_dataloader(self,
                        task_id: int,
                        train_batch_size: int = 1,
                        test_batch_size: int = 1,
                        num_workers: int = 4,
                        subsample_rate: float = 1,
                        shuffle: bool = False,
                        save_csv_path: str = None,
                        model_name: str = None,
                        last_layer_only = None):

        # Generate Test dataset for all cases
        dataloaders_dict = {
            "train":None,
            "val":None,
            "test":get_inference_dataloader_from_df(df=self.test_df.iloc[self.num_samples_per_task * task_id:self.num_samples_per_task * (task_id + 1)], batch_size=test_batch_size, model_name=model_name),
        }

        # Generate Train / Val datasets for task_id>0
        if task_id>0:
            temp_test_df = self.test_df.copy()
            label_col = "golden_label" if self.label_type == "golden_label" else "label"

            # Stream: previous window data
            stream_df = temp_test_df.iloc[self.num_samples_per_task*(task_id-1):self.num_samples_per_task*task_id]

            if self.db_max_per_class is not None:
                # --- AdaInf-style: per-label budgeted selection from stream + DB ---
                # Seed DB with initial training data on first use
                if len(self.db) == 0:
                    self._update_db(self.train_df)

                db_df = self.db.copy()
                num_total_to_pick = max(int(len(stream_df)*subsample_rate), 2)

                # Gather unique labels from stream + DB
                stream_labels = stream_df[label_col].values
                unique_labels = np.unique(stream_labels)
                if len(db_df) > 0:
                    unique_labels = np.unique(np.concatenate([unique_labels, db_df[label_col].values]))

                # Per-label budgets: blend of uniform + stream-proportional
                num_labels = len(unique_labels)
                uniform_budget = max(1, num_total_to_pick // num_labels)
                total_budget = uniform_budget * num_labels
                stream_counts = {label: np.sum(stream_labels == label) for label in unique_labels}
                total_stream = max(1, sum(stream_counts.values()))
                label_budgets = {}
                for label in unique_labels:
                    proportional_share = total_budget * stream_counts.get(label, 0) / total_stream
                    blended = (1.0-self.stream_follow_factor)*uniform_budget + self.stream_follow_factor*proportional_share
                    label_budgets[label] = max(1, int(round(blended)))

                # Per-label selection: half stream (random) + half DB (random)
                selected_stream_dfs = []
                selected_db_dfs = []
                for label in unique_labels:
                    budget = label_budgets[label]
                    stream_target = budget // 2
                    db_target = budget - stream_target

                    # Stream candidates for this label
                    stream_candidates = stream_df[stream_df[label_col] == label]
                    # DB candidates for this label
                    db_candidates = db_df[db_df[label_col] == label] if len(db_df) > 0 else pd.DataFrame()

                    # First pass
                    actual_stream = min(len(stream_candidates), stream_target)
                    actual_db = min(len(db_candidates), db_target)

                    # Second pass: fill shortfall from the other source
                    stream_shortfall = stream_target - actual_stream
                    db_shortfall = db_target - actual_db
                    actual_db = min(len(db_candidates), db_target+stream_shortfall)
                    actual_stream = min(len(stream_candidates), stream_target+db_shortfall)

                    # Cap total to budget
                    if actual_stream+actual_db > budget:
                        excess = (actual_stream+actual_db) - budget
                        if actual_stream > stream_target:
                            trim = min(excess, actual_stream-stream_target)
                            actual_stream -= trim
                            excess -= trim
                        if excess > 0:
                            actual_db -= excess

                    if actual_stream > 0:
                        selected_stream_dfs.append(stream_candidates.sample(n=actual_stream, random_state=RANDOM_SEED))
                    if actual_db > 0:
                        selected_db_dfs.append(db_candidates.sample(n=actual_db, random_state=RANDOM_SEED))

                parts = selected_stream_dfs + selected_db_dfs
                if parts:
                    combined_df = pd.concat(parts, ignore_index=True)
                else:
                    combined_df = pd.DataFrame()

                # Update DB with stream data (after selection)
                self._update_db(stream_df)

            else:
                # --- Original Ekya behavior: random sample from previous 1-2 windows ---
                num_samples_to_pick = max(int(len(stream_df)*subsample_rate), 2)
                stream_sampled = stream_df.sample(n=num_samples_to_pick, random_state=RANDOM_SEED)

                if task_id == 1:
                    old_candidates = self.train_df
                else:
                    old_candidates = temp_test_df.iloc[self.num_samples_per_task*(task_id-2):self.num_samples_per_task*(task_id-1)]
                num_old = min(max(int(len(old_candidates)*subsample_rate), 2), num_samples_to_pick)
                old_sampled = old_candidates.sample(n=num_old, random_state=RANDOM_SEED)

                combined_df = pd.concat([stream_sampled, old_sampled], ignore_index=True)

            # Shuffle then split into train / val
            combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            num_train = int(len(combined_df)*self.train_split)
            train_df = combined_df.iloc[:num_train]
            val_df = combined_df.iloc[num_train:]

            # Save train / val dfs
            if save_csv_path is not None:
                atomic_to_csv(train_df, path=os.path.join(save_csv_path, "train", f"camera_{self.camera_idx}_task_{task_id}.csv"))
                atomic_to_csv(val_df, path=os.path.join(save_csv_path, "val", f"camera_{self.camera_idx}_task_{task_id}.csv"))

            dataloaders_dict["train"] = get_train_dataloader_from_df(df=train_df, batch_size=train_batch_size, model_name=model_name, num_workers=num_workers, last_layer_only=last_layer_only)
            dataloaders_dict["val"] = get_inference_dataloader_from_df(df=val_df, batch_size=test_batch_size, model_name=model_name, num_workers=num_workers)

        return dataloaders_dict

    def _update_db(self, new_data_df):
        """Ingest new window data into the balanced DB, keeping at most db_max_per_class per class."""
        label_col = "golden_label" if "golden_label" in new_data_df.columns else "label"
        non_dummy = new_data_df[new_data_df["is_dummy"] == False] if "is_dummy" in new_data_df.columns else new_data_df
        self.db = pd.concat([self.db, non_dummy], ignore_index=True)
        # Trim to db_max_per_class per class (keep most recent samples)
        self.db = (
            self.db
            .groupby(label_col, group_keys=False)
            .apply(lambda g: g.tail(self.db_max_per_class))
            .reset_index(drop=True)
        )

    def _sample_balanced_from_db(self, n_total):
        """Sample n_total from DB, balanced across classes.
        Target: N/m per class. If a class has fewer, take all and fill shortfall
        by random-sampling from non-selected rows across other classes.
        """
        if len(self.db) == 0:
            return pd.DataFrame()
        label_col = "golden_label" if "golden_label" in self.db.columns else "label"
        classes = self.db[label_col].unique()
        m = len(classes)
        n_per_class = max(n_total // m, 1)

        selected_parts = []
        shortfall = 0
        non_selected_parts = []
        for cls in classes:
            cls_df = self.db[self.db[label_col] == cls]
            n_sample = min(n_per_class, len(cls_df))
            sampled = cls_df.sample(n=n_sample, random_state=RANDOM_SEED)
            selected_parts.append(sampled)
            shortfall += (n_per_class - n_sample)
            # Track non-selected rows from this class for shortfall filling
            non_selected = cls_df.drop(sampled.index)
            if len(non_selected) > 0:
                non_selected_parts.append(non_selected)

        result = pd.concat(selected_parts, ignore_index=True)

        # Fill shortfall by random-sampling from non-selected rows
        if shortfall > 0 and len(non_selected_parts) > 0:
            non_selected_pool = pd.concat(non_selected_parts, ignore_index=True)
            extra = min(shortfall, len(non_selected_pool))
            if extra > 0:
                result = pd.concat([result, non_selected_pool.sample(n=extra, random_state=RANDOM_SEED)], ignore_index=True)

        return result

    def set_current_task(self, new_current_task: int):
        self.current_task = new_current_task

    
    def update_inference_model(self,
                               hyperparameters: dict,
                               gpu_fraction: float,
                               restore_path: str = "",
                               blocking: bool = False):
        self.hyperparameters = hyperparameters
        self.inference_gpu_fraction = gpu_fraction

        self.current_inference_path = restore_path or self.current_inference_path  # Use restore path if specified, else use current_inference_path

        # Kill existing inference actor immediately (force-kill, no graceful shutdown)
        # Previous approach: graceful terminate then force-kill (could delay resource release)
        # if hasattr(self, "inference_model"):
        #     try:
        #         ray.get(self.inference_model.__ray_terminate__.remote())
        #     except RayActorError as e:
        #         print("Got exception while killing model. Continuing. {}".format(str(e)))
        #     finally:
        #         try:
        #             ray.kill(self.inference_model, no_restart=True)
        #         except Exception:
        #             pass
        if hasattr(self, "inference_model"):
            try:
                ray.kill(self.inference_model, no_restart=True)
            except Exception:
                pass
            del self.inference_model
        model_updated = False
        while not model_updated:
            try:
                gpu_idx=0
                physical_gpu=logical_to_physical_gpu(gpu_idx)
                gpu_percent = int(self.inference_gpu_fraction)
                print(f"[Inference] for {self.id} GPU:{gpu_percent}% Allocated")
                self.inference_model = RayMLModel.options(
                    name="{}_inference".format(self.id),
                    num_gpus=0.01,
                    resources={f"GPU{gpu_idx}": self.inference_gpu_fraction / 100},
                    runtime_env={
                        "env_vars":{
                            "CUDA_VISIBLE_DEVICES": physical_gpu,
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent)
                        }
                    }
                ).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.inference_gpu_fraction,
                    inference_scaling_function=self.inference_scaling_function,
                    restore_path=self.current_inference_path,
                    name="{}_inference".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
                    label_type=self.label_type,
                )
                model_updated = True
            except ValueError as e:
                pass  # Retrying
        if blocking:
            print("WARNING: Model init is blocking=True. This may cause training jobs to start before the clock timer starts.")
            ray.get(self.inference_model.ready.remote())

    def update_inference_from_retrained_model(self,
                                              path: str = None):
        if not path:
            message = f"Original code did not specify the exact save path, I would not allow such behavior"
            stop_sys(message)
        
        ray.get(self.training_model.save_model.remote(path))
        # Kill training actor immediately after save (force-kill, no graceful shutdown)
        # Previous approach: graceful terminate (could delay resource release)
        # try:
        #     ray.get(self.training_model.__ray_terminate__.remote())
        # except RayActorError as e:
        #     print("Got exception while killing model. Continuing. {}".format(str(e)))
        #     pass
        try:
            ray.kill(self.training_model, no_restart=True)
        except Exception:
            pass
        del self.training_model
        self.current_inference_path = path
        print("{}, request update {}".format(self.id, self))
        self.update_inference_model(self.hyperparameters,
                                    self.inference_gpu_fraction,
                                    restore_path=self.current_inference_path)
    
    def training_memory_footprint(self):
        return self.training_memory_footprint_value
        
    def inference_memory_footprint(self):
        return self.inference_memory_footprint_value
        
    # Original Logic Kept

    # Not touched

    # Deprecated...
    @staticmethod
    def get_infer_profile(min_inference_resources_to_meet_slo=1,
                          profile_path='real_inference_profiles.csv',
                          camera='c1'):
        message = f"This function is deprecated, please check if this is really needed"
        stop_sys(message, raise_error=True)

    def setup_dataset(self):
        message = f"This function is deprecated, please check if this is really needed"
        stop_sys(message, raise_error=True)
    
# Implementations from Lite
def get_train_dataloader_from_df(df, batch_size, model_name=None, return_vals="img_tensor_and_label_and_golden_label", num_workers=4, last_layer_only=None):
    mixed_dataset = MixedDatasetPILV2(df, inference_train_related_dict={"train": True, "last_layer_only":last_layer_only}, model_name=model_name, return_vals=return_vals)
    dataloader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=1, pin_memory=True, persistent_workers=True, multiprocessing_context="spawn") # persistent_workers=True for train function as dataset is static
    return dataloader

def get_inference_dataloader_from_df(df, batch_size, model_name=None, return_vals="img_tensor_and_label_and_golden_label", task_num=None, total_task_num=None, num_workers=4, prefetch_factor=1):
    df = get_df_partition_from_params(df, task_num=task_num, total_task_num=total_task_num)
    mixed_dataset = MixedDatasetPILV2(df, inference_train_related_dict={"train": False}, model_name=model_name, return_vals=return_vals)
    dataloader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True, multiprocessing_context="spawn")
    return dataloader

def get_df_partition_from_params(df, task_num=None, total_task_num=None):
    if isinstance(task_num, int) and isinstance(total_task_num, int):
        total = len(df)
        per_partition = total // total_task_num
        remainder = total % total_task_num

        start_idx = task_num * per_partition + min(task_num, remainder)
        end_idx = start_idx + per_partition + (1 if task_num < remainder else 0)

        df = df.iloc[start_idx:end_idx]
    
    return df

# Dataset 
class MixedDataset(Dataset):
    def __init__(self, df, return_vals="img_tensor_and_label_and_golden_label"):
        # Decide cols to keep (Reduce memory), return_func (What values are required)
        self.df = df.copy()
        cols_to_keep = []
        if return_vals=="img_tensor_and_label":
            cols_to_keep.extend(["label", "img_save_path", "is_dummy"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_label
        elif return_vals=="img_tensor_and_label_and_golden_label":
            cols_to_keep.extend(["label", "golden_label", "img_save_path", "is_dummy"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_label_golden_label
        elif return_vals=="img_tensor_and_label_and_feature_save_path_template":
            cols_to_keep.extend(["label", "img_save_path", "feature_save_path_template", "is_dummy"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_label_and_feature_save_path_template
        elif return_vals=="img_tensor_and_soft_label":
            cols_to_keep.extend(["img_save_path", "soft_label", "is_dummy"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_soft_label
            
            # if this df was read in from a file "soft_label" would be string, requiring ast.literal_eval
            if isinstance(self.df["soft_label"].iloc[0], str):
                self.df = apply_ast_on_col(df=self.df, col="soft_label")
            
            # Convert the list to torch tensor
            self.df["soft_label"] = self.df['soft_label'].apply(torch.tensor)
        else:
            message=f"return function for - additional_return_vals:{return_vals} is not defined, exiting sys from MixedDataset"
            stop_sys(message)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.return_func(idx)

    def get_indexes(self):
        return list(range(len(self)))

    def get_filtered_dataset(self, idxs):
        return torch.utils.data.Subset(self, idxs)

    # -------- shared return funcs --------
    def return_func_img_tensor_and_label(self, idx):
        row = self.df.iloc[idx]

        label, img_save_path, is_dummy = row["label"], row["img_save_path"], row["is_dummy"]
        transformed_img = self.load_and_transform_img(img_save_path)

        return transformed_img, label, is_dummy

    def return_func_img_tensor_and_label_golden_label(self, idx):
        row = self.df.iloc[idx]

        label, golden_label, img_save_path, is_dummy = row["label"], row["golden_label"], row["img_save_path"], row["is_dummy"]
        transformed_img = self.load_and_transform_img(img_save_path)

        return transformed_img, label, golden_label, is_dummy

    def return_func_img_tensor_and_label_and_feature_save_path_template(self, idx):
        row = self.df.iloc[idx]

        label, img_save_path, feature_save_path_template, is_dummy = row["label"], row["img_save_path"], row["feature_save_path_template"], row["is_dummy"]
        transformed_img = self.load_and_transform_img(img_save_path)

        return transformed_img, label, feature_save_path_template, is_dummy

    def return_func_img_tensor_and_soft_label(self, idx):
        row = self.df.iloc[idx]
        img_save_path, soft_label, is_dummy = row["img_save_path"], row["soft_label"], row["is_dummy"]
        transformed_img = self.load_and_transform_img(img_save_path)

        return transformed_img, soft_label, is_dummy

    # -------- abstract hooks --------
    def load_and_transform_img(self, img_save_path):
        raise NotImplementedError

class MixedDatasetPILV2(MixedDataset):
    def __init__(self, df, inference_train_related_dict, model_name=None, return_vals="img_tensor_and_label"):
        super().__init__(df, return_vals=return_vals)

        input_size = get_input_size_from_model_name(model_name)
        normalize_params = get_normalize_params_from_model_name(model_name)
        interpolation = InterpolationMode.BICUBIC if "ViT" in model_name else InterpolationMode.BILINEAR

        train = inference_train_related_dict["train"]
        if train:
            last_layer_only = inference_train_related_dict["last_layer_only"]
            if last_layer_only:
                # self.transform = transforms_v2.Compose([
                #     transforms_v2.Resize(
                #         input_size,
                #         interpolation=interpolation,
                #         antialias=True,
                #     ),
                #     transforms_v2.RandAugment(interpolation=interpolation),
                #     transforms_v2.ToImage(),
                #     transforms_v2.ToDtype(torch.float32, scale=True),
                #     transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                # ])
                
                self.transform = transforms_v2.Compose([
                    transforms_v2.Resize(
                        input_size,
                        interpolation=interpolation,
                        antialias=True,
                    ),
                    transforms_v2.RandomHorizontalFlip(p=0.5),
                    transforms_v2.ToImage(),
                    transforms_v2.ToDtype(torch.float32, scale=True),
                    transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                ])
            else:
                google_vit_model_names = [
                    "ViT_Tiny_timm",
                    "ViT_Small_timm", "ViT_Small_32_timm",
                    "ViT_Base_timm", "ViT_Base_32_timm",
                    "ViT_Large_timm", "ViT_Large_32_timm",
                ]
                if model_name in google_vit_model_names:
                    # self.transform = transforms_v2.Compose([
                    #     transforms_v2.Resize(
                    #         input_size,
                    #         interpolation=interpolation,
                    #         antialias=True,
                    #     ),
                    #     transforms_v2.RandomHorizontalFlip(p=0.5),
                    #     # Translation (0.1) allows the 50x50 image to shift slightly without being destroyed
                    #     transforms_v2.RandomAffine(degrees=10, translate=(0.1, 0.1), interpolation=interpolation),
                    #     transforms_v2.RandAugment(num_ops=2, magnitude=5), # Lower magnitude (5 vs 9) for small images
                    #     transforms_v2.ToImage(),
                    #     transforms_v2.ToDtype(torch.float32, scale=True),
                    #     transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                    # ])
                    
                    self.transform = transforms_v2.Compose([
                        transforms_v2.Resize(
                            input_size,
                            interpolation=interpolation,
                            antialias=True,
                        ),
                        transforms_v2.RandomHorizontalFlip(),
                        transforms_v2.RandomCrop(input_size, padding=14),
                        transforms_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms_v2.ToImage(),
                        transforms_v2.ToDtype(torch.float32, scale=True),
                        transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                    ])
                else:
                    # self.transform = transforms_v2.Compose([
                    #     # Pad by 4 pixels then crop back to 50x50 before resizing to model size
                    #     transforms_v2.RandomCrop(50, padding=4), 
                    #     transforms_v2.Resize(input_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                    #     transforms_v2.RandomHorizontalFlip(p=0.5),
                    #     transforms_v2.ColorJitter(brightness=0.1, contrast=0.1), # Safer than RandAugment for tiny images
                    #     transforms_v2.ToImage(),
                    #     transforms_v2.ToDtype(torch.float32, scale=True),
                    #     transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                    # ])
                    
                    self.transform = transforms_v2.Compose([
                        transforms_v2.Resize(
                            input_size,
                            interpolation=interpolation,
                            antialias=True,
                        ),
                        transforms_v2.RandomHorizontalFlip(),
                        transforms_v2.RandomCrop(input_size, padding=14),
                        transforms_v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms_v2.ToImage(),
                        transforms_v2.ToDtype(torch.float32, scale=True),
                        transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
                    ])
        else:
            self.transform = transforms_v2.Compose([
                transforms_v2.Resize(
                    input_size,
                    interpolation=interpolation,
                    antialias=True,
                ),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=normalize_params["mean"], std=normalize_params["std"]),
            ])
        
    def load_and_transform_img(self, img_save_path):
        try:
            img = Image.open(img_save_path).convert("RGB")
        except:
            message = f"Error in loading img from mixed dataset, please check if img:{img_save_path} exists"
            stop_sys(message=message)
        transformed_img = self.transform(img)
        return transformed_img

