"""
This is a substitution of 
'baselines/ekya/ekya/classes/camera.py' 
consider the previous version as deprecated

As trying to follow the previous version, some names may sound strange
(ex: test samples comes from train_sample_list / train is called pre-train)
"""
import pandas as pd, ray, os, torch, time
from ray.exceptions import RayActorError
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms_v2
from torchvision.transforms.functional import InterpolationMode
from ekya.CONFIG import RANDOM_SEED
from ekya_update.model_update import RayMLModel
from ekya_update.common import atomic_to_csv, stop_sys, apply_ast_on_col, check_mps_is_running

class CameraSubstitution(object):
    # Updated some part at least
    def __init__(self, 
                # Identification
                id, 

                # Dataset Related
                dataset_name,
                train_sample_list_path,
                test_sample_list_path,
                test_golden_sample_list_path,

                # Dataset Split related
                num_tasks,
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
                max_inference_resources,

                training_memory_footprint_value,
                inference_memory_footprint_value,

                # Logging related
                log_dir,
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
            self.test_df = pd.read_csv(test_sample_list_path)
            self.test_golden_df = pd.read_csv(test_golden_sample_list_path)
            fin_t = time.time()
            print(f"loading dfs took {fin_t-start_t}s for camera:{self.id}")


            # Save additional infos
            self.dataset_name = dataset_name
            self.num_tasks = num_tasks
            self.num_chunks = num_chunks
            self.start_task = start_task
            self.termination_task = termination_task

            self.train_split = train_split
            self.inference_profile_path = inference_profile_path
            self.max_inference_resources = max_inference_resources

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

            # Internal params to keep track of where to load...
            self.current_task = -1
            self.num_samples_per_task = int(len(self.test_df)/self.num_tasks)
            
            print("Warning: As realtime inference scaling function is not aquireable without runtime profiling, we use the default ver - labmda x:1")
            self.inference_scaling_function = lambda x: 1  

    def update_training_model(self,
                              hyperparameters: dict,
                              training_gpu_weight: float,
                              ray_resource_demand: float,
                              restore_path: str = "",
                              blocking: bool = False):
        self.hyperparameters = hyperparameters
        self.training_gpu_weight = training_gpu_weight
        self.training_ray_demand = ray_resource_demand
        
        # Kill the existing model by waiting it to terminate any running tasks
        if hasattr(self, "training_model"):
            try:
                ray.get(self.training_model.__ray_terminate__.remote())
                ray.kill(self.training_model, no_restart=True)
            except RayActorError:
                # Model already killed
                pass
        model_updated = False
        while not model_updated:
            try:
                self.training_model = RayMLModel.options(name="{}_training".format(self.id), num_gpus=self.training_ray_demand).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.training_gpu_weight,
                    restore_path=restore_path,
                    name="{}_training".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
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
                       training_gpu_weight: float,
                       ray_resource_demand: float,
                       dataloaders_dict: dict = {},
                       validation_freq: int = -1,
                       restore_path: str = "",
                       profiling_mode: bool = False,
                       task_num=None,
                       ):
        self.update_training_model(hyperparameters, training_gpu_weight, ray_resource_demand, restore_path=restore_path, blocking=False)
        if not dataloaders_dict:
            dataloaders_dict = self._get_dataloader(self.current_task,
                                                    train_batch_size=hyperparameters["train_batch_size"],
                                                    subsample_rate=hyperparameters["subsample"])
        
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
                        shuffle: bool = False):

        # Generate Test dataset for all cases
        dataloaders_dict = {
            "train":None,
            "val":None,
            "test":get_inference_dataloader_from_df(df=self.test_df.iloc[self.num_samples_per_task * task_id:self.num_samples_per_task * (task_id + 1)], batch_size=test_batch_size),
        }
        
        # Generate Train / Val datasets for task_id>0
        if task_id>0:
            # Set col "golden_label" for test_df
            if self.label_type=="golden_label":
                temp_test_df = self.test_df.copy()
                temp_test_df.insert(0, "golden_label", self.test_golden_df["pred"].values)
            else:
                temp_test_df = self.test_df.copy()
                
            # Generate Train / Val dfs
            # Sample from Test samples (Use previous window history only)
            train_and_val_candidate_from_test_indexes = temp_test_df.iloc[self.num_samples_per_task * (task_id - 1):self.num_samples_per_task * task_id]
            num_samples_to_pick_from_test = max(int(len(train_and_val_candidate_from_test_indexes) * subsample_rate), 2)
            num_samples_to_pick_from_test_for_train = int(num_samples_to_pick_from_test * self.train_split)
            train_and_val_from_test_indexes = train_and_val_candidate_from_test_indexes.sample(n=num_samples_to_pick_from_test, random_state=RANDOM_SEED).index
            train_from_test_indexes = train_and_val_from_test_indexes[:num_samples_to_pick_from_test_for_train]
            val_from_test_indexes  = train_and_val_from_test_indexes[num_samples_to_pick_from_test_for_train:]
            train_from_test_df = temp_test_df.loc[train_from_test_indexes]
            val_from_test_df = temp_test_df.loc[val_from_test_indexes]
            
            # Sample from Train samples (Use all of train -> May need to cut the bounds... lets cut up to "num_samples_to_pick_from_test")
            train_and_val_candidate_from_train_indexes = self.train_df
            num_samples_to_pick_from_train = min(max(int(len(train_and_val_candidate_from_train_indexes) * subsample_rate), 2), num_samples_to_pick_from_test)
            num_samples_to_pick_from_train_for_train = int(num_samples_to_pick_from_train * self.train_split)
            train_and_val_from_train_indexes = train_and_val_candidate_from_train_indexes.sample(n=num_samples_to_pick_from_train, random_state=RANDOM_SEED).index
            train_from_train_indexes = train_and_val_from_train_indexes[:num_samples_to_pick_from_train_for_train]
            val_from_train_indexes = train_and_val_from_train_indexes[num_samples_to_pick_from_train_for_train:]
            train_from_train_df = self.train_df.loc[train_from_train_indexes]
            val_from_train_df = self.train_df.loc[val_from_train_indexes]
            
            train_df = pd.concat([train_from_test_df, train_from_train_df])
            train_df_save_path = os.path.join(self.log_dir, "runtime_logs", "train", f"task_{task_id}.csv")
            val_df = pd.concat([val_from_test_df, val_from_train_df])
            val_df_save_path = os.path.join(self.log_dir, "runtime_logs", "val", f"task_{task_id}.csv")
            
            # Save train / val dfs
            atomic_to_csv(train_df, path=train_df_save_path)
            atomic_to_csv(val_df, path=val_df_save_path)
            
            # Swap label to golden labels
            message = "The below golden_label part seems to have a bug, look at it, fix it and than run!"
            stop_sys(message, raise_error=True)
            if self.label_type=="golden_label":
                if "golden_label" not in train_df.columns or "golden_label" not in val_df.columns:
                    message = "Error! You need to have golden_label in both the train_df and val_df to use label_type=golden_label"
                    stop_sys(message, raise_error=True)
                else:
                    train_df["label"] = train_df["golden_label"].fillna(train_df["label"])
                    train_df.drop(columns=["label"], inplace=True)
                    train_df.rename(columns={"golden_label": "label"}, inplace=True)
                    val_df["label"] = val_df["golden_label"].fillna(val_df["label"])
                    val_df.drop(columns=["label"], inplace=True)
                    val_df.rename(columns={"golden_label": "label"}, inplace=True)
            else:
                # Try to drop golden_label if it exists
                if "golden_label" in train_df.columns:
                    train_df.drop(columns=["golden_label"], inplace=True)
                    
                if "golden_label" in val_df.columns:
                    val_df.drop(columns=["golden_label"], inplace=True)
                
            dataloaders_dict["train"] = get_train_dataloader_from_df(df=train_df, batch_size=train_batch_size, num_workers=num_workers)
            dataloaders_dict["val"] = get_inference_dataloader_from_df(df=val_df, batch_size=test_batch_size, num_workers=num_workers)

        return dataloaders_dict

    def set_current_task(self, new_current_task: int):
        self.current_task = new_current_task

    
    def update_inference_model(self,
                               hyperparameters: dict,
                               inference_gpu_weight: float,
                               ray_resource_demand: float,
                               restore_path: str = "",
                               blocking: bool = False):
        self.hyperparameters = hyperparameters
        self.inference_gpu_weight = inference_gpu_weight
        self.inference_ray_demand = ray_resource_demand
        
        self.current_inference_path = restore_path or self.current_inference_path  # Use restore path if specified, else use current_inference_path
        
        # Kill the existing model by waiting it to terminate any running tasks
        if hasattr(self, "inference_model"):
            try:
                ray.get(self.inference_model.__ray_terminate__.remote())
            except RayActorError as e:
                # Model already killed
                print("Got exception while killing model. Continuing. {}".format(str(e)))
                pass
        model_updated = False
        while not model_updated:
            try:
                self.inference_model = RayMLModel.options(name="{}_inference".format(self.id), num_gpus=self.inference_ray_demand).remote(
                    hyperparameters=self.hyperparameters,
                    gpu_allocation_percentage=self.inference_gpu_weight,
                    inference_scaling_function=self.inference_scaling_function,
                    restore_path=self.current_inference_path,
                    name="{}_inference".format(self.id),
                    camera_idx=self.camera_idx,
                    log_dir=self.log_dir,
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
        # Kill training model after it is saved.
        try:
            ray.get(self.training_model.__ray_terminate__.remote())
        except RayActorError as e:
            # Model already killed
            print("Got exception while killing model. Continuing. {}".format(str(e)))
            pass
        self.current_inference_path = path
        print("{}, request update {}".format(self.id, self))
        self.update_inference_model(self.hyperparameters,
                                    self.inference_gpu_weight,
                                    self.inference_ray_demand,
                                    restore_path=self.current_inference_path)
    
    def training_memory_footprint(self):
        return self.training_memory_footprint_value
        
    def inference_memory_footprint(self):
        return self.inference_memory_footprint_value
        
    # Original Logic Kept

    # Not touched

    # Deprecated...
    @staticmethod
    def get_infer_profile(max_inference_resources=1,
                          profile_path='real_inference_profiles.csv',
                          camera='c1'):
        message = f"This function is deprecated, please check if this is really needed"
        stop_sys(message, raise_error=True)

    def setup_dataset(self):
        message = f"This function is deprecated, please check if this is really needed"
        stop_sys(message, raise_error=True)
    
# Implementations from Lite
def get_train_dataloader_from_df(df, batch_size, input_size=(224,224), return_vals="img_tensor_and_label", num_workers=4):
    mixed_dataset = MixedDatasetPILV2(df, train=True, input_size=input_size, return_vals=return_vals)
    dataloader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=1, pin_memory=True, persistent_workers=True) # persistent_workers=True for train function as dataset is static
    return dataloader

def get_inference_dataloader_from_df(df, batch_size, input_size=(224, 224), return_vals="img_tensor_and_label", task_num=None, total_task_num=None, num_workers=4, prefetch_factor=1):
    df = get_df_partition_from_params(df, task_num=task_num, total_task_num=total_task_num)
    mixed_dataset = MixedDatasetPILV2(df, train=False, input_size=input_size, return_vals=return_vals)
    dataloader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)
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
    def __init__(self, df, return_vals="img_tensor_and_label"):
        # Decide cols to keep (Reduce memory), return_func (What values are required)
        self.df = df.copy()
        cols_to_keep = []
        if return_vals=="img_tensor_and_label":
            cols_to_keep.extend(["label", "img_save_path"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_label
        elif return_vals=="img_tensor_and_label_and_feature_save_path_template":
            cols_to_keep.extend(["label", "img_save_path", "feature_save_path_template"])
            self.df = self.df[cols_to_keep]
            self.return_func = self.return_func_img_tensor_and_label_and_feature_save_path_template
        elif return_vals=="img_tensor_and_soft_label":
            cols_to_keep.extend(["img_save_path", "soft_label"])
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
        
        label, img_save_path = row["label"], row["img_save_path"]
        transformed_img = self.load_and_transform_img(img_save_path)
        
        return transformed_img, label
    
    def return_func_img_tensor_and_label_and_feature_save_path_template(self, idx):
        row = self.df.iloc[idx]
        
        label, img_save_path, feature_save_path_template = row["label"], row["img_save_path"], row["feature_save_path_template"]
        transformed_img = self.load_and_transform_img(img_save_path)
        
        return transformed_img, label, feature_save_path_template

    def return_func_img_tensor_and_soft_label(self, idx):
        row = self.df.iloc[idx]
        img_save_path, soft_label = row["img_save_path"], row["soft_label"]
        transformed_img = self.load_and_transform_img(img_save_path)
        
        return transformed_img, soft_label

    # -------- abstract hooks --------
    def load_and_transform_img(self, img_save_path):
        raise NotImplementedError

class MixedDatasetPILV2(MixedDataset):
    def __init__(self, df, train, input_size=(224,224), return_vals="img_tensor_and_label"):
        super().__init__(df, return_vals=return_vals)
        
        if train:
            self.transform = transforms_v2.Compose([
                transforms_v2.Resize(
                    input_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms_v2.RandAugment(interpolation=InterpolationMode.BILINEAR),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms_v2.Compose([
                transforms_v2.Resize(
                    input_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
    def load_and_transform_img(self, img_save_path):
        try:
            img = Image.open(img_save_path).convert("RGB")
        except:
            message = f"Error in loading img from mixed dataset, please check if img:{img_save_path} exists"
            stop_sys(message=message)
        transformed_img = self.transform(img)
        return transformed_img
