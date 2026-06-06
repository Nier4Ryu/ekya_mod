import copy, os, time, uuid, torch, ray, pandas as pd
from collections import defaultdict
from lite.src.core.functions_train_inference_for_detection import (
    get_detection_prediction_samples_from_outputs_and_metadata_list_from_params,
    get_detection_train_components_from_params,
)
from lite.src.models.models_for_detection import (
    get_default_detection_model_from_params,
    set_detection_model_freeze_regions_from_params,
)
from lite.src.utils.common import atomic_json_dump
from ekya_update.common import atomic_to_csv, atomic_torch_save, save_dict_as_temp_file, stop_sys
from ekya_update.detection_dataset_update import get_detection_inference_dataloader_from_dataset_dict
from ekya_update.detection_metrics_update import get_detection_map_metrics_from_dataset_dicts


class MLDetectionModelSubstitution(object):
    def __init__(self,
                 hyperparameters: dict,
                 gpu_allocation_percentage: float,
                 restore_path: str = None,
                 device: str = "auto",
                 name: str = "unnamed_detection",
                 camera_idx: int = None,
                 log_dir: str = None,
                 label_type: str = "golden_label"):
        self.hyperparameters = hyperparameters
        self.gpu_allocation_percentage = gpu_allocation_percentage
        self.restore_path = restore_path
        self.label_type = label_type
        self.name = name
        self.camera_idx = camera_idx
        self.log_dir = log_dir
        
        print("Initializing detection model {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))
        self.model = EkyaDetectionModelWrapper(
            hyperparameters=self.hyperparameters,
            restore_path=self.restore_path,
            device=device,
            camera_idx=self.camera_idx,
            log_dir=self.log_dir,
        )
        print("Initializing detection model done {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))
    
    def retrain_model(self,
                      train_loader,
                      val_loader,
                      test_loader,
                      hyperparameters: dict,
                      validation_freq: int = -1,
                      profiling_mode=False,
                      task_num=None,
                      save_model=True,
                      window_end_time=None):
        total_start_time = time.time()
        num_epochs = int(hyperparameters["epochs"])
        learning_rate = float(hyperparameters["learning_rate"])
        momentum = float(hyperparameters.get("momentum", 0.9))
        
        if validation_freq == -1:
            validation_freq = num_epochs
        else:
            pass
        
        dataloaders_dict = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        
        profile_preretrain_test_map = None
        profile_test_map = None
        if profiling_mode:
            profile_preretrain_test_map, profile_prediction_dataset_dict, profile_metrics = self.model.infer(test_loader)
        else:
            pass
        
        retrain_start_time = time.time()
        train_results = self.model.train_model(
            dataloaders=dataloaders_dict,
            num_epochs=num_epochs,
            lr=learning_rate,
            momentum=momentum,
            validation_freq=validation_freq,
            task_num=task_num,
            save_model=save_model,
            window_end_time=window_end_time,
            train_gpu_fraction=self.gpu_allocation_percentage,
        )
        model, val_map_history, best_val_map, profile, subprofile_test_results, misc_return = train_results
        retrain_end_time = time.time()
        
        if profiling_mode:
            profile_test_map, profile_prediction_dataset_dict, profile_metrics = self.model.infer(test_loader)
        else:
            pass
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        misc_return["total_time"] = total_time
        misc_return["init_time"] = total_time - num_epochs*misc_return["per_epoch_avg_time"]
        misc_return["retrain_time"] = retrain_end_time - retrain_start_time
        
        return best_val_map, profile, subprofile_test_results, profile_preretrain_test_map, profile_test_map, misc_return
    
    def test_map(self, test_loader, resource_scaled=True, task_num=None, chunk_id=None):
        test_map, prediction_dataset_dict, metrics = self.model.infer(test_loader)
        prediction_dataset_temp_path = self.save_prediction_dataset_dict_as_temp_json(prediction_dataset_dict)
        log_items = {
            "map": [metrics["map"]],
            "map50": [metrics["map50"]],
            "map75": [metrics["map75"]],
            "mar100": [metrics["mar100"]],
            "num_images": [metrics["num_images"]],
            "num_gt_boxes": [metrics["num_gt_boxes"]],
            "num_prediction_boxes": [metrics["num_prediction_boxes"]],
            "prediction_dataset_temp_path": [prediction_dataset_temp_path],
            "task_num": [task_num],
            "chunk_id": [chunk_id],
        }
        log_items_path = save_dict_as_temp_file(log_items, self.log_dir)
        
        return test_map, log_items_path
    
    def infer_dataset_dict(self, dataset_dict, batch_size=1, num_workers=4, score_threshold=None):
        dataloader = get_detection_inference_dataloader_from_dataset_dict(
            dataset_dict=dataset_dict,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        map_score, prediction_dataset_dict, metrics = self.model.infer(
            dataloader=dataloader,
            score_threshold=score_threshold,
        )
        
        return prediction_dataset_dict, metrics
    
    def save_prediction_dataset_dict_as_temp_json(self, prediction_dataset_dict):
        temp_dir = os.path.join(self.log_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{time.time_ns()}_prediction.json")
        atomic_json_dump(prediction_dataset_dict, temp_path, indent=2)
        
        return temp_path
    
    def save_model(self, path):
        self.model.save(path)
    
    def ready(self):
        return True
    
    def get_gpu_allocation(self):
        return self.gpu_allocation_percentage


RayMLDetectionModel = ray.remote(num_gpus=0.01)(MLDetectionModelSubstitution)


class EkyaDetectionModelWrapper(object):
    def __init__(self, hyperparameters, restore_path=None, device="auto", camera_idx=None, log_dir=None):
        self.hyperparameters = hyperparameters
        self.camera_idx = camera_idx
        self.log_dir = log_dir
        
        if device=="auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = get_default_detection_model_from_params(
            detection_model_name=hyperparameters["detection_model_name"],
            feature_extractor_name=hyperparameters["feature_extractor_name"],
            pretrained=hyperparameters.get("pretrained", True),
            num_out=hyperparameters["num_classes"],
            weight_path=restore_path,
            device=self.device,
            min_size=hyperparameters.get("min_size", 224),
            max_size=hyperparameters.get("max_size", 224),
            layers_to_use=hyperparameters.get("layers_to_use", 1),
            allow_partial_weight_load=hyperparameters.get("allow_partial_weight_load", False),
        )
        freeze_region_names = hyperparameters.get("freeze_region_names", None)
        if freeze_region_names is not None:
            set_detection_model_freeze_regions_from_params(
                model=self.model,
                freeze_region_names=freeze_region_names,
            )
        else:
            pass
    
    def train_model(self,
                    dataloaders,
                    subprofile_test_epochs=None,
                    num_epochs=1,
                    lr=0.001,
                    momentum=0.9,
                    validation_freq=1,
                    task_num=None,
                    save_model=True,
                    window_end_time=None,
                    train_gpu_fraction=None):
        since = time.time()
        optimizer, lr_scheduler = get_detection_train_components_from_params(
            model=self.model,
            optimizer_name=self.hyperparameters.get("optimizer_name", "SGD"),
            lr=lr,
            lr_scheduler_name=self.hyperparameters.get("lr_scheduler_name", "StepLR"),
            num_epochs=num_epochs,
        )
        
        if validation_freq > num_epochs or validation_freq < 1:
            message = "Detection validation frequency can be at most num_epochs and at least one."
            stop_sys(message, raise_error=True)
        else:
            pass
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_map = 0.0
        best_epoch = 0
        val_map_history = []
        profile = []
        subprofile_test_results = {}
        per_epoch_times = []
        
        if dataloaders["train"] is None:
            has_train_loader = False
        elif len(dataloaders["train"].dataset)==0:
            has_train_loader = False
        else:
            has_train_loader = True
        
        if has_train_loader:
            print("Detection training with {} samples.".format(len(dataloaders["train"].dataset)))
        else:
            print("Detection training skipped because no non-empty target samples were selected.")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            profile_data = defaultdict(lambda: defaultdict(lambda: 0))
            
            if has_train_loader:
                train_metrics = self.run_train_epoch(
                    dataloader=dataloaders["train"],
                    optimizer=optimizer,
                    num_batches_per_epoch=self.hyperparameters.get("num_batches_per_epoch", None),
                )
                if lr_scheduler is not None:
                    lr_scheduler.step()
                else:
                    pass
                profile_data["train"]["time"] = train_metrics["time"]
                profile_data["train"]["loss"] = train_metrics["loss"]
                profile_data["train"]["map"] = 0.0
                profile_data["train"]["num_samples"] = train_metrics["num_samples"]
            else:
                profile_data["train"]["time"] = 0.0
                profile_data["train"]["loss"] = 0.0
                profile_data["train"]["map"] = 0.0
                profile_data["train"]["num_samples"] = 0
            
            should_validate = True if epoch==num_epochs-1 else epoch!=0 and epoch%validation_freq==validation_freq-1
            if should_validate and dataloaders["val"] is not None and len(dataloaders["val"].dataset)>0:
                val_map, val_prediction_dataset_dict, val_metrics = self.infer(dataloaders["val"])
                val_map_history.append(val_map)
                profile_data["val"]["time"] = val_metrics["elapsed_time"]
                profile_data["val"]["loss"] = 0.0
                profile_data["val"]["map"] = val_map
                profile_data["val"]["num_samples"] = val_metrics["num_images"]
                if val_map > best_val_map:
                    best_val_map = val_map
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                else:
                    pass
            else:
                pass
            
            profile_this_epoch = [epoch_start_time]
            for phase in ["train", "val", "test"]:
                for metric in ["time", "loss", "map", "num_samples"]:
                    profile_this_epoch.append(profile_data.get(phase, {}).get(metric, 0))
            profile.append(profile_this_epoch)
            per_epoch_times.append(time.time()-epoch_start_time)
        
        if len(val_map_history)==0 and dataloaders["val"] is not None and len(dataloaders["val"].dataset)>0:
            best_val_map, val_prediction_dataset_dict, val_metrics = self.infer(dataloaders["val"])
            best_model_wts = copy.deepcopy(self.model.state_dict())
        else:
            pass
        
        if len(per_epoch_times)>0:
            per_epoch_avg_time = sum(per_epoch_times) / len(per_epoch_times)
        else:
            per_epoch_avg_time = 0.0
        
        time_elapsed = time.time() - since
        print("Detection training complete in {:.0f}m {:.0f}s. Best val mAP: {:.4f}".format(
            time_elapsed // 60,
            time_elapsed % 60,
            best_val_map,
        ))
        
        self.model.load_state_dict(best_model_wts)
        misc_return = {"per_epoch_avg_time": per_epoch_avg_time}
        
        if save_model:
            training_fin_time = time.time()
            if window_end_time is not None:
                idle_time = window_end_time - training_fin_time
            else:
                idle_time = None
            weight_save_path = os.path.join(self.log_dir, str(self.camera_idx), f"{task_num}.pt")
            atomic_torch_save(self.model.state_dict(), weight_save_path)
            self.log_model_train_history(
                weight_save_path=weight_save_path,
                task_num=task_num,
                best_epoch=best_epoch,
                num_epochs=num_epochs,
                lr=lr,
                momentum=momentum,
                num_samples=len(dataloaders["train"].dataset) if dataloaders["train"] is not None else None,
                idle_time=idle_time,
                train_gpu_fraction=train_gpu_fraction,
                best_val_map=best_val_map,
            )
        else:
            pass
        
        return self.model, val_map_history, float(best_val_map), profile, subprofile_test_results, misc_return
    
    def run_train_epoch(self, dataloader, optimizer, num_batches_per_epoch=None):
        start_time = time.time()
        self.model.train()
        loss_sum = 0.0
        num_batches = 0
        num_samples = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            should_train_batch = True if num_batches_per_epoch is None else batch_idx<num_batches_per_epoch
            if should_train_batch:
                images = [image.to(self.device, non_blocking=True) for image in images]
                targets = [
                    {key:value.to(self.device, non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                loss = sum(loss_value for loss_value in loss_dict.values())
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                num_batches += 1
                num_samples += len(images)
            else:
                pass
        
        if num_batches>0:
            avg_loss = loss_sum / num_batches
        else:
            avg_loss = 0.0
        
        train_metrics = {
            "time": time.time()-start_time,
            "loss": avg_loss,
            "num_samples": num_samples,
        }
        
        return train_metrics
    
    def infer(self, dataloader, reference_dataset_dict=None, score_threshold=None):
        self.model.eval()
        records = []
        prediction_samples = []
        num_prediction_boxes = 0
        source_dataset_dict = dataloader.dataset.dataset_dict
        source_samples = source_dataset_dict["samples"]
        inference_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, targets, metadata_list) in enumerate(dataloader):
                images = [image.to(self.device, non_blocking=True) for image in images]
                outputs = self.model(images)
                if score_threshold is not None:
                    outputs = [
                        self.get_thresholded_detection_output(output=output, score_threshold=score_threshold)
                        for output in outputs
                    ]
                else:
                    pass
                batch_prediction_samples, batch_num_prediction_boxes = get_detection_prediction_samples_from_outputs_and_metadata_list_from_params(
                    outputs=outputs,
                    metadata_list=metadata_list,
                    source_samples=source_samples,
                    score_threshold=None,
                )
                prediction_samples.extend(batch_prediction_samples)
                num_prediction_boxes += batch_num_prediction_boxes
        
        prediction_dataset_dict = copy.deepcopy(source_dataset_dict)
        prediction_dataset_dict["samples"] = prediction_samples
        elapsed_time = time.time() - inference_start_time
        prediction_dataset_dict["prediction_metadata"] = {
            "elapsed_time": elapsed_time,
            "num_prediction_frames": len(prediction_samples),
            "num_prediction_boxes": num_prediction_boxes,
            "score_threshold": score_threshold,
        }
        
        if reference_dataset_dict is None:
            metric_reference_dataset_dict = source_dataset_dict
        else:
            metric_reference_dataset_dict = reference_dataset_dict
        
        metrics = get_detection_map_metrics_from_dataset_dicts(
            reference_dataset_dict=metric_reference_dataset_dict,
            prediction_dataset_dict=prediction_dataset_dict,
        )
        metrics["elapsed_time"] = elapsed_time
        map_score = float(metrics["map"])
        print("Detection inference done. mAP: {:.4f}, AP50: {:.4f}".format(metrics["map"], metrics["map50"]))
        
        return map_score, prediction_dataset_dict, metrics
    
    def get_thresholded_detection_output(self, output, score_threshold):
        keep_mask = output["scores"]>=score_threshold
        thresholded_output = {
            "boxes": output["boxes"][keep_mask],
            "labels": output["labels"][keep_mask],
            "scores": output["scores"][keep_mask],
        }
        
        return thresholded_output
    
    def log_model_train_history(self,
                                weight_save_path,
                                task_num,
                                best_epoch,
                                num_epochs,
                                lr,
                                momentum,
                                num_samples,
                                idle_time,
                                train_gpu_fraction,
                                best_val_map):
        model_train_history_df_path = os.path.join(self.log_dir, str(self.camera_idx), "model_train_history.csv")
        df = pd.DataFrame({
            "weight_save_path": [weight_save_path],
            "task_num": [task_num],
            "chunk_num": [None],
            "best_epoch": [best_epoch],
            "num_epochs": [num_epochs],
            "lr": [lr],
            "momentum": [momentum],
            "num_samples": [num_samples],
            "idle_time": [idle_time],
            "train_gpu_fraction": [train_gpu_fraction],
            "best_val_map": [best_val_map],
        })
        if os.path.exists(model_train_history_df_path):
            prev_df = pd.read_csv(model_train_history_df_path)
            new_df = pd.concat([prev_df, df], ignore_index=True)
        else:
            new_df = df
        atomic_to_csv(new_df, model_train_history_df_path)
    
    def save(self, path):
        atomic_torch_save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
