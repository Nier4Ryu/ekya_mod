import gc, torch
from typing import List
from ekya.microprofilers.base_microprofiler import BaseMicroprofiler
from ekya_update.model_detection_update import MLDetectionModelSubstitution


def microprofile_detection(hyperparameters,
                           epochs,
                           dataloaders,
                           res_alloc,
                           pretrained_model_path,
                           device,
                           task_num,
                           camera_idx=None,
                           log_dir=None):
    microprofile_hyperparameters = hyperparameters.copy()
    microprofile_hyperparameters["epochs"] = int(epochs)
    model = MLDetectionModelSubstitution(
        hyperparameters=microprofile_hyperparameters,
        gpu_allocation_percentage=res_alloc*100,
        restore_path=pretrained_model_path,
        device=device,
        name="detection_microprofile",
        camera_idx=camera_idx,
        log_dir=log_dir,
    )
    
    if dataloaders["val"] is not None and len(dataloaders["val"].dataset)>0:
        preretrain_val_map, prediction_dataset_dict, metrics = model.model.infer(dataloaders["val"])
    else:
        preretrain_val_map = 0.0
    
    results = model.retrain_model(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        hyperparameters=microprofile_hyperparameters,
        validation_freq=epochs,
        profiling_mode=False,
        task_num=task_num,
        save_model=False,
    )
    best_val_map, profile, subprofile_test_results, profile_preretrain_test_map, profile_test_map, misc_results = results
    
    if dataloaders["val"] is not None and len(dataloaders["val"].dataset)>0:
        post_val_map, prediction_dataset_dict, metrics = model.model.infer(dataloaders["val"])
    else:
        post_val_map = best_val_map
    
    result = {
        "hyperparameters": microprofile_hyperparameters,
        "init_time": misc_results["init_time"],
        "time_per_epoch": misc_results["per_epoch_avg_time"],
        "preretrain_test_acc": preretrain_val_map,
        "test_acc": post_val_map,
    }
    
    return result


class SimpleDetectionMicroprofilerSubstitution(BaseMicroprofiler):
    def __init__(self, device):
        self.device = device
    
    def run_microprofiling(self,
                           candidate_hyperparams: List[dict],
                           dataloaders: List[dict],
                           resources: float,
                           epochs: int,
                           pretrained_model_path: str,
                           subsample_rate: float = 1,
                           task_num=None,
                           camera_idx: int = None,
                           log_dir: str = None,
                           model_name: str = None):
        assert len(dataloaders) == len(candidate_hyperparams)
        results = []
        
        for hyperparameters, hp_dataloaders in zip(candidate_hyperparams, dataloaders):
            result = microprofile_detection(
                hyperparameters=hyperparameters,
                epochs=epochs,
                dataloaders=hp_dataloaders,
                res_alloc=resources,
                pretrained_model_path=pretrained_model_path,
                device=self.device,
                task_num=task_num,
                camera_idx=camera_idx,
                log_dir=log_dir,
            )
            results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                pass
            gc.collect()
        
        if len(results)>0:
            best_result = max(results, key=lambda result: result["test_acc"])
        else:
            best_result = None
        
        return best_result, results
    
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            pass
        gc.collect()
