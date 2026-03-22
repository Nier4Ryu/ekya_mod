import gc, os, time, ray, torch, numpy as np
from typing import List
from torch.utils.data import Subset
from ekya_update.model_update import MLModelSubstitution
from ekya.microprofilers.base_microprofiler import BaseMicroprofiler
from ekya_update.common import stop_sys, logical_to_physical_gpu


def microprofile(hyperparameters: dict,
                 epochs: dict,
                 dataloaders: dict,
                 res_alloc: float,
                 pretrained_model_path: str,
                 device: str,
                 task_num,
                 camera_idx:int=None,
                 log_dir:str=None,
                 ):
    res_allocation_percentage = res_alloc*100
    hyperparameters['epochs'] = epochs
    model = MLModelSubstitution(hyperparameters=hyperparameters,
                    gpu_allocation_percentage=res_allocation_percentage,
                    restore_path=pretrained_model_path,
                    device=device,
                    camera_idx=camera_idx,
                    log_dir=log_dir,
                    )
    start_time = time.time()
    results = model.retrain_model(train_loader=dataloaders['train'],
                        val_loader=dataloaders['val'],
                        test_loader=dataloaders['test'],
                        hyperparameters=hyperparameters,
                        profiling_mode=True,
                        task_num=task_num,
                        save_model=False, # For micro profiling do not save the model!
                        )
    time_taken = time.time() - start_time
    best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_results = results
    time_per_epoch = misc_results['per_epoch_avg_time']
    #init_time = time_taken - time_per_epoch * epochs
    init_time = misc_results['init_time']
    ret_val = {
            'best_val_acc': best_val_acc,
            'hyperparameters': hyperparameters,
            'init_time': init_time,
            'time_per_epoch': time_per_epoch,
            'preretrain_test_acc': profile_preretrain_test_acc,
            'test_acc': profile_test_acc
        }
    return ret_val

def subsample_dataloader(dataloader: torch.utils.data.DataLoader,
                         subsample_rate: float):
    # if dataloader is None:
    #     return None
    dataset = dataloader.dataset
    subsampled_dataset = resample_substitution(dataset=dataset, num_to_subsample=int(subsample_rate * len(dataset)))
    subsampled_dataloader = torch.utils.data.DataLoader(subsampled_dataset,
                                                        batch_size=dataloader.batch_size,
                                                        shuffle=False,
                                                        num_workers=0)
    return subsampled_dataloader

def resample_substitution(dataset, num_to_subsample):
    current_num_samples = len(dataset)
    print(f"Resampling from {current_num_samples} to {num_to_subsample} samples")
    
    indices = np.arange(current_num_samples)
    
    if current_num_samples >= num_to_subsample:
        # Subsampling: Randomly pick unique indices
        resample_idxs = np.random.choice(indices, num_to_subsample, replace=False)
    else:
        # Supersampling: Repeat indices to reach target size
        # This handles both the full repetitions and the remainder
        resample_idxs = np.random.choice(indices, num_to_subsample, replace=True)
        
    return Subset(dataset, resample_idxs.tolist())

@ray.remote
class MicroprofileActor:
    """Ray actor for microprofiling. Killing the actor releases the CUDA context."""
    def run(self, hyperparameters, epochs, dataloaders, res_alloc, pretrained_model_path, device, task_num, camera_idx=None, log_dir=None):
        return microprofile(hyperparameters, epochs, dataloaders, res_alloc, pretrained_model_path, device, task_num, camera_idx, log_dir)


class SimpleMicroprofilerSubstitution(BaseMicroprofiler):
    def __init__(self, device='cuda'):
        assert device in ['cuda', 'cpu', 'auto']
        self.device = device

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def run_microprofiling(self,
                           candidate_hyperparams: List[dict],
                           dataloaders: List[dict],
                           resources: float,
                           epochs: int,
                           pretrained_model_path:str,
                           subsample_rate: float = 1,
                           task_num = None,
                           camera_idx:int=None,
                           log_dir:str=None,
                           model_name: str = None,
                           ) -> dict:
        assert len(dataloaders) == len(candidate_hyperparams)
        results = []

        # Parallelism degree depends on model size:
        # ResNet-family: up to 4 concurrent workers (smaller GPU footprint)
        # ViT-family: up to 2 concurrent workers (larger GPU footprint)
        # if model_name and "ViT" in model_name:
        #     max_parallel = 2
        # else:
        #     max_parallel = 4
        max_parallel = 2
        resources_per_trial = resources / max_parallel

        if self.device == 'cpu':
            message = "profiling in CPU is impossible in my standard, what is this??"
            stop_sys(message, raise_error=True)
        else:
            gpu_idx = 0
            physical_gpu = logical_to_physical_gpu(gpu_idx)
            gpu_percent = int(resources_per_trial*100)
            print(f"[Micro profiling] Allocation:{gpu_percent} Allocated")

            # Submit all tasks upfront — Ray schedules at most "N" concurrently
            # due to the GPU0 resource constraint
            future_to_actor = {}
            futures = []
            for hp, hp_dataloaders in zip(candidate_hyperparams, dataloaders):
                subsampled_dataloaders = {mode: subsample_dataloader(d, subsample_rate) for mode, d in hp_dataloaders.items()}
                actor = MicroprofileActor.options(
                    num_gpus=0.01,
                    resources={f"GPU{gpu_idx}": resources_per_trial},
                    runtime_env={
                        "env_vars": {
                            "CUDA_VISIBLE_DEVICES": physical_gpu,
                            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(gpu_percent)
                        }
                    }
                ).remote()
                future = actor.run.remote(
                    hp, epochs, subsampled_dataloaders, resources_per_trial, pretrained_model_path, self.device, task_num, camera_idx, log_dir
                )
                future_to_actor[future] = actor
                futures.append(future)

            # Drain results one-by-one as each finishes — no idle waiting
            remaining = list(futures)
            while remaining:
                done, remaining = ray.wait(remaining, num_returns=1)
                result = ray.get(done[0])
                ray.kill(future_to_actor[done[0]], no_restart=True)
                results.append(result)

        best_result = max(results, key=lambda i: i['test_acc'])
        return best_result, results