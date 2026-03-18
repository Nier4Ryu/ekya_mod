"""
This is a replacement of previous 'baselines/ekya/ekya/models/resnet.py' so as to support other backbones!
"""


import re, torch, torch.nn as nn, torch.nn.functional as F, torchvision.models as models, ray, os, time, numpy as np, torch.optim as optim, copy, pandas as pd, timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, List
from collections import defaultdict
from ekya.utils.mps import set_mps_envvars
from ekya.CONFIG import RANDOM_SEED
from ekya.utils.helpers import seed_all
from ekya_update.common import stop_sys, atomic_to_csv, atomic_torch_save, save_dict_as_temp_file, DINOv3_MODEL_LOCAL_REPO, DINOv3_MODEL_WEIGHT_PATH_BASE

"""
This part is from ekya/classes/models.py
"""
# Set random seed for reproducibility
class MLModelSubstitution(object):
    def __init__(self,
                 hyperparameters: dict,
                 gpu_allocation_percentage: float,
                 inference_scaling_function: callable = lambda x: 1,
                 restore_path: str = "",
                 device: str = 'auto',
                 name='unnamed',
                 camera_idx:int=None,
                 log_dir:str=None,
                 label_type:str="ground_truth",
                 ):
        self.hyperparameters = hyperparameters
        self.gpu_allocation_percentage = gpu_allocation_percentage
        self.restore_path = restore_path
        self.label_type = label_type
        set_mps_envvars(gpu_allocation_percentage)
        print("Initializing {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))
        NUM_CLASSES = self.hyperparameters["num_classes"]
        self.inference_scaling_function = inference_scaling_function    # The input range of the inference scaling function should be 0-1. Need to translate percentage by /100.
        self.name = name
        
        self.camera_idx=camera_idx
        self.log_dir=log_dir

        self.model = EkyaModelWrapper(NUM_CLASSES, hyperparameters=self.hyperparameters, restore_path=self.restore_path, device=device, camera_idx=self.camera_idx, log_dir=self.log_dir)

    def retrain_model(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader,
                      hyperparameters: dict,
                      validation_freq: int = -1,
                      profiling_mode = False,
                      task_num=None,
                      save_model=True,
                      window_end_time=None,
                      ):
        """
        Retrains a model given new dataloader.
        :param train_loader: Dataloader for the train set
        :param val_loader: Dataloader for the validation set
        :return:
        """
        total_start_time = time.time()
        NUM_EPOCHS = hyperparameters["epochs"]
        LR = hyperparameters["learning_rate"]
        MOMENTUM = hyperparameters["momentum"]

        if validation_freq == -1:
            validation_freq = NUM_EPOCHS

        dataloaders_dict = {'train': train_loader,
                            'val': val_loader,
                            'test': test_loader}

        profile_preretrain_test_acc = None
        profile_test_acc = None

        if profiling_mode:
            start_time = time.time()
            profile_preretrain_test_acc, log_items = self.model.infer(dataloaders_dict['test'], label_type=self.label_type)
            infer_time_pre = time.time() - start_time
            print("Profile mode: pre-retrain testing took {} seconds and got acc {:.2f}".format(infer_time_pre, profile_preretrain_test_acc))
            if infer_time_pre > 5:
                print("[WARNING] Inference is taking too long - make preretrain and post retrain testing only in profile mode.")

        retrain_start_time = time.time()
        _, _, best_val_acc, profile, subprofile_test_results, misc_return = self.model.train_model(dataloaders_dict, num_epochs=NUM_EPOCHS, lr=LR,
                                                                                      momentum=MOMENTUM, validation_freq=validation_freq, task_num=task_num, save_model=save_model, label_type=self.label_type,
                                                                                      subprofile_test_epochs={}, window_end_time=window_end_time, train_gpu_fraction=self.gpu_allocation_percentage)
        retrain_end_time = time.time()
        retrain_time = retrain_end_time - retrain_start_time

        if profiling_mode:
            start_time = time.time()
            profile_test_acc, log_items = self.model.infer(dataloaders_dict['test'], label_type=self.label_type)
            infer_time_post = time.time() - start_time
            print("Profile mode: post-retrain testing took {} seconds and got acc {:.2f}".format(infer_time_post, profile_test_acc))
            if infer_time_post > 5:
                print("[WARNING] Inference is taking too long - make preretrain and post retrain testing only in profile mode.")
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        misc_return['total_time'] = total_time
        misc_return['init_time'] = total_time - NUM_EPOCHS*misc_return['per_epoch_avg_time']

        return best_val_acc, profile, subprofile_test_results, profile_preretrain_test_acc, profile_test_acc, misc_return

    def test_acc(self, test_loader: torch.utils.data.DataLoader, resource_scaled=True):
        test_acc, log_items = self.model.infer(test_loader, label_type=self.label_type)
        log_items_path = save_dict_as_temp_file(log_items, self.log_dir)
        # Implement scaling with GPU allocation here.
        if resource_scaled:
            scaling_factor = self.inference_scaling_function(self.gpu_allocation_percentage/100)
            print("[INFERENCE DEBUG] GPU weight = {}. Scaling factor: {}. Orig Accuracy = {}.".format(self.gpu_allocation_percentage/100,
                                                                                                      scaling_factor,
                                                                                                      test_acc))
            scaled_test_acc = scaling_factor * test_acc
        else:
            scaled_test_acc = test_acc
        return scaled_test_acc, log_items_path

    def save_model(self, path, ):
        '''Checkpoint to disk.'''
        self.model.save(path)

    def get_pid(self):
        print("I am PID: {}".format(os.getpid()))
        print("Envvar: {}".format(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "Not Defined")))
        return os.getpid(), os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "Not Defined")

    def ready(self):
        '''
        Dummy function to signal actor is ready.
        :return:
        '''
        return True

    def get_gpu_allocation(self):
        return self.gpu_allocation_percentage

RayMLModel = ray.remote(num_gpus=0.01)(MLModelSubstitution)


"""
This part is to replace /ekya/models/resnet.py
"""

# Wrapping codes so as to fit ekya calling conventions!
class EkyaModelWrapper(object):
    DEFAULT_HYPER_PARAMS = {'num_hidden': 512,
                            'last_layer_only': True,
                            'model_name': "resnet18"}
    def __init__(self, num_classes, pretrained=True, restore_path=None, hyperparameters=None, device='auto', camera_idx=None, log_dir=None):
        # hyper parameters matching process (Adapter)
        self.hyperparameters = hyperparameters if hyperparameters else self.DEFAULT_HYPER_PARAMS
        model_name = hyperparameters["model_name"]
        last_layer_only = hyperparameters["last_layer_only"]
        hidden_layers = hyperparameters["num_hidden"]
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.camera_idx = camera_idx
        self.log_dir = log_dir
        
        # Loading model
        self.model = get_default_model_from_params(model_name=model_name, pretrained=True, hidden_layers=hidden_layers, num_out=num_classes, last_layer_only=last_layer_only, weight_path=restore_path, device=self.device)
        
        # Accumulating only the params to update
        # print("Params to learn:")
        if last_layer_only:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    # print("\t", name)
        else:
            params_to_update = self.model.parameters()
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    # print("\t", name)
                    pass

        self.params_to_update = params_to_update
    
    def train_model(self, dataloaders, subprofile_test_epochs = None, num_epochs=1, lr=0.001, momentum=0.9,
                    validation_freq = 1,
                    task_num=None,
                    save_model=True,
                    label_type="ground_truth",
                    window_end_time=None,
                    train_gpu_fraction=None,
                    ):
        since = time.time()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.params_to_update, lr=lr, momentum=momentum)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        val_acc_history = []
        if validation_freq > num_epochs or validation_freq < 1:
            raise ValueError("Validation frequency can be at most num_epochs or min 1. Else the model will not be updated with best weights.")

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_epoch=0

        profile = []    # List of [timestamp, train metrics, val metrics, test metrics]
        subprofile_test_results = {}

        if subprofile_test_epochs is None:
            if "test" in dataloaders:
                subprofile_test_epochs = {num_epochs-1: {-1: dataloaders["test"]}}   # -1 = current task
            else:
                subprofile_test_epochs = {}

        if dataloaders["train"] is None:
            # 0th task, just run inference for subprofiles
            # Ideally this needs to run just once and assign the same subprofile to all epochs since there's no retraining
            # However, not optimizing this because user may pass different test loaders for different epochs.

            for epoch in subprofile_test_epochs.keys():
                subprofile_test_this_epoch = {}
                for task_id, task_test_loader in subprofile_test_epochs[epoch].items():
                    subprofile_test_this_epoch[task_id], log_items = self.infer(task_test_loader, label_type=label_type)
                subprofile_test_results[epoch] = subprofile_test_this_epoch

        per_epoch_avg_time = 0
        if dataloaders["train"] is not None:
            print("Training with {} samples.".format(len(dataloaders["train"].dataset)))
            sgd_start_time = time.time()
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                profile_data = defaultdict(lambda: defaultdict(lambda: 0))

                if epoch != 0 and epoch % validation_freq == validation_freq-1: # Validation is pointless for the first epoch
                    this_epoch_phases = ["train", "val"]
                else:
                    this_epoch_phases = ["train"]


                # Each epoch has a training and validation phase
                for phase in this_epoch_phases:
                    start_time = time.time()
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    print_frequency = max(len(dataloaders[phase])//10, 10)
                    for batch_idx, (inputs, labels, golden_labels, is_dummy) in enumerate(dataloaders[phase]):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        golden_labels = golden_labels.to(self.device)


                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # Select reference labels based on label_type
                        if label_type == "golden_label":
                            ref_labels = golden_labels
                        else:
                            ref_labels = labels

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, ref_labels)
                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == ref_labels.data)

                        # # Print output at every 10%.
                        # if (batch_idx % print_frequency) == 0:
                        #     print(
                        #         '{} Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                        #             format(phase, epoch, num_epochs, batch_idx * len(inputs), len(dataloaders[phase]) * len(inputs),
                        #                    100. * batch_idx / len(dataloaders[phase]), loss))

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)

                    if phase == 'train':
                        scheduler.step(epoch_loss)

                    end_time = time.time()
                    print('{} epoch {}/{} done. Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch, num_epochs, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val':
                        val_acc_history.append(epoch_acc)
                        if epoch_acc > best_acc:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(self.model.state_dict())
                            best_epoch = epoch
                    profile_data[phase]['time'] = end_time-start_time
                    profile_data[phase]['loss'] = float(epoch_loss)
                    profile_data[phase]['acc'] = float(epoch_acc)
                    profile_data[phase]['num_samples'] = len(dataloaders[phase].dataset)
                profile_this_epoch = [epoch_start_time]
                for phase in ["train", "val", "test"]:
                    for metric in ['time', 'loss', 'acc', 'num_samples']:
                        profile_this_epoch.append(profile_data.get(phase, {}).get(metric, 0))
                profile.append(profile_this_epoch)

                # Epoch done, check if this is a subprofile epoch and run testing on all tasks if required
                if epoch in subprofile_test_epochs.keys():
                    subprofile_test_this_epoch = {}
                    for task_id, task_test_loader in subprofile_test_epochs[epoch].items():
                        subprofile_test_this_epoch[task_id], log_items = self.infer(task_test_loader, label_type=label_type)
                    subprofile_test_results[epoch] = subprofile_test_this_epoch
            sgd_time = time.time() - sgd_start_time
            per_epoch_avg_time = (sgd_time)/num_epochs
            print('{} epochs complete. SGD time: {}. Per epoch time: {}'.format(num_epochs, sgd_time, per_epoch_avg_time))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        misc_return = {'per_epoch_avg_time': per_epoch_avg_time}

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Save the best model weights for this tasks training results!
        if save_model:
            training_fin_time = time.time()
            idle_time = window_end_time - training_fin_time if window_end_time is not None else None
            weight_save_path = os.path.join(self.log_dir, str(self.camera_idx), f"{task_num}.pt")
            atomic_torch_save(self.model.state_dict(), weight_save_path)
            model_train_history_df_path = os.path.join(self.log_dir, str(self.camera_idx), "model_train_history.csv")
            num_samples = len(dataloaders['train'].dataset) if dataloaders['train'] is not None else None
            df = pd.DataFrame(
                {
                    'weight_save_path': [weight_save_path],
                    'task_num': [task_num],
                    'chunk_num': [None],
                    'best_epoch': [best_epoch],
                    'num_epochs': [num_epochs],
                    'lr': [lr],
                    'momentum': [momentum],
                    'num_samples': [num_samples],
                    'idle_time': [idle_time],
                    'train_gpu_fraction': [train_gpu_fraction],
                }
            )
            if os.path.exists(model_train_history_df_path):
                prev_df = pd.read_csv(model_train_history_df_path)
                new_df = pd.concat([prev_df, df], ignore_index=True)
            else:
                new_df = df
            atomic_to_csv(new_df, model_train_history_df_path)
        
        return self.model, val_acc_history, float(best_acc), profile, subprofile_test_results, misc_return
    
    def infer(self, dataloader, label_type="ground_truth"):
        # Actual inference
        results = defaultdict(list)

        self.model.eval()
        running_corrects = 0

        # Iterate over data.
        print_frequency = max(len(dataloader)//10, 10)
        for batch_idx, (inputs, labels, golden_labels, is_dummy) in enumerate(dataloader):
            results["label"].extend(labels.tolist())
            results["golden_label"].extend(golden_labels.tolist())
            results["is_dummy"].extend(is_dummy.tolist())

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            golden_labels = golden_labels.to(self.device, non_blocking=True)

            # forward
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)

                # Save the logits
                results["logits"].extend(outputs.detach().cpu().tolist())

                probs = F.softmax(outputs, dim=1)
                maximum_softmaxs, preds = torch.max(probs, dim=1)
                results["softmax_outputs"].extend(probs.tolist())
                results["maximum_softmax_output"].extend(maximum_softmaxs.tolist())

                # Log results (preds, softmax_outputs)
                results["pred"].extend(preds.tolist())

            # Measure accuracy against golden_label or ground truth based on label_type
            if label_type == "golden_label":
                ref_labels = golden_labels
            else:
                ref_labels = labels
            num_corrects = torch.sum(preds == ref_labels.data)
            running_corrects += num_corrects

            # # Print output at every 10%.
            # if (batch_idx % print_frequency) == 0:
            #     print(  'Infer [{}/{} ({:.0f}%)]\tBatch acc: {:.2f}% \tRunning acc: {:.2f}%'.
            #             format(batch_idx * len(inputs), len(dataloader) * len(inputs),
            #                    100. * batch_idx / len(dataloader),
            #                    float(num_corrects) / len(inputs),
            #                    float(running_corrects) / batch_idx * len(inputs)))

        acc = float(running_corrects) / len(dataloader.dataset)

        print('Inference done. Nnet Acc: {:.2f}'.format(acc))
        return float(acc), results
    
    def save(self, path):
        atomic_torch_save(self.model, path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
    
#####################################################################
# Below codes are basically codes that does not require wrapping!
#####################################################################

# Specialized models
class SimpleMLPModel(nn.Module):
    def __init__(self, input_size=3*224*224, hidden_layers:Union[int, List[int]]=None, out_size=2, starting_layers:list=None):
        super().__init__()
        
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        
        # Set the initial starting_layers
        if starting_layers!=None:
            layers = starting_layers
        else:
            layers = []
        
        # flattening
        layers.append(nn.Flatten())
        
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, out_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, tensors):
        # Forward process
        outputs = self.model(tensors)
        
        # Return outputs
        return outputs


# DinoV2FeatureExtractor is a mod of _LinearClassifierWrapper from DinoV2 official repo, especially for layer=4
class DinoV2FeatureExtractor(nn.Module): 
    def __init__(self, *, feature_extractor: nn.Module, layers: int = 4):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers
        
        if self.layers == 1:
            self.forward_function = self.forward_layer_1
        elif self.layers == 4:
            self.forward_function = self.forward_layer_4
        else:
            message = f"Unsupported number of layers: {self.layers}, exiting sys"
            stop_sys(message, raise_error=True)
        
    def forward(self, tensors):
        features = self.forward_function(tensors)
        return features
    
    def forward_layer_1(self, tensors):
        x = self.feature_extractor.forward_features(tensors)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        linear_input = torch.cat([
            cls_token,
            patch_tokens.mean(dim=1),
        ], dim=1)
        return linear_input
    
    def forward_layer_4(self, tensors):
        x = self.feature_extractor.get_intermediate_layers(tensors, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(dim=1),
        ], dim=1)
        return linear_input

# DinoV3FeatureExtractor is a mod of _LinearClassifierWrapper from DinoV3 official repo
class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, *, feature_extractor:nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        
    def forward(self, tensors):
        x = self.feature_extractor.forward_features(tensors)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        linear_input = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),
            ],
            dim=1,
        )
        return linear_input
        
class FeatureExtractor(nn.Module):
    def __init__(self, feature_extractor_name, pretrained=True):
        super().__init__()
        
        self.feature_extractor_name = feature_extractor_name
        
        # Load model with default_classification_heads
        model = get_model_from_model_name(model_name=self.feature_extractor_name, pretrained=pretrained)
        
        # Remove the default_classification_heads
        self.feature_extractor, self.feature_size = remove_classification_heads_get_feature_size(model_name=self.feature_extractor_name, model=model)
        
        models_requiring_flattening_prefix = ["resnet"]
        if any(self.feature_extractor_name.startswith(model_prefix) for model_prefix in models_requiring_flattening_prefix):
            self.forward_function = self.forward_with_flattening
        elif self.feature_extractor_name.startswith("ViT") and self.feature_extractor_name.endswith("timm"):
            self.forward_function = self.forward_with_extracting
        else:
            self.forward_function = self.forward_with_default
    
    def get_feature_size(self):
        return self.feature_size
        
    def freeze_weights(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, tensors):
        features = self.forward_function(tensors)
        return features
    
    def forward_with_default(self, tensors):
        features = self.feature_extractor(tensors)
        return features
    
    def forward_with_flattening(self, tensors):
        features = self.feature_extractor(tensors)
        flattened_features = torch.flatten(features, 1)
        return flattened_features

    def forward_with_extracting(self, tensors):
        features = self.feature_extractor(tensors)
        extracted_features = features[:, 0]
        return extracted_features

class FullyConnectedLayers(nn.Module):
    def __init__(self, feature_extractor_name:str=None, input_size:int=None, hidden_layers:Union[int, List[int]]=None, out_size:int=None):
        super().__init__()
        
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        
        if feature_extractor_name!=None and input_size==None:
            message = f"This part was designed to get the input size for a fc layer without having to load a model when generating a gating network! As gating networks are currently deprecated, please re-implement the call of gating networks and this part after wards! (This is currently deprecated!!)"
            stop_sys(message, raise_error=True)
            
        elif feature_extractor_name==None and input_size!=None:
            pass # Nothing to be done here, just continue on
        else:
            message=f"Either a feature_extractor_name or a input_size should be given, currently feature_extractor_name:{feature_extractor_name} / input_size:{input_size}"
            stop_sys(message, raise_error=True)
        
        # Build MLP model with input_size / hidden_layers / out_size
        self.fc_layers = SimpleMLPModel(input_size=input_size, hidden_layers=hidden_layers, out_size=out_size)
        
    def forward(self, tensors):
        # Forward process
        outputs = self.fc_layers(tensors)
        
        # Return outputs
        return outputs
        
class FeatureAndOutputModel(nn.Module):
    def __init__(self, model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, return_val="output"):
        super().__init__()
        
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        
        # model
        self.feature_extractor = FeatureExtractor(feature_extractor_name=model_name, pretrained=pretrained)
        if last_layer_only:
            self.feature_extractor.freeze_weights()
        feature_size = self.feature_extractor.get_feature_size()
        
        self.fc_layers = FullyConnectedLayers(input_size=feature_size, hidden_layers=hidden_layers, out_size=num_out)
        
        # Set forward_function depending on return_val
        self.set_return_val(return_val)
    
    def set_return_val(self, return_val):
        self.return_val = return_val
        if self.return_val == "output":
            self.forward_function = self.forward_return_outputs
        elif self.return_val == "feature_and_output":
            self.forward_function = self.forward_return_features_and_outputs
        else:
            message=f"Return val:{return_val} is not defined for forward functions, exiting sys"
            stop_sys(message, raise_error=True)
    
    def get_return_val(self):
        return self.return_val
    
    def forward(self, tensors):
        # As return val could be outputs / features & outputs, just return the val
        return self.forward_function(tensors)
         
    def forward_return_outputs(self, tensors):
        features = self.feature_extractor(tensors)
        outputs = self.fc_layers(features)
        return outputs
    
    def forward_return_features_and_outputs(self, tensors):
        features = self.feature_extractor(tensors)
        outputs = self.fc_layers(features)
        return features, outputs

def get_default_model_from_params(model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, weight_path=None, device='cpu', return_val="output"):
    # Generate Model
    # 1) MoE model case -> str name should be given as {moe_type}|MoEModel|{model_name}
    if "MoEModel" in model_name:
        pattern = r"(?P<moe_type>[^|]+)\|(?P<moe_model>[^|]+)\|(?P<num_experts>\d+)experts\|(?P<model_name>.+)"
        match = re.match(pattern, model_name)
        if match:
            data = match.groupdict()
            moe_type = data['moe_type']
            num_experts = data['num_experts']
            model_name = data['model_name']
            
            if moe_type == "Conditional":
                model = ConditionalMoEModel(num_experts, model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, return_val=return_val)
            elif moe_type == "Matryoshka":
                message = "Currently in the implementation stage of MatryshkaMoEModel, please implement it first"
                stop_sys(message, raise_error=True)
            else:
                message = f"moe_type:{moe_type} does not have a implementation, please make one! - from model_name:{model_name}"
                stop_sys(message, raise_error=True)
        else:
            message = f"model_name:{model_name} contains 'MoEModel' but does not mathches the pattern:{pattern}, what was the model you were trying to ask for?"
            stop_sys(message, raise_error=True)
    else:
        # This is actually a feature_and_output_model with reuturn_func set to outputs
        model = FeatureAndOutputModel(model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, return_val=return_val)
    
    # Load Weights
    if weight_path!=None:
        loaded = torch.load(weight_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            model.load_state_dict(loaded)
        else:
            # Legacy format: full model object was saved instead of state_dict
            model.load_state_dict(loaded.state_dict())
    
    model = model.to(device)
    
    return model

def get_model_from_model_name(model_name, pretrained=True):
    # For Mobilenet models
    if "mobilenet" in model_name:
        # Load pretrained weights from MobileNet_v2
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
        elif model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=pretrained)
        else:
            message=f"No model for {model_name} ready for mobilenet"
            stop_sys(message, raise_error=True)
    
    # For efficientnet
    elif "efficientnet" in model_name:
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
        else:
            message=f"no model {model_name} ready for efficient"
            stop_sys(message, raise_error=True)
    
    # For Squeezenet
    elif "squeezenet" in model_name:
        if model_name=="squeezenet1_1":
            model = models.squeezenet1_1(pretrained=pretrained)
        else:
            message=f"No model for {model_name} ready for squeezenet"
            stop_sys(message, raise_error=True)
        
    # For NasNet
    elif "nasnet" in model_name:
        if model_name == "mnasnet0_5":
            model = models.mnasnet0_5(pretrained=pretrained)
        elif model_name == "mnasnet0_75":
            model = models.mnasnet0_75(pretrained=pretrained)
        elif model_name == "mnasnet1_0":
            model = models.mnasnet1_0(pretrained=pretrained)
        else:
            message=f"No model for {model_name} ready for nasnet"
            stop_sys(message, raise_error=True)
    
    # For Shufflenet models
    elif "shufflenet" in model_name:
        if model_name=="shufflenet_v2_x0_5":
            model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        elif model_name=="shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        else:
            message=f"No model for {model_name} ready for shufflenet"
            stop_sys(message, raise_error=True)
    
    # For resnet models
    elif "resnet" in model_name:
        # Load pretrained weights from ImageNet-1K
        if model_name == 'resnet18':
            # model = models.resnet18(pretrained=pretrained)
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            message=f"No model for {model_name} in torch vision models"
            stop_sys(message, raise_error=True)
            
    elif "resnext" in model_name:
        # Load pretrained weights from ImageNet-1K
        if model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None)
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None)
        elif model_name == 'resnext101_64x4d':
            model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT if pretrained else None)
        else:
            message=f"No model for {model_name} in torch vision models"
            stop_sys(message, raise_error=True)
            
    # For ViT models
    elif 'ViT' in model_name:
        # Basic models fromn torch
        if 'timm' in model_name:
            if model_name == 'ViT_Tiny_timm':
                model_proxy = "vit_tiny_patch16_224"
            elif model_name == 'ViT_Small_timm':
                model_proxy = "vit_small_patch16_224"
            elif model_name == 'ViT_Small_32_timm':
                model_proxy = "vit_small_patch32_224"
            elif model_name == 'ViT_Base_timm':
                model_proxy = "vit_base_patch16_224"
            elif model_name == 'ViT_Base_32_timm':
                model_proxy = "vit_base_patch32_224"
            elif model_name == 'ViT_Large_timm':
                model_proxy = "vit_large_patch16_224"
            elif model_name == 'ViT_Large_32_timm':
                model_proxy = "vit_large_patch32_224"
            else:
                message=f"no model {model_name} exists for timm model"
                stop_sys(message, raise_error=True)
            model = timm.create_model(model_proxy, pretrained=True)
            
        # # Large pretrained models from DINO, No training for these models (Use Registers as well)
        elif 'Dino' in model_name:
            # For Dino v2 models
            if model_name.endswith("v2"):
                if model_name == 'ViT_Small_Dino_v2':
                    model_proxy = 'dinov2_vits14_reg_lc'
                elif model_name == 'ViT_Base_Dino_v2':
                    model_proxy = 'dinov2_vitb14_reg_lc'
                elif model_name == 'ViT_Large_Dino_v2':
                    model_proxy = 'dinov2_vitl14_reg_lc'
                elif model_name == 'ViT_Gigantic_Dino_v2':
                    model_proxy = 'dinov2_vitg14_reg_lc'
                else:
                    message=f"no model {model_name} exists for Dino model"
                    stop_sys(message, raise_error=True)
                    
                # Set the Google DNS servers -> After ray connection who cares about local host!
                model = torch.hub.load('facebookresearch/dinov2', model_proxy)

            # For Dino v3 models
            elif model_name.endswith("v3"):
                if model_name == 'ViT_Small_Dino_v3':
                    model_weight_file = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
                    model_proxy = "dinov3_vits16"
                elif model_name == 'ViT_Small_Plus_Dino_v3':
                    model_weight_file = "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
                    model_proxy = "dinov3_vits16plus"
                elif model_name == 'ViT_Base_Dino_v3':
                    model_weight_file = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
                    model_proxy = "dinov3_vitb16"
                elif model_name == 'ViT_Large_Dino_v3':
                    model_weight_file = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
                    model_proxy = "dinov3_vitl16"
                elif model_name == 'ViT_Huge_Plus_Dino_v3':
                    model_weight_file = "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
                    model_proxy = "dinov3_vith16plus"
                elif model_name == 'ViT_7B_Dino_v3':
                    model_weight_file = "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
                    model_proxy = "dinov3_vit7b16"
                else:
                    message=f"no model {model_name} exists for Dino model"
                    stop_sys(message, raise_error=True)
                    
                # Set the Google DNS servers -> After ray connection who cares about local host!
                weight_path = os.path.join(DINOv3_MODEL_WEIGHT_PATH_BASE, model_weight_file)
                model = torch.hub.load(DINOv3_MODEL_LOCAL_REPO, model_proxy, source='local', weights=weight_path)
                
            # Undefined Dino models
            else:
                message = f"Due to the introduction of Dino v3 models, please use the names with version at the end"
                stop_sys(message, raise_error=True)
            
        
        # Large Pretrained models from EVA-02, No training for these models, only features are provided for these models
        elif "EVA02" in model_name:
            # Feature Extractor Only
            if model_name == "ViT_Tiny_EVA02":
                model_proxy = 'eva02_tiny_patch14_224.mim_in22k'
            elif model_name == "ViT_Small_EVA02":
                model_proxy = 'eva02_small_patch14_224.mim_in22k'
            elif model_name == "ViT_Base_EVA02":
                model_proxy = 'eva02_base_patch14_224.mim_in22k'
            elif model_name == "ViT_Large_EVA02":
                model_proxy = 'eva02_large_patch14_224.mim_in22k'
            
            # Feature Extractor + fc layers
            elif model_name == "ViT_Tiny_EVA02_336":
                model_proxy = 'eva02_tiny_patch14_336.mim_in22k_ft_in1k'
            elif model_name == "ViT_Small_EVA02_336":
                model_proxy = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
            elif model_name == "ViT_Base_EVA02_448":
                model_proxy = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'
            elif model_name == "ViT_Large_EVA02_448":
                model_proxy = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
            else:
                message=f"no model {model_name} exists for EVA02 model"
                stop_sys(message, raise_error=True)
            
            model = timm.create_model(model_proxy, pretrained=True)
        
        # Model finding failure
        else:
            message=f"{model_name} match not found"
            stop_sys(message, raise_error=True)
    
    # For Simple MLP models
    elif "SimpleMLP" in model_name:
        match = re.search(r"\[([0-9,]+)\]", model_name)

        if match:
            # Extract the first captured group, which is the number
            list_hidden_layers=list(map(int, match.group(1).split(',')))
            model = SimpleMLPModel(input_size=3*224*224, hidden_layers=list_hidden_layers, out_size=2, starting_layers=[])
        else:
            message="Specify the hidden_layers_layers based on the template!"
            stop_sys(message, raise_error=True)

    # For Simple CNN models
    elif "SimpleCNN" in model_name:
        # Template and actual string
        template = "SimpleCNN_num_conv_layers:{num_conv_layers}"

        # Convert the template to a regular expression to capture the number
        pattern = template.replace("{num_conv_layers}", r"(\d+)")

        # Use re.match to find the number
        match = re.match(pattern, model_name)

        if match:
            # Extract the first captured group, which is the number
            num_conv_layers = int(match.group(1))
            model = SimpleCNNModel(num_conv_layers=num_conv_layers, hidden_sizes=[], out_size=1)
        else:
            message="Specify the num_conv_layers based on the template!"
            stop_sys(message, raise_error=True)
    
    # For not implemented models
    else:
        message=f"No such model {model_name} is implemented, please implement it and run"
        stop_sys(message, raise_error=True)
    
    return model

# strip fc layer from model
def remove_classification_heads_get_feature_size(model_name, model):
    # Step - 2 Remove classification head
    
    """Start for mobilenet / efficientnet / nasnet squeezenet"""
    
    # initialize last layer for mobilenet
    if "mobilenet" in model_name or "efficientnet" in model_name or "nasnet" in model_name:
        # Remove last of classifier && Set hidden_layers so classification layer could be added in step - 4
        feature_size = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],
        )
        feature_extractor = model

    # initailize last layer for squeezenet
    elif "squeezenet" in model_name:
        message="Squeeze net with out hidden layer is currently not defined exiting sys"
        stop_sys(message, raise_error=True)
        
    # initialize last layer for resnet
    elif "resnet" in model_name or 'resnext' in model_name or "shufflenet" in model_name:
        feature_size = model.fc.in_features
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
            
    # initialize last layer for ViT
    elif 'ViT' in model_name:
        # Different ways of accessing fc layer for timm/Dino feature_extractors
        if 'timm' in model_name:
            # Remove head layer && Set hidden_layers to head.in_features so classification layer could be added in step - 4
            feature_size = model.head.in_features
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
        elif "Dino" in model_name:
            # For Dino v2 models
            if model_name.endswith("v2"):
                # Remove linear_head layer && Set hidden_layers to linear_head.in_features so classification layer could be added in step - 4
                feature_size = model.linear_head.in_features # As using input size from original output heads, processed (concated) feature_size is incoded righaway in the model
                feature_extractor = DinoV2FeatureExtractor(feature_extractor=model.backbone, layers=model.layers)
            # For Dino v3 models
            elif model_name.endswith("v3"):
                # As no linear head exists for Dino v3 models, use norm.normalized_shape[0]: the original feautre size to get the processed (concated) feature size, so classification layer could be added in step - 4
                feature_size = (model.norm.normalized_shape[0]) * 2 # Due to custom forwarding logic, uses twice the outputed features
                feature_extractor = DinoV3FeatureExtractor(feature_extractor=model)
            # Not Implemented Dino error handling
            else:
                message=f"Dino model handling for:{model_name} is not implemented, please implement it"
                stop_sys(message, raise_error=True)
        elif "EVA02" in model_name:
            # For EVA02 models, we get the feature extractor only from "get_model_from_model_name", so set is as the feature extractor right away
            feature_size = model.fc_norm.normalized_shape[0]
            
            # Remove head for None feature extractors only
            if hasattr(model, "head") and not isinstance(model.head, nn.Identity):
                model.head = nn.Identity()
            feature_extractor = model
            
        else:
            print(f"No method to initialize {model_name}")
    
    # initialize lasy layer for SimpleMLP, SimpleCNN
    elif "SimpleMLP" in model_name:
        feature_size = model.model[-1].in_features
        params_list = (list(model.model.children())[:-1])
        
        # The Simple MLP itself becomes the feature extractor itself
        model = nn.Sequential(*params_list)
        
    elif "SimpleCNN" in model_name:
        feature_size = model.model[-1].in_features
        params_list = (list(model.model.children())[:-1])
        
        # The Simple CNN itself becomes the feature extractor itself
        model = nn.Sequential(*params_list)
    
    # not implemented
    else:
        message=f'No method defined to initialize fc layer for {model_name}'
        stop_sys(message, raise_error=True)
    
    return feature_extractor, feature_size
    
