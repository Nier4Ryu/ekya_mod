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
        # set_mps_envvars(gpu_allocation_percentage)
        print("Initializing Start {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))
        NUM_CLASSES = self.hyperparameters["num_classes"]
        self.inference_scaling_function = inference_scaling_function    # The input range of the inference scaling function should be 0-1. Need to translate percentage by /100.
        self.name = name
        
        self.camera_idx=camera_idx
        self.log_dir=log_dir

        self.model = EkyaModelWrapper(NUM_CLASSES, hyperparameters=self.hyperparameters, restore_path=self.restore_path, device=device, camera_idx=self.camera_idx, log_dir=self.log_dir)
        
        print("Initializing Done! {}.\n Got ray.get_gpu_ids(): {}".format(name, ray.get_gpu_ids()))

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
        # print("Model Generation Start")
        self.model = get_default_model_from_params(model_name=model_name, pretrained=True, hidden_layers=hidden_layers, num_out=num_classes, last_layer_only=last_layer_only, weight_path=restore_path, device=self.device)
        # print("Model Generation Fin")
        
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

        if save_model:
            print("!!!!Training at least started!!!!")

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

# From here, implementation to custum models
class SimpleMLPModel(nn.Module):
    def __init__(self, input_size=3*224*224, hidden_layers:Union[int, List[int]]=None, out_size=2, starting_layers:list=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = []
        elif isinstance(hidden_layers, int):
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

class SimpleCNNBackbone(nn.Module):
    def __init__(self, num_conv_layers=1):
        super().__init__()
        
        layers = []
        input_dimension = 3
        out_dimension = 16
        # Convolutional Layers
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(input_dimension, out_dimension , kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2,2))
            input_dimension = out_dimension
            out_dimension = 2*out_dimension
        
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, tensors):
        outputs = self.backbone(tensors)
        print(outputs, "from backbone")
        return outputs

class SimpleCNNModel(nn.Module):
    def __init__(self, num_conv_layers=1, hidden_sizes:list=[512], out_size=2):
        super().__init__()
        
        # Define backbone
        backbone = SimpleCNNBackbone(num_conv_layers=num_conv_layers)
        
        # Convolutional Layers + connection infos
        layers = list(backbone.backbone.children())
        
        input_dimension = 3
        out_dimension = 16
        for _ in range(num_conv_layers):
            input_dimension = out_dimension
            out_dimension = 2*out_dimension
            
        # flattening
        layers.append(nn.Flatten())
            
        # Linear layers
        current_size = int(input_dimension*(224/(2**num_conv_layers))**2) # out_channel * (224:height/2^num_conv) * (224:width/2^num_conv)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, out_size))
        
        self.model=nn.Sequential(*layers)
        
        
    def forward(self, tensors):
        outputs = self.model(tensors)
        
        print(outputs, "from model")
        
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
        
# Specialized models
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
        # elif self.feature_extractor_name.startswith("ViT") and self.feature_extractor_name.endswith("timm"):
        #     self.forward_function = self.forward_with_extracting
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

    # This is suppose to be depricated, just leave it behind for backup
    # def forward_with_extracting(self, tensors):
    #     features = self.feature_extractor(tensors)
    #     extracted_features = features[:, 0]
    #     return extracted_features

class FullyConnectedLayers(nn.Module):
    def __init__(self, feature_extractor_name:str=None, input_size:int=None, hidden_layers:Union[int, List[int]]=None, out_size:int=None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = []
        elif isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        
        if feature_extractor_name!=None and input_size==None:
            message = f"This part was designed to get the input size for a fc layer without having to load a model when generating a gating network! As gating networks are currently deprecated, please re-implement the call of gating networks and this part after wards! (This is currently deprecated!!)"
            stop_sys(message, raise_error=True)
            # if feature_extractor_name == "ViT_Small_Dino_v2":
            #     input_size=1920
            # elif feature_extractor_name == "ViT_Small_timm":
            #     input_size=384
            # else:
            #     message=f"Extracting input_size from feature extractor for feature_extractor:{feature_extractor_name} is currently not implemented! exiting sys"
            #     stop_sys(message, raise_error=True)
            
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
        
        if hidden_layers is None:
            hidden_layers = []
        elif isinstance(hidden_layers, int):
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

# VMapModel Related
class VMapModelCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_info_list=None
        self.model_info_dict=None
        self.forward_function=None
        
        # Containers for model_params, model_buffers used for vmap
        self.runtime_model_params = None
        self.runtime_model_buffers = None
    
    def forward(self, *args, **kwargs):
        return self.forward_function(*args, **kwargs)
    
    
    def get_model_memory(self):
        def tree_bytes(tree) -> int:
            from torch.utils._pytree import tree_flatten
            leaves, _ = tree_flatten(tree)  # list of leaves
            return sum(t.numel() * t.element_size() for t in leaves if isinstance(t, torch.Tensor))
            
        # originals vs stacked (CPU or GPU — same formula)
        stacked_bytes = tree_bytes(self.model_params) + tree_bytes(self.model_buffers)
        stacked_MB = stacked_bytes/1e6
        print(f"stacked MB: {stacked_MB:.1f}")
        
        return stacked_MB

# VMapModel1 => Naive VMap version without any memory optimizations
class VMapModel1(VMapModelCore):
    def __init__(self, measure_timing=False):
        super().__init__()
        
        # Containers for timing info
        self.measure_timing = measure_timing
        self.end_to_end_records= []
        
        
    def initialize_vmap_model_from_params(self, model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, weight_paths_dict:dict=None, device='cpu', random_test=False):
        # self.forward_function = self.forward_with_data_and_model_mapping_info_by_list_return_as_list
        self.forward_function = self.forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor
        self.device=device

        # generate_model on "CPU" -> so weights do not get stacked on GPU for now
        self.model_info_dict={
            model_key:model_idx for model_idx, (model_key, weight_path) in enumerate(weight_paths_dict.items())
        }
        
        # This is the original code to run
        if not random_test:
            models = [
                get_default_model_from_params(
                    model_name=model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, weight_path=weight_path, device="cpu"
                ).eval() for model_key, weight_path in weight_paths_dict.items()
            ]
        # This is temp code for test purpose
        else:
            print("For testing purpose add a random val to change vals of each model!")
            
            models = []
            for model_key, weight_path in weight_paths_dict.items():
                model = get_default_model_from_params(
                    model_name=model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, weight_path=weight_path, device="cpu"
                ).eval()
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(0.01 * torch.randn_like(p))
                models.append(model)
        
        self.initialize_rest(models)
        
    def initialize_vmap_model_with_models_dict(self, models_dict:dict=None, device:str="cuda"):
        self.forward_function = self.forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor
        self.device=device
        
        # generate_model on "CPU" -> so weights do not get stacked on GPU for now
        self.model_info_dict={
            model_key:model_idx for model_idx, model_key in enumerate(models_dict.keys())
        }
        
        models = list(models_dict.values())
        self.initialize_rest(models)
    
    
    def initialize_rest(self, models):
        # --- Free previous GPU allocations ---
        if self.runtime_model_params!=None or self.runtime_model_buffers!=None:
            del self.runtime_model_params
            del self.runtime_model_buffers
            # Don not use empty_cache (Systems works do not allow GPU mem de-allocation overheads...)
            
        # Fill the model_params and model_buffers
        # print(f"Trying to move params to:{self.device}")
        # start_t = time.perf_counter()
        self.runtime_model_params, self.runtime_model_buffers = stack_module_state(models)              # leaves: (M, ...)
        # fin_t = time.perf_counter()
        # print(f"module stack took: {(fin_t-start_t)*1000}ms")
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()
        # start.record()
        self.runtime_model_params = tree_map(lambda t: t.to(self.device), self.runtime_model_params)
        self.runtime_model_buffers = tree_map(lambda t: t.to(self.device), self.runtime_model_buffers)
        # end.record()
        # torch.cuda.synchronize()
        # elapsed_ms = start.elapsed_time(end)
        # print(f"Param move time: {elapsed_ms:.3f} ms")
        
        # Define vmap forwarding for later useage
        base = copy.deepcopy(models[0]).to('meta').eval()
        def fmodel(p, b, x): 
            return functional_call(base, (p, b), (x,))
        
        self.forward_logic = vmap(fmodel, in_dims=(0,0,0)) # in_dims=(params, buffers, input) where, params.shape=(N, param_shape), buffers.shape=(N, buffer_shape), input.shape=(N, input_shape) -> out = forward_logic(p, b, x) -> out.shape = (N, output_shape)
    
    def forward_with_data_and_model_mapping_info_by_list_return_as_list(self, tensors: torch.Tensor, data_and_model_mapping_info_by_list):
        # run for: data_and_model_mapping_info_by_list = [(data_idx, model_key), ...] where K=len(data_and_model_mapping_info_by_list)
        
        # --- build index tensors (order preserved) ---
        data_indexes = torch.tensor([data_idx for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device) # (K,)
        model_indexes = torch.tensor([self.model_info_dict[model_key] for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device)  # (K,)

        # --- gather params/buffers for selected models (K, ...) ---
        flattened_model_params = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_model_params)
        flattened_model_buffers = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_model_buffers)

        # --- gather inputs (K, 3, 224, 224) ---
        flattened_tensors = tensors.index_select(0, data_indexes) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
        flattened_tensors = flattened_tensors.unsqueeze(1)        # (K, 1, C, H, W): Add batch dimension as model expects it

        # --- single vmap over pairs ---
        flattened_outputs = self.forward_logic(flattened_model_params, flattened_model_buffers, flattened_tensors) # (K, ...)
        
        # --- stitch back into a list with original mapping ---
        data_and_model_and_output_mapping_info_by_list = [
            (data_idx, model_key, flattened_outputs[idx]) for idx, (data_idx, model_key) in enumerate(data_and_model_mapping_info_by_list)
        ]
        return data_and_model_and_output_mapping_info_by_list
    
    def forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor(self, tensors: torch.Tensor, data_and_model_mapping_info_by_list):
        # run for: data_and_model_mapping_info_by_list = [(data_idx, model_key), ...] where K=len(data_and_model_mapping_info_by_list)
        if self.measure_timing:
            end_to_end_start = torch.cuda.Event(enable_timing=True)
            end_to_end_end   = torch.cuda.Event(enable_timing=True)
        
        if self.measure_timing:
            end_to_end_start.record()
        
        # --- build index tensors (order preserved) ---
        data_indexes = torch.tensor([data_idx for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device) # (K,)
        model_indexes = torch.tensor([self.model_info_dict[model_key] for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device)  # (K,)

        # --- gather params/buffers for selected models (K, ...) ---
        flattened_model_params = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_model_params)
        flattened_model_buffers = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_model_buffers)

        # --- gather inputs (K, 3, 224, 224) ---
        flattened_tensors = tensors.index_select(0, data_indexes) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
        flattened_tensors = flattened_tensors.unsqueeze(1)        # (K, 1, C, H, W): Add batch dimension as model expects it

        # --- single vmap over pairs ---
        
            
        flattened_outputs = self.forward_logic(flattened_model_params, flattened_model_buffers, flattened_tensors) # (K, ...)
        
        if self.measure_timing:
            end_to_end_end.record()
        
        if self.measure_timing:
            self.end_to_end_records.append((end_to_end_start, end_to_end_end))     
                
        return flattened_outputs

    def get_measure_timing_results(self):
        if not self.measure_timing:
            message=f"Error, you didn't measure any timing"
            stop_sys(message)
        else:
            # Make sure every operation is really finished
            torch.cuda.synchronize()
            
            end_to_end_times = [s.elapsed_time(e) for (s,e) in self.end_to_end_records] # ms
            
            warmup = len(end_to_end_times)//2

            def drop_warmup(x, n):
                return x[n:] if len(x) > n else []

            end_to_end_times_wo = drop_warmup(end_to_end_times, warmup)
            
            message = f"Have dropped {warmup} iterations for warmup"
            warn_user(message=message, warning_time=3)

            end_to_end_time_avg=sum(end_to_end_times_wo)/len(end_to_end_times_wo)
            print(f"AVG compute: {end_to_end_time_avg:.3f} ms")

            # optional: show per-chunk first few
            print("first 5 compute times:", end_to_end_times_wo[:5])
            return 0, 0, end_to_end_time_avg # Return loadtime=0, compute_time=0, end_to_end_time_avg
    
# VMapModel2 => Advancement of VMapModel1 to support less time overhead for stacked params & buffer movement
class VMapModel2(VMapModelCore):
    def __init__(self, measure_timing=False):
        super().__init__()
        
        # Containers for stacked model_params, model_buffers so as to save time for stacking
        self.stacked_model_params = None
        self.stacked_model_buffers = None
        
        # Containers for runtime model_params, model_buffers used for vmap
        self.runtime_stacked_model_params = None
        self.runtime_stacked_model_buffers = None
        
        # Container for mapping from 
        self.runtime_model_info_dict = None
        self.runtime_idx_converter_dict = None
        
        # Containers for timing info
        self.measure_timing = measure_timing
        self.load_records   = []
        self.compute_records= []
        self.end_to_end_records= []
        
    def initialize_vmap_model_with_models_dict(self, models_dict:dict, device:str="cuda"):
        self.forward_function = self.forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor
        self.device=device
        
        # generate_model on "CPU" -> so weights do not get stacked on GPU for now
        self.model_info_dict={
            model_key:model_idx for model_idx, model_key in enumerate(models_dict.keys())
        }
        
        models = list(models_dict.values())
        
        # Fill the stacked_model_params and stacked_model_buffers
        self.stacked_model_params, self.stacked_model_buffers = stack_module_state(models)              # leaves: (M, ...)
        
        # Define vmap forwarding for later useage
        base = copy.deepcopy(models[0]).to('meta').eval()
        def fmodel(p, b, x): 
            return functional_call(base, (p, b), (x,))
        
        self.forward_logic = vmap(fmodel, in_dims=(0,0,0)) # in_dims=(params, buffers, input) where, params.shape=(N, param_shape), buffers.shape=(N, buffer_shape), input.shape=(N, input_shape) -> out = forward_logic(p, b, x) -> out.shape = (N, output_shape)
        
    # Instead of calling initialize_vmap_model, the user can use update_runtime so as to reduce (params, buffer) stacking & moving
    def update_runtime_vmap_model_with_models_dict(self, model_keys:list):
        # check all the keys exist
        for key in model_keys:
            if key not in self.model_info_dict.keys():
                message = f"Given key:{key} does not exist in self.model_info_dict.keys():{self.model_info_dict.keys()}"
                stop_sys(message, raise_error=True)
            # Checking model itself is removed for now
        
        # key:offline_idx mapping
        self.runtime_model_info_dict = {
            key:self.model_info_dict[key]
            for key in model_keys
        }
        
        # offlin_idx:runtime_idx mapping
        self.runtime_idx_converter_dict = {
            offline_idx:runtime_idx
            for runtime_idx, offline_idx in enumerate(self.runtime_model_info_dict.values())
        }
        
        if self.measure_timing:
            load_start = torch.cuda.Event(enable_timing=True)
            load_end   = torch.cuda.Event(enable_timing=True)
        
        if self.measure_timing:
            load_start.record()
                
        # --- Free previous GPU allocations ---
        if self.runtime_stacked_model_params!=None or self.runtime_stacked_model_buffers!=None:
            del self.runtime_stacked_model_params
            self.runtime_stacked_model_params=None
            del self.runtime_stacked_model_buffers
            self.runtime_stacked_model_buffers=None
            # Empty cache should be never used in systems work! (only set position as free-for re-use)
            
        # Fill the runtime model_params and model_buffers
        
        indexes = torch.tensor(list(self.runtime_model_info_dict.values()), dtype=torch.long, device="cpu")
        
        self.runtime_stacked_model_params = tree_map(lambda t: t.index_select(0, indexes).to(self.device, non_blocking=True), self.stacked_model_params)
        self.runtime_stacked_model_buffers = tree_map(lambda t: t.index_select(0, indexes).to(self.device, non_blocking=True), self.stacked_model_buffers)
        
        if self.measure_timing:
            load_end.record()
            
        if self.measure_timing:
            self.load_records.append((load_start, load_end))
        
    def forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor(self, tensors: torch.Tensor, data_and_model_mapping_info_by_list):
        if self.measure_timing:
            end_to_end_start = torch.cuda.Event(enable_timing=True)
            end_to_end_end   = torch.cuda.Event(enable_timing=True)
        
        if self.measure_timing:
            end_to_end_start.record()
            
        # Update Remote Model Values
        self.update_runtime_vmap_model_with_models_dict(model_keys=[model_key for (data_idx, model_key) in data_and_model_mapping_info_by_list])
        
        # run for: data_and_model_mapping_info_by_list = [(data_idx, model_key), ...] where K=len(data_and_model_mapping_info_by_list)
        
        # --- build index tensors (order preserved) ---
        data_indexes = torch.tensor([data_idx for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device) # (K,)
        
        # This is previous part
        # model_indexes = torch.tensor([self.model_info_dict[model_key] for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device)  # (K,)
        
        model_indexes = torch.tensor([self.runtime_idx_converter_dict[self.model_info_dict[model_key]] for data_idx, model_key in data_and_model_mapping_info_by_list], dtype=torch.long, device=self.device)  # (K,)
        
        # --- gather params/buffers for selected models (K, ...) ---
        flattened_model_params = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_stacked_model_params)
        flattened_model_buffers = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_stacked_model_buffers)

        # --- gather inputs (K, 3, 224, 224) ---
        flattened_tensors = tensors.index_select(0, data_indexes) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
        flattened_tensors = flattened_tensors.unsqueeze(1)        # (K, 1, C, H, W): Add batch dimension as model expects it

        # --- single vmap over pairs ---
        if self.measure_timing:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end   = torch.cuda.Event(enable_timing=True)
        
        if self.measure_timing:
            compute_start.record()
            
        flattened_outputs = self.forward_logic(flattened_model_params, flattened_model_buffers, flattened_tensors) # (K, ...)
        
        if self.measure_timing:
            compute_end.record()
        
        if self.measure_timing:
            self.compute_records.append((compute_start, compute_end))
        
        if self.measure_timing:
            end_to_end_end.record()
        
        if self.measure_timing:
            self.end_to_end_records.append((end_to_end_start, end_to_end_end))     
            
        return flattened_outputs

    def get_measure_timing_results(self):
        if not self.measure_timing:
            message=f"Error, you didn't measure any timing"
            stop_sys(message)
        else:
            # Make sure every operation is really finished
            torch.cuda.synchronize()
            
            load_times    = [s.elapsed_time(e) for (s,e) in self.load_records]    # ms
            compute_times = [s.elapsed_time(e) for (s,e) in self.compute_records] # ms
            end_to_end_times = [s.elapsed_time(e) for (s,e) in self.end_to_end_records] # ms
            
            warmup = len(load_times)//2

            def drop_warmup(x, n):
                return x[n:] if len(x) > n else []

            load_times_wo    = drop_warmup(load_times, warmup)
            compute_times_wo = drop_warmup(compute_times, warmup)
            end_to_end_times_wo = drop_warmup(end_to_end_times, warmup)
            
            message = f"Have dropped {warmup} iterations for warmup"
            warn_user(message=message, warning_time=3)

            load_time_avg=sum(load_times_wo)/len(load_times_wo)
            compute_time_avg=sum(compute_times_wo)/len(compute_times_wo)
            end_to_end_time_avg=sum(end_to_end_times_wo)/len(end_to_end_times_wo)
            print(f"AVG load   : {load_time_avg:.3f} ms")
            print(f"AVG compute: {compute_time_avg:.3f} ms")
            print(f"AVG end_to_end: {end_to_end_time_avg:.3f} ms")

            # optional: show per-chunk first few
            print("\nfirst 5 load times   :", load_times_wo[:5])
            print("first 5 compute times:", compute_times_wo[:5])
            return load_time_avg, compute_time_avg, end_to_end_time_avg
        
# VMapModel3 => Advancement of VMapModel2 to support pipelined parameter swap to overlap compute / transfer, binary pipeline for now (easy install)
class VMapModel3(VMapModelCore):
    def __init__(self, num_per_stream=8, measure_timing=False):
        super().__init__()

        # Containers for stacked model_params, model_buffers so as to save time for stacking
        self.stacked_model_params = None
        self.stacked_model_buffers = None

        # stream infos
        self.num_streams=2
        self.num_per_stream=num_per_stream

        self.load_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()

        self.load_events_by_stream_dict = {
            stream_idx:torch.cuda.Event()
            for stream_idx in range(self.num_streams)
        }
        
        self.load_events_list = []

        # Containers for runtime model_params, model_buffers used for vmap
        self.runtime_stacked_model_params_by_stream_dict = {
            stream_idx:None
            for stream_idx in range(self.num_streams)
        }
        self.runtime_stacked_model_buffers_by_stream_dict = {
            stream_idx:None
            for stream_idx in range(self.num_streams)
        }

        # Container for mapping from 
        self.runtime_model_info_dict_by_stream_dict = {
            stream_idx:None
            for stream_idx in range(self.num_streams)
        }
        self.runtime_idx_converter_dict_by_stream_dict = {
            stream_idx:None
            for stream_idx in range(self.num_streams)
        }
        
        # Containers for timing info
        self.measure_timing = measure_timing
        self.load_records   = []
        self.compute_records= []
        self.end_to_end_records= []
        
    # 1) Initialization (Stack models on CPU, Allocate Buffer space on GPU)
    def initialize_vmap_model_with_models_dict(self, models_dict:dict, device:str="cuda"):
        self.forward_function = self.forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor
        self.device=device

        # generate_model on "CPU" -> so weights do not get stacked on GPU for now
        self.model_info_dict={
            model_key:model_idx for model_idx, model_key in enumerate(models_dict.keys())
        }

        models = list(models_dict.values())

        # Fill the stacked_model_params and stacked_model_buffers, pin them to memory for pipelining purposes
        stacked_model_params, stacked_model_buffers = stack_module_state(models)              # leaves: (M, ...)
        self.stacked_model_params  = tree_map(lambda t: t.pin_memory(), stacked_model_params)
        self.stacked_model_buffers = tree_map(lambda t: t.pin_memory(), stacked_model_buffers)

        # Generate Buffers for 
        def _alloc_leaf_like_src_firstdim_K(src_leaf, num_per_stream, device):
            if src_leaf is None:
                return None
            shape = (num_per_stream, *src_leaf.shape[1:])
            gpu_space = torch.empty(shape, device=device, dtype=src_leaf.dtype)
            return gpu_space

        self.runtime_stacked_model_params_by_stream_dict={
            stream_idx:tree_map(lambda t:_alloc_leaf_like_src_firstdim_K(t, self.num_per_stream, device), self.stacked_model_params)
            for stream_idx in range(self.num_streams)
        }
        self.runtime_stacked_model_buffers_by_stream_dict={
            stream_idx:tree_map(lambda t:_alloc_leaf_like_src_firstdim_K(t, self.num_per_stream, device), self.stacked_model_buffers)
            for stream_idx in range(self.num_streams)
        }

        # Define vmap forwarding for later useage
        base = copy.deepcopy(models[0]).to('meta').eval()
        def fmodel(p, b, x):
            return functional_call(base, (p, b), (x,))

        self.forward_logic = vmap(fmodel, in_dims=(0,0,0)) # in_dims=(params, buffers, input) where, params.shape=(N, param_shape), buffers.shape=(N, buffer_shape), input.shape=(N, input_shape) -> out = forward_logic(p, b, x) -> out.shape = (N, output_shape)

    # 2) Forward Function
    def forward_with_data_and_model_mapping_info_by_list_return_as_vmap_tensor(self, tensors: torch.Tensor, data_and_model_mapping_info_by_list):
        # Reset load_events to remove previous launches events
        self.load_events_list = []
        
        if self.measure_timing:
            end_to_end_start = torch.cuda.Event(enable_timing=True)
            end_to_end_end   = torch.cuda.Event(enable_timing=True)
        
        if self.measure_timing:
            end_to_end_start.record()
        
        data_and_model_mapping_info_by_list_chunks = [data_and_model_mapping_info_by_list[i:i+self.num_per_stream] for i in range(0, len(data_and_model_mapping_info_by_list), self.num_per_stream)]

        # Loop of forwarding in chunked manner (As compute is done for previous iteration, +1 times loop)
        num_chunks = len(data_and_model_mapping_info_by_list_chunks)
        current_data_and_model_mapping_info_by_list_chunk = None
        previous_data_and_model_mapping_info_by_list_chunk = None
        flattened_outputs_list = []

        for chunk_idx in range(num_chunks+1):
            # print(f"{chunk_idx}|{current_data_and_model_mapping_info_by_list_chunk}|{previous_data_and_model_mapping_info_by_list_chunk}")

            # Things to do for current chunk_idx
            if chunk_idx<num_chunks:
                # Save current chunk info (Use this chunk info for loading models onto GPU space)
                current_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx]

                # Load for current chunk -> Skip for chunk_idx==num_chunks
                stream_idx = chunk_idx % 2
                self.update_runtime_vmap_model_with_models_dict_for_stream(data_and_model_mapping_info_by_list_chunk=current_data_and_model_mapping_info_by_list_chunk, stream_idx=stream_idx, chunk_idx=chunk_idx)
                
            # Compute for previous chunk -> Skip for chunk_idx==0
            if chunk_idx>0 and previous_data_and_model_mapping_info_by_list_chunk!=None:
                stream_idx = (chunk_idx-1) % 2
                flattened_outputs_chunk = self.forward_with_data_and_model_mapping_info_by_list_chunk_return_as_vmap_tensor_for_stream(tensors=tensors, data_and_model_mapping_info_by_list_chunk=previous_data_and_model_mapping_info_by_list_chunk, stream_idx=stream_idx, previous_chunk_idx=chunk_idx-1)
                flattened_outputs_list.append(flattened_outputs_chunk)
            
            # Things to do for current chunk_idx
            if chunk_idx<num_chunks:                
                # Update previous chunk info (Use this chunk info for forwarding data in the next iteration)
                previous_data_and_model_mapping_info_by_list_chunk = current_data_and_model_mapping_info_by_list_chunk


            
        # Wait till the last compute stream finishes!
        self.compute_stream.synchronize()
        flattened_outputs_tensor = torch.cat(flattened_outputs_list, dim=0)
        
        if self.measure_timing:
            # torch.cuda.current_stream().wait_stream(self.compute_stream)
            with torch.cuda.stream(self.compute_stream):
                end_to_end_end.record()       # record AFTER ordering
            
        if self.measure_timing:
            self.end_to_end_records.append((end_to_end_start, end_to_end_end))     
            
        return flattened_outputs_tensor

    # 2-1) Load Pipeline
    def update_runtime_vmap_model_with_models_dict_for_stream(self, data_and_model_mapping_info_by_list_chunk, stream_idx:int, chunk_idx:int):
        # Fill the runtime model_params and model_buffers
        if self.measure_timing:
            load_start = torch.cuda.Event(enable_timing=True)
            load_end   = torch.cuda.Event(enable_timing=True)
        
        load_stream = self.load_stream
        with torch.cuda.stream(load_stream):
            if self.measure_timing:
                load_start.record()
        
            # Loading logic for chunk
            model_keys = [model_key for (data_idx, model_key) in data_and_model_mapping_info_by_list_chunk]

            # check all the keys exist
            for key in model_keys:
                if key not in self.model_info_dict.keys():
                    message = f"Given key:{key} does not exist in self.model_info_dict.keys():{self.model_info_dict.keys()}"
                    stop_sys(message, raise_error=True)
                # Checking model itself is removed for now

            # key:offline_idx mapping
            self.runtime_model_info_dict_by_stream_dict[stream_idx] = {
                key:self.model_info_dict[key]
                for key in model_keys
            }

            # offlin_idx:runtime_idx mapping
            self.runtime_idx_converter_dict_by_stream_dict[stream_idx] = {
                offline_idx:runtime_idx
                for runtime_idx, offline_idx in enumerate(self.runtime_model_info_dict_by_stream_dict[stream_idx].values())
            }
        
            indexes = torch.tensor(list(self.runtime_model_info_dict_by_stream_dict[stream_idx].values()), dtype=torch.long, device="cpu")
            
            tree_map(lambda dst, src: self._copy(indexes, dst, src), self.runtime_stacked_model_params_by_stream_dict[stream_idx], self.stacked_model_params)
            tree_map(lambda dst, src: self._copy(indexes, dst, src), self.runtime_stacked_model_buffers_by_stream_dict[stream_idx], self.stacked_model_buffers)
            
            if self.measure_timing:
                load_end.record()

            # Memo the fin of loading event
            # self.load_events_by_stream_dict[stream_idx].record(load_stream)
            self.load_events_list.append(torch.cuda.Event(blocking=False, enable_timing=False))
            self.load_events_list[chunk_idx].record(load_stream)

        if self.measure_timing:
            self.load_records.append((load_start, load_end))

    def _copy(self, indexes, dst_tensor, src_tensor):
        if dst_tensor is None or src_tensor is None:
            return dst_tensor

        # 1) select the rows on CPU (no GPU allocation here)
        selected = src_tensor.index_select(0, indexes)
        selected = selected.pin_memory()

        # 2) copy into pre-allocated GPU buffer
        n = selected.shape[0]
        dst_tensor[:n].copy_(selected, non_blocking=True)
        return dst_tensor
    
    # 2-2) Forward Pipeline
    def forward_with_data_and_model_mapping_info_by_list_chunk_return_as_vmap_tensor_for_stream(self, tensors: torch.Tensor, data_and_model_mapping_info_by_list_chunk:list, stream_idx:int, previous_chunk_idx:int):
        # Forward logic for chunk
        # --- single vmap over pairs ---
        if self.measure_timing:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end   = torch.cuda.Event(enable_timing=True)
        
        # load_event = self.load_events_by_stream_dict[stream_idx]
        load_event = self.load_events_list[previous_chunk_idx]
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(load_event)

            # --- build index tensors (order preserved) ---
            data_indexes = torch.tensor([data_idx for data_idx, model_key in data_and_model_mapping_info_by_list_chunk], dtype=torch.long, device=self.device) # (K,)
            model_indexes = torch.tensor([self.runtime_idx_converter_dict_by_stream_dict[stream_idx][self.model_info_dict[model_key]] for data_idx, model_key in data_and_model_mapping_info_by_list_chunk], dtype=torch.long, device=self.device)  # (K,)

            # --- gather params/buffers for selected models (K, ...) ---
            flattened_model_params = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_stacked_model_params_by_stream_dict[stream_idx])
            flattened_model_buffers = tree_map(lambda t: t.index_select(0, model_indexes), self.runtime_stacked_model_buffers_by_stream_dict[stream_idx])

            # --- gather inputs (K, 3, 224, 224) ---
            flattened_tensors = tensors.index_select(0, data_indexes) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
            flattened_tensors = flattened_tensors.unsqueeze(1)        # (K, 1, C, H, W): Add batch dimension as model expects it

            if self.measure_timing:
                compute_start.record()
            
            flattened_outputs_chunk = self.forward_logic(flattened_model_params, flattened_model_buffers, flattened_tensors) # (K, ...)
            
            if self.measure_timing:
                compute_end.record()
        
        if self.measure_timing:
            self.compute_records.append((compute_start, compute_end))

        return flattened_outputs_chunk

    
    def get_measure_timing_results(self):
        if not self.measure_timing:
            message=f"Error, you didn't measure any timing"
            stop_sys(message)
        else:
            # Make sure every operation is really finished
            torch.cuda.synchronize()
            
            load_times    = [s.elapsed_time(e) for (s,e) in self.load_records]    # ms
            compute_times = [s.elapsed_time(e) for (s,e) in self.compute_records] # ms
            end_to_end_times = [s.elapsed_time(e) for (s,e) in self.end_to_end_records] # ms
            
            warmup = len(load_times)//2
            warmup_end_to_end = len(end_to_end_times)//2
            
            def drop_warmup(x, n):
                return x[n:] if len(x) > n else []

            load_times_wo    = drop_warmup(load_times, warmup)
            compute_times_wo = drop_warmup(compute_times, warmup)
            end_to_end_times_wo = drop_warmup(end_to_end_times, warmup_end_to_end)
            
            message = f"Have dropped {warmup} iterations for warmup"
            warn_user(message=message, warning_time=3)

            load_time_avg=sum(load_times_wo)/len(load_times_wo)
            compute_time_avg=sum(compute_times_wo)/len(compute_times_wo)
            end_to_end_time_avg=sum(end_to_end_times_wo)/len(end_to_end_times_wo)
            print(f"AVG load   : {load_time_avg:.3f} ms")
            print(f"AVG compute: {compute_time_avg:.3f} ms")
            print(f"AVG end2end: {end_to_end_time_avg:.3f} ms")

            # optional: show per-chunk first few
            print("\nfirst 5 load times   :", load_times_wo[:5])
            print("first 5 compute times:", compute_times_wo[:5])
            return load_time_avg, compute_time_avg, end_to_end_time_avg

# VMapModel4 => Version That Overlaps compute & Tranfertime During Runtime
class VMapModel4(VMapModelCore):
    def __init__(self, num_load_streams=2, num_per_stream=8, measure_timing=False):
        super().__init__()
        
        # Pipelining related
        # 1) Default numbers setting
        self.num_load_streams = num_load_streams
        self.num_per_stream = num_per_stream
        
        # 2) Streams
        self.load_streams_dict = {
            stream_idx:torch.cuda.Stream()
            for stream_idx in range(self.num_load_streams)
        }
        self.compute_stream = torch.cuda.Stream()
        
        # 3) Events list
        self.load_events = None
        
        # Space Related
        # 1) Space for flattend models on CPU (pinned)
        self.flattened_models_on_cpu = None
        
        # 2) Space for mem copy by chunks
        self.pinned_gpu_mem_dict = None
        
        # 3) Space for params, buffers
        self.stacked_model_params_gpu = None
        self.stacked_model_buffers_gpu = None
        
        # 4) Space for data_indexes during runtime
        self.data_indexes_space = None
        
        # Time measure related
        self.measure_timing = measure_timing
        self.load_records = []
        self.compute_records = []
        self.end_to_end_times = []
        
    def initialize(self, models_dict:dict, device:str="cuda"):
        # Default values store
        self.model_info_dict = {
            model_key:model_idx
            for model_idx, model_key in enumerate(models_dict.keys())
        }
        self.device = device
        self.forward_function = self.custom_forward
        
        
        # Reserve space for future usages
        # 1) Space for flattend models on CPU (pinned)
        models = list(models_dict.values())
        num_models = len(models)
        stacked_model_params, stacked_model_buffers = stack_module_state(models)
        
        flat_list = []
        self.shape_table = {} #  Shape table is used for disassmble of models from GPU
        self.offset_table = {}# Offset table is used for disassmble of models from GPU
        offset=0
        for name, tensor in stacked_model_params.items():
            self.shape_table[name] = tensor.shape[1:]       # Save shape of ?
            num = tensor[0].numel()                         # number of elements for 1 model
            self.offset_table[name] = (offset, num)         # Where this block will live in the flattened row
            flat_list.append(tensor.reshape(num_models, -1))# flatten from (num_models, *shape) -> (num_models, num)
            offset += num
            
        for name, tensor in stacked_model_buffers.items():
            self.shape_table[name] = tensor.shape[1:]       
            num = tensor[0].numel()                         
            self.offset_table[name] = (offset, num)         
            flat_list.append(tensor.reshape(num_models, -1))
            offset += num
    
        self.flattened_models_on_cpu = torch.cat(flat_list, dim=1).contiguous().pin_memory()
        self.num_params_per_model = self.flattened_models_on_cpu.shape[1] # -> Use this val for future model copy
        
        # 2) Space for mem copy by chunks
        self.pinned_gpu_mem_dict = {
            stream_idx:torch.empty((self.num_per_stream, self.num_params_per_model), dtype=self.flattened_models_on_cpu.dtype, device=self.device)
            for stream_idx in range(self.num_load_streams)
        }
        
        # 3) Space for params, buffers (Initialy Empty)
        self.stacked_model_params_gpu = {
            name: torch.empty((self.num_per_stream, *tensor.shape[1:]), device=self.device)
            for name, tensor in stacked_model_params.items()
        }
        self.stacked_model_buffers_gpu = {
            name: torch.empty((self.num_per_stream, *tensor.shape[1:]), device=self.device)
            for name, tensor in stacked_model_buffers.items()
        }
        
        # Define vmap forwarding
        meta_model = copy.deepcopy(models[0]).to('meta').eval()
        def fmodel(p, b, x):
            return functional_call(meta_model, (p, b), (x,))

        self.forward_logic = vmap(fmodel, in_dims=(0,0,0)) # in_dims=(params, buffers, input) where, params.shape=(N, param_shape), buffers.shape=(N, buffer_shape), input.shape=(N, input_shape) -> out = forward_logic(p, b, x) -> out.shape = (N, output_shape)
        
        # Pre-allocate space for data_indexes for forwarding
        self.data_indexes_space = torch.tensor([0 for _ in range(self.num_per_stream)], dtype=torch.long, device=self.device) # (K,)
        
    def initialize_vmap_model_from_params(self, model_name, hidden_layers, num_out, last_layer_only, weight_paths_dict, device):
        models_dict = {
            model_key:get_default_model_from_params(
                model_name=model_name, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, weight_path=vmap_model_ckpt_path, device="cpu"
            )
            for model_key, vmap_model_ckpt_path in weight_paths_dict.items()
        }
        self.initialize(models_dict=models_dict, device=device)
        
    
    
    def custom_forward(self, tensors, data_and_model_mapping_info_by_list):
        # Need to perform multipple iterations for long data_and_model_mapping
        if self.measure_timing:
            start_time = time.perf_counter()
        
        # Forward Logic
        # 1) Split data_and_model_mapping_info_by_list into chunks
        data_and_model_mapping_info_by_list_chunks = [data_and_model_mapping_info_by_list[i:i+self.num_per_stream] for i in range(0, len(data_and_model_mapping_info_by_list), self.num_per_stream)]
        num_chunks=len(data_and_model_mapping_info_by_list_chunks)
        
        # 2) Remove previous events + Generate outputs container
        self.load_events_list = [None] * num_chunks
        outputs = []
        
        # 3) Forward for split chunks
        for chunk_idx in range(num_chunks+1):
            # Load Logic (Load chunk k)
            if chunk_idx<num_chunks:
                current_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx]
                self.update_for_chunks(current_chunk_idx=chunk_idx, current_data_and_model_mapping_info_by_list_chunk=current_data_and_model_mapping_info_by_list_chunk)
                
            # Forward Logic (Compute chunk k-1)
            if chunk_idx>0:
                prev_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx-1]
                outputs_chunk = self.forward_for_block(previous_chunk_idx=chunk_idx-1, tensors=tensors, prev_data_and_model_mapping_info_by_list_chunk=prev_data_and_model_mapping_info_by_list_chunk)
                outputs.append(outputs_chunk)
        
        # 4) Synch computation for this batch
        self.compute_stream.synchronize()
        outputs = torch.cat(outputs)
        
        if self.measure_timing:
            end_time = time.perf_counter()
            self.end_to_end_times.append((start_time, end_time))
        
        return outputs
        
        
    def update_for_chunks(self, current_chunk_idx, current_data_and_model_mapping_info_by_list_chunk):
        # CPU required operations
        stream_idx = current_chunk_idx % self.num_load_streams
        model_keys = list([model_key for data_idx, model_key in current_data_and_model_mapping_info_by_list_chunk])

        # Measure event starts
        if self.measure_timing:
            load_start = torch.cuda.Event(enable_timing=True)
            load_end   = torch.cuda.Event(enable_timing=True)
            load_start.record(self.load_streams_dict[stream_idx])
        
        # Copy to fixed GPU space by chunks
        load_event = torch.cuda.Event()
        with torch.cuda.stream(self.load_streams_dict[stream_idx]):
            for model_index, model_key in enumerate(model_keys):
                self.pinned_gpu_mem_dict[stream_idx][model_index:(model_index+1)].copy_(self.flattened_models_on_cpu[self.model_info_dict[model_key]], non_blocking=True)
            load_event.record(self.load_streams_dict[stream_idx])
        
        # Memo the time when load finishes -> This is used to check loading finish from compute_stream
        self.load_events_list[current_chunk_idx] = load_event
        
        # Measure event fin
        if self.measure_timing:
            load_end.record(self.load_streams_dict[stream_idx])
            self.load_records.append((load_start, load_end))
    
    def forward_for_block(self, previous_chunk_idx, tensors, prev_data_and_model_mapping_info_by_list_chunk):
        # CPU required operations
        stream_idx = previous_chunk_idx % self.num_load_streams
        num_model_keys=len(prev_data_and_model_mapping_info_by_list_chunk)
        data_indexes = torch.tensor([data_idx for data_idx, model_key in prev_data_and_model_mapping_info_by_list_chunk], dtype=torch.long)
        
        # Forward logic on GPU
        # 0) Wait for load_stream to finish
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(self.load_events_list[previous_chunk_idx])
            
            # Measure event start
            if self.measure_timing:
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end   = torch.cuda.Event(enable_timing=True)
                compute_start.record(self.compute_stream)
                
            # 1) Disassemble loaded weights from GPU chunk space into params, buffers space
            # This is done here as we have 2 GPU chunk spaces but only 1 params+buffers space and as this processes overhead is small
            # 1-1) Params
            for name in self.stacked_model_params_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_params_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)

            params_slice = {
                name: p[:num_model_keys]
                for name, p in self.stacked_model_params_gpu.items()
            }
            

            # 1-2) Buffers
            for name in self.stacked_model_buffers_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_buffers_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)
                
            buffers_slice = {
                name: b[:num_model_keys]
                for name, b in self.stacked_model_buffers_gpu.items()
            }

            # 2) Copy data indexes to GPU + Generate flattened tensors
            # self.data_indexes_space[:num_model_keys].copy_(data_indexes) # Shouldn't it be len(data_indexes)? => below / why is it num_model_keys?)
            self.data_indexes_space[:len(data_indexes)].copy_(data_indexes)
            # flattened_tensors = tensors.index_select(0, self.data_indexes_space[:num_model_keys]) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding 
            flattened_tensors = tensors.index_select(0, self.data_indexes_space[:len(data_indexes)]) # Shouldn't it be len(data_indexes)? => current / why is it num_model_keys? (up) check and remove above if this is correct
            flattened_tensors = flattened_tensors.unsqueeze(1) # (K, 1, C, H, W): Add batch dimension as model expects it
            
            # 3) Perform forwarding
            # flattened_outputs = self.forward_logic(self.stacked_model_params_gpu, self.stacked_model_buffers_gpu, flattened_tensors) # (K, ...)
            flattened_outputs = self.forward_logic(params_slice, buffers_slice, flattened_tensors) # (K, ...)
            
        # Measure event fin
        if self.measure_timing:
            compute_end.record(self.compute_stream)
            self.compute_records.append((compute_start, compute_end))
        
        return flattened_outputs
    
    def get_measure_timing_results(self, batch_size=None):
        if not self.measure_timing:
            message=f"Error, you didn't measure any timing"
            stop_sys(message)
        else:
            def drop_warmup(x, n):
                if len(x) <= n:
                    message = f"Cannot drop {n} samples when I only have {len(x)} samples in the list, stopping sys"
                    stop_sys(message)
                else:
                    dropped_x = x[n:] 
                    return dropped_x
            
            def avg_of_list(x):
                if len(x) < 1:
                    message = f"We cannot calculate the avg for:{x}"
                    stop_sys(message, raise_error=True)
                else:
                    avg = sum(x)/len(x)
                    return avg
                
            # Make sure every operation is really finished
            load_times    = [s.elapsed_time(e) for (s,e) in self.load_records]    # ms
            load_times_len = len(load_times)
            compute_times = [s.elapsed_time(e) for (s,e) in self.compute_records] # ms
            compute_times_len = len(compute_times)
            end_to_end_times = [(e-s)*1000 for (s,e) in self.end_to_end_times] # ms
            end_to_end_times_len = len(end_to_end_times)
            
            load_times_warmup = drop_warmup(load_times, load_times_len//2)
            compute_times_warmup = drop_warmup(compute_times, compute_times_len//2)
            end_to_end_times_warmup = drop_warmup(end_to_end_times, end_to_end_times_len//2)
            
            message = f"Have dropped {end_to_end_times_warmup} batches for warmup"
            warn_user(message=message, warning_time=0)
            
            load_times_avg = avg_of_list(load_times_warmup)
            compute_times_avg = avg_of_list(compute_times_warmup)
            end_to_end_times_avg = avg_of_list(end_to_end_times_warmup)
            
            print(f"AVG load   : {load_times_avg:.3f} ms")
            print(f"AVG compute: {compute_times_avg:.3f} ms")
            print(f"AVG end2end: {end_to_end_times_avg:.3f} ms")
            
            # Measure overlap of compute and load
            overlaps = []
            if batch_size!=None and batch_size>self.num_per_stream:
                num_stages = (batch_size + self.num_per_stream - 1) // self.num_per_stream
                print("num_stages", num_stages)
                
                def group_by_stages(records, num_stages):
                    return [records[i:i+num_stages] for i in range(0, len(records), num_stages)]
                load_groups = group_by_stages(self.load_records, num_stages)
                load_groups_wo = drop_warmup(load_groups, len(load_groups)//2)
                compute_groups = group_by_stages(self.compute_records, num_stages)
                compute_groups_wo = drop_warmup(compute_groups, len(compute_groups)//2)
                
                for load_group, compute_group in zip(load_groups_wo, compute_groups_wo):
                    for stage_idx in range(num_stages-1):
                        load_start, load_end = load_group[stage_idx+1]
                        compute_start, compute_end = compute_group[stage_idx]
                        
                        overlap = load_start.elapsed_time(compute_end)
                        overlap = max(0.0, overlap)
                    
                        overlaps.append(overlap)
                        
                        overlaps_avg=sum(overlaps)/len(overlaps)
                print(f"Overlap:{overlaps_avg}")
            else:
                print(f"For batch_size:{batch_size}, num_per_stream:{self.num_per_stream} overlap is impossible")

# VMapModel5 => Version That Overlaps and uses scheduling to batch same main model outputs
class VMapModel5(VMapModelCore):
    def __init__(self, num_load_streams=2, num_per_stream=8, main_model_batch_size=None, measure_timing=False):
        super().__init__()
        
        # Pipelining related
        # 1) Default numbers setting
        self.num_load_streams = num_load_streams
        self.num_per_stream = num_per_stream
        
        # 2) Streams
        self.load_streams_dict = {
            stream_idx:torch.cuda.Stream()
            for stream_idx in range(self.num_load_streams)
        }
        self.compute_stream = torch.cuda.Stream()
        
        # 3) Events list
        self.load_events = None
        
        # Space Related
        # 1) Space for flattend models on CPU (pinned)
        self.flattened_models_on_cpu = None
        
        # 2) Space for mem copy by chunks
        self.pinned_gpu_mem_dict = None
        
        # 3) Space for params, buffers
        self.stacked_model_params_gpu = None
        self.stacked_model_buffers_gpu = None
        
        # 4) Space for data_indexes during runtime
        self.data_indexes_space = None
        
        # Time measure related
        self.measure_timing = measure_timing
        self.load_records = []
        self.compute_records = []
        self.end_to_end_times = []
        
        # 5) Scheduler
        self.main_model_batch_size = main_model_batch_size
        self.batch_scheduler = NaiveBatchScheduler1(num_per_stream=self.num_per_stream)
        
    def initialize(self, models_dict:dict, device:str="cuda"):
        # Default values store
        self.model_info_dict = {
            model_key:model_idx
            for model_idx, model_key in enumerate(models_dict.keys())
        }
        self.device = device
        self.forward_function = self.custom_forward_with_scheduler
        
        # Reserve space for future usages
        # 1) Space for flattend models on CPU (pinned)
        models = list(models_dict.values())
        num_models = len(models)
        stacked_model_params, stacked_model_buffers = stack_module_state(models)
        
        flat_list = []
        self.shape_table = {} #  Shape table is used for disassmble of models from GPU
        self.offset_table = {}# Offset table is used for disassmble of models from GPU
        offset=0
        for name, tensor in stacked_model_params.items():
            self.shape_table[name] = tensor.shape[1:]       # Save shape of ?
            num = tensor[0].numel()                         # number of elements for 1 model
            self.offset_table[name] = (offset, num)         # Where this block will live in the flattened row
            flat_list.append(tensor.reshape(num_models, -1))# flatten from (num_models, *shape) -> (num_models, num)
            offset += num
            
        for name, tensor in stacked_model_buffers.items():
            self.shape_table[name] = tensor.shape[1:]       
            num = tensor[0].numel()                         
            self.offset_table[name] = (offset, num)         
            flat_list.append(tensor.reshape(num_models, -1))
            offset += num
    
        self.flattened_models_on_cpu = torch.cat(flat_list, dim=1).contiguous().pin_memory()
        self.num_params_per_model = self.flattened_models_on_cpu.shape[1] # -> Use this val for future model copy
        
        # 2) Space for mem copy by chunks
        self.pinned_gpu_mem_dict = {
            stream_idx:torch.empty((self.num_per_stream, self.num_params_per_model), dtype=self.flattened_models_on_cpu.dtype, device=self.device)
            for stream_idx in range(self.num_load_streams)
        }
        
        # 3) Space for params, buffers (Initialy Empty)
        self.stacked_model_params_gpu = {
            name: torch.empty((self.num_per_stream, *tensor.shape[1:]), device=self.device)
            for name, tensor in stacked_model_params.items()
        }
        self.stacked_model_buffers_gpu = {
            name: torch.empty((self.num_per_stream, *tensor.shape[1:]), device=self.device)
            for name, tensor in stacked_model_buffers.items()
        }
        
        # Define vmap forwarding
        meta_model = copy.deepcopy(models[0]).to('meta').eval()
        def fmodel(p, b, x):
            return functional_call(meta_model, (p, b), (x,))

        self.forward_logic = vmap(fmodel, in_dims=(0,0,0)) # in_dims=(params, buffers, input) where, params.shape=(N, param_shape), buffers.shape=(N, buffer_shape), input.shape=(N, input_shape) -> out = forward_logic(p, b, x) -> out.shape = (N, output_shape)
        
        # Pre-allocate space for data_indexes for forwarding
        self.data_indexes_space = torch.tensor([0 for _ in range(self.main_model_batch_size)], dtype=torch.long, device=self.device) # (K,)
        
        
    def initialize_vmap_model_from_params(self, model_name, hidden_layers, num_out, last_layer_only, weight_paths_dict, device):
        models_dict = {
            model_key:get_default_model_from_params(
                model_name=model_name, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, weight_path=vmap_model_ckpt_path, device="cpu"
            )
            for model_key, vmap_model_ckpt_path in weight_paths_dict.items()
        }
        self.initialize(models_dict=models_dict, device=device)
        
    def custom_forward(self, tensors, data_and_model_mapping_info_by_list):
        # Need to perform multipple iterations for long data_and_model_mapping
        if self.measure_timing:
            start_time = time.perf_counter()
        
        # Forward Logic
        # 1) Split data_and_model_mapping_info_by_list into chunks
        data_and_model_mapping_info_by_list_chunks = [data_and_model_mapping_info_by_list[i:i+self.num_per_stream] for i in range(0, len(data_and_model_mapping_info_by_list), self.num_per_stream)]
        num_chunks=len(data_and_model_mapping_info_by_list_chunks)
        
        # 2) Remove previous events + Generate outputs container
        self.load_events_list = [None] * num_chunks
        outputs = []
        
        # 3) Forward for split chunks
        for chunk_idx in range(num_chunks+1):
            # Load Logic (Load chunk k)
            if chunk_idx<num_chunks:
                current_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx]
                self.update_for_chunks(current_chunk_idx=chunk_idx, current_data_and_model_mapping_info_by_list_chunk=current_data_and_model_mapping_info_by_list_chunk)
                
            # Forward Logic (Compute chunk k-1)
            if chunk_idx>0:
                prev_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx-1]
                outputs_chunk = self.forward_for_block(previous_chunk_idx=chunk_idx-1, tensors=tensors, prev_data_and_model_mapping_info_by_list_chunk=prev_data_and_model_mapping_info_by_list_chunk)
                outputs.append(outputs_chunk)
        
        # 4) Synch computation for this batch
        self.compute_stream.synchronize()
        outputs = torch.cat(outputs)
        
        if self.measure_timing:
            end_time = time.perf_counter()
            self.end_to_end_times.append((start_time, end_time))
        
        return outputs
        
    def custom_forward_with_scheduler(self, tensors, data_and_model_mapping_info_by_list, main_model_outputs):
        # Need to perform multipple iterations for long data_and_model_mapping
        if self.measure_timing:
            start_time = time.perf_counter()
        
        # Forward Logic
        # 1) Split data_and_model_mapping_info_by_list into chunks
        data_idxes_and_model_mapping_infos_list = self.batch_scheduler.generate_batch_schedule(data_and_model_mapping_info_by_list, main_model_outputs)
        num_chunks=len(data_idxes_and_model_mapping_infos_list)
        
        # print("inspect here!")
        # print(data_idxes_and_model_mapping_infos_list)
        # print(num_chunks)
        # stop_sys()
        
        # 2) Remove previous events + Generate outputs container
        self.load_events_list = [None] * num_chunks
        outputs = []
        
        # 3) Forward for split chunks
        for chunk_idx in range(num_chunks+1):
            # Load Logic (Load chunk k)
            if chunk_idx<num_chunks:
                # current_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx]
                current_data_idxes_and_model_mapping_infos_chunk = data_idxes_and_model_mapping_infos_list[chunk_idx]
                # self.update_for_chunks(current_chunk_idx=chunk_idx, current_data_and_model_mapping_info_by_list_chunk=current_data_and_model_mapping_info_by_list_chunk)
                self.update_for_chunks2(current_chunk_idx=chunk_idx, current_data_idxes_and_model_mapping_infos_chunk=current_data_idxes_and_model_mapping_infos_chunk)
                
            # Forward Logic (Compute chunk k-1)
            if chunk_idx>0:
                # prev_data_and_model_mapping_info_by_list_chunk = data_and_model_mapping_info_by_list_chunks[chunk_idx-1]
                prev_data_idxes_and_model_mapping_infos_chunk = data_idxes_and_model_mapping_infos_list[chunk_idx-1]
                # outputs_chunk = self.forward_for_block(previous_chunk_idx=chunk_idx-1, tensors=tensors, prev_data_and_model_mapping_info_by_list_chunk=prev_data_and_model_mapping_info_by_list_chunk)
                outputs_chunk, results_indexing_dict = self.forward_for_block2(previous_chunk_idx=chunk_idx-1, tensors=tensors, prev_data_idxes_and_model_mapping_infos_chunk=prev_data_idxes_and_model_mapping_infos_chunk)
                outputs.append((outputs_chunk, results_indexing_dict))
        
        # 4) Synch computation for this batch
        self.compute_stream.synchronize()

        # outputs = torch.cat(outputs) # -> This part should change to re-create the original structure
        temp_outputs = {}
        for outputs_chunk_idx, (outputs_chunk, results_indexing_dict) in enumerate(outputs):
            for result_indexing_key, results_indexing_value in results_indexing_dict.items():
                temp_outputs[result_indexing_key] = (outputs_chunk_idx, results_indexing_value)

        outputs_by_single_tensors = []
        for data_idx_and_model_key in data_and_model_mapping_info_by_list:
            outputs_chunk_idx, results_indexing_value = temp_outputs[data_idx_and_model_key]
            outputs_chunk, _ = outputs[outputs_chunk_idx]
            outputs_single_tensor = outputs_chunk[results_indexing_value]
            outputs_by_single_tensors.append(outputs_single_tensor)
        
        # Convert list of tensors to single stacked tensor of (vmap_dimension, 1, outputs): same as without schedule
        outputs = torch.stack(outputs_by_single_tensors, dim=0)
        outputs = outputs.unsqueeze(1)
        
        if self.measure_timing:
            end_time = time.perf_counter()
            self.end_to_end_times.append((start_time, end_time))
            
        return outputs
      
    def update_for_chunks(self, current_chunk_idx, current_data_and_model_mapping_info_by_list_chunk):
        # CPU required operations
        stream_idx = current_chunk_idx % self.num_load_streams
        model_keys = list([model_key for data_idx, model_key in current_data_and_model_mapping_info_by_list_chunk])

        # Measure event starts
        if self.measure_timing:
            load_start = torch.cuda.Event(enable_timing=True)
            load_end   = torch.cuda.Event(enable_timing=True)
            load_start.record(self.load_streams_dict[stream_idx])
        
        # Copy to fixed GPU space by chunks
        load_event = torch.cuda.Event()
        with torch.cuda.stream(self.load_streams_dict[stream_idx]):
            for model_index, model_key in enumerate(model_keys):
                self.pinned_gpu_mem_dict[stream_idx][model_index:(model_index+1)].copy_(self.flattened_models_on_cpu[self.model_info_dict[model_key]], non_blocking=True)
            load_event.record(self.load_streams_dict[stream_idx])
        
        # Memo the time when load finishes -> This is used to check loading finish from compute_stream
        self.load_events_list[current_chunk_idx] = load_event
        
        # Measure event fin
        if self.measure_timing:
            load_end.record(self.load_streams_dict[stream_idx])
            self.load_records.append((load_start, load_end))
    
    def update_for_chunks2(self, current_chunk_idx, current_data_idxes_and_model_mapping_infos_chunk):
        # CPU required operations
        stream_idx = current_chunk_idx % self.num_load_streams
        _, model_keys = current_data_idxes_and_model_mapping_infos_chunk
        
        # Measure event starts
        if self.measure_timing:
            load_start = torch.cuda.Event(enable_timing=True)
            load_end   = torch.cuda.Event(enable_timing=True)
            load_start.record(self.load_streams_dict[stream_idx])
        
        # Copy to fixed GPU space by chunks
        load_event = torch.cuda.Event()
        with torch.cuda.stream(self.load_streams_dict[stream_idx]):
            for model_index, model_key in enumerate(model_keys):
                self.pinned_gpu_mem_dict[stream_idx][model_index:(model_index+1)].copy_(self.flattened_models_on_cpu[self.model_info_dict[model_key]], non_blocking=True)
            load_event.record(self.load_streams_dict[stream_idx])
        
        # Memo the time when load finishes -> This is used to check loading finish from compute_stream
        self.load_events_list[current_chunk_idx] = load_event
        
        # Measure event fin
        if self.measure_timing:
            load_end.record(self.load_streams_dict[stream_idx])
            self.load_records.append((load_start, load_end))
    
    def forward_for_block(self, previous_chunk_idx, tensors, prev_data_and_model_mapping_info_by_list_chunk):
        # CPU required operations
        stream_idx = previous_chunk_idx % self.num_load_streams
        num_model_keys=len(prev_data_and_model_mapping_info_by_list_chunk)
        data_indexes = torch.tensor([data_idx for data_idx, model_key in prev_data_and_model_mapping_info_by_list_chunk], dtype=torch.long)
        
        # Forward logic on GPU
        # 0) Wait for load_stream to finish
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(self.load_events_list[previous_chunk_idx])
            
            # Measure event start
            if self.measure_timing:
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end   = torch.cuda.Event(enable_timing=True)
                compute_start.record(self.compute_stream)
                
            # 1) Disassemble loaded weights from GPU chunk space into params, buffers space
            # This is done here as we have 2 GPU chunk spaces but only 1 params+buffers space and as this processes overhead is small
            # 1-1) Params
            for name in self.stacked_model_params_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_params_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)

            params_slice = {
                name: p[:num_model_keys]
                for name, p in self.stacked_model_params_gpu.items()
            }
            

            # 1-2) Buffers
            for name in self.stacked_model_buffers_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_buffers_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)
                
            buffers_slice = {
                name: b[:num_model_keys]
                for name, b in self.stacked_model_buffers_gpu.items()
            }

            # 2) Copy data indexes to GPU + Generate flattened tensors
            self.data_indexes_space[:num_model_keys].copy_(data_indexes)
            flattened_tensors = tensors.index_select(0, self.data_indexes_space[:num_model_keys]) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
            flattened_tensors = flattened_tensors.unsqueeze(1) # (K, 1, C, H, W): Add batch dimension as model expects it
            
            # 3) Perform forwarding
            # flattened_outputs = self.forward_logic(self.stacked_model_params_gpu, self.stacked_model_buffers_gpu, flattened_tensors) # (K, ...)
            flattened_outputs = self.forward_logic(params_slice, buffers_slice, flattened_tensors) # (K, ...)
            
        # Measure event fin
        if self.measure_timing:
            compute_end.record(self.compute_stream)
            self.compute_records.append((compute_start, compute_end))
        
        return flattened_outputs
    
    def forward_for_block2(self, previous_chunk_idx, tensors, prev_data_idxes_and_model_mapping_infos_chunk):
        # CPU required operations
        stream_idx = previous_chunk_idx % self.num_load_streams
        data_idxes, model_keys = prev_data_idxes_and_model_mapping_infos_chunk
        num_model_keys=len(model_keys)
        data_indexes = torch.tensor(data_idxes, dtype=torch.long)
        
        # 0) Position index tensor generation
        # i = model index, j = data index
        results_indexing_dict = {
            (data_idx, model_key):(i, j)
            for i, model_key in enumerate(model_keys)
            for j, data_idx in enumerate(data_idxes)
        }
        
        # Forward logic on GPU
        # 1) Wait for load_stream to finish
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(self.load_events_list[previous_chunk_idx])
            
            # Measure event start
            if self.measure_timing:
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end   = torch.cuda.Event(enable_timing=True)
                compute_start.record(self.compute_stream)
            
            # 2) Disassemble loaded weights from GPU chunk space into params, buffers space
            # This is done here as we have 2 GPU chunk spaces but only 1 params+buffers space and as this processes overhead is small
            # 2-1) Params
            for name in self.stacked_model_params_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_params_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)

            params_slice = {
                name: p[:num_model_keys]
                for name, p in self.stacked_model_params_gpu.items()
            }
            
            # 2-2) Buffers
            for name in self.stacked_model_buffers_gpu.keys():
                start, length = self.offset_table[name]
                shape = self.shape_table[name]
                
                flat_slice = self.pinned_gpu_mem_dict[stream_idx][:num_model_keys, start:start+length]
                self.stacked_model_buffers_gpu[name][:num_model_keys].copy_(flat_slice.reshape(num_model_keys, *shape), non_blocking=True)
                
            buffers_slice = {
                name: b[:num_model_keys]
                for name, b in self.stacked_model_buffers_gpu.items()
            }
            
            # 3) Copy data indexes to GPU + Generate flattened tensors
            self.data_indexes_space[:len(data_indexes)].copy_(data_indexes)
            flattened_tensors = tensors.index_select(0, self.data_indexes_space[:len(data_indexes)]) # (K, C, H, W): flattened means, inputs are flattened out for vmap forwarding
            # flattened_tensors = flattened_tensors.unsqueeze(1) # (K, 1, C, H, W): Add batch dimension as model expects it
            flattened_tensors = flattened_tensors.unsqueeze(0).expand(num_model_keys, -1, -1, -1, -1) # (Vectorization size, K, C, H, W) :via unsqueeze(0), make (1, K, C, H, W) and via expand, make the vmap_dimension
            flattened_outputs = self.forward_logic(params_slice, buffers_slice, flattened_tensors) # (Vectorization size, K, C, H, W)
        
        # Measure event fin
        if self.measure_timing:
            compute_end.record(self.compute_stream)
            self.compute_records.append((compute_start, compute_end))

        return flattened_outputs, results_indexing_dict
    
    def get_measure_timing_results(self, batch_size=None, result_path=None):
        if not self.measure_timing:
            message=f"Error, you didn't measure any timing"
            stop_sys(message)
        else:
            def drop_warmup(x, n):
                if len(x) <= n:
                    message = f"Cannot drop {n} samples when I only have {len(x)} samples in the list, stopping sys"
                    stop_sys(message)
                else:
                    dropped_x = x[n:] 
                    return dropped_x
            
            def avg_of_list(x):
                if len(x) < 1:
                    message = f"We cannot calculate the avg for:{x}"
                    stop_sys(message, raise_error=True)
                else:
                    avg = sum(x)/len(x)
                    return avg
                
            log_lines = []   # <-- collect logs here
            
            # Make sure every operation is really finished
            load_times    = [s.elapsed_time(e) for (s,e) in self.load_records]    # ms
            load_times_len = len(load_times)
            compute_times = [s.elapsed_time(e) for (s,e) in self.compute_records] # ms
            compute_times_len = len(compute_times)
            end_to_end_times = [(e-s)*1000 for (s,e) in self.end_to_end_times] # ms
            end_to_end_times_len = len(end_to_end_times)
            
            load_times_warmup = drop_warmup(load_times, load_times_len//2)
            compute_times_warmup = drop_warmup(compute_times, compute_times_len//2)
            end_to_end_times_warmup = drop_warmup(end_to_end_times, end_to_end_times_len//2)
            
            message = f"Have dropped {end_to_end_times_warmup} batches for warmup"
            warn_user(message=message, warning_time=0)
            
            load_times_avg = avg_of_list(load_times_warmup)
            compute_times_avg = avg_of_list(compute_times_warmup)
            end_to_end_times_avg = avg_of_list(end_to_end_times_warmup)
            
            message=f"AVG load   : {load_times_avg:.3f} ms"
            print(message)
            log_lines.append(message)
            log_lines.append(f"#load: {len(load_times_warmup)}")
            
            message=f"AVG compute: {compute_times_avg:.3f} ms"
            print(message)
            log_lines.append(message)
            log_lines.append(f"#compute: {len(compute_times_warmup)}")
            
            message=f"AVG end2end: {end_to_end_times_avg:.3f} ms"
            print(message)
            log_lines.append(message)
            log_lines.append(f"#end_to_end: {len(end_to_end_times_warmup)}")
            
            # Measure overlap of compute and load
            overlaps = []
            if batch_size!=None and batch_size>self.num_per_stream:
                num_stages = (batch_size + self.num_per_stream - 1) // self.num_per_stream
                message="num_stages", num_stages
                print(message)
                log_lines(message)
                
                def group_by_stages(records, num_stages):
                    return [records[i:i+num_stages] for i in range(0, len(records), num_stages)]
                load_groups = group_by_stages(self.load_records, num_stages)
                load_groups_wo = drop_warmup(load_groups, len(load_groups)//2)
                compute_groups = group_by_stages(self.compute_records, num_stages)
                compute_groups_wo = drop_warmup(compute_groups, len(compute_groups)//2)
                
                for load_group, compute_group in zip(load_groups_wo, compute_groups_wo):
                    for stage_idx in range(num_stages-1):
                        load_start, load_end = load_group[stage_idx+1]
                        compute_start, compute_end = compute_group[stage_idx]
                        
                        overlap = load_start.elapsed_time(compute_end)
                        overlap = max(0.0, overlap)
                    
                        overlaps.append(overlap)
                        
                        overlaps_avg=sum(overlaps)/len(overlaps)
                message=f"Overlap:{overlaps_avg}"
                print(message)
                log_lines.append(message)
            else:
                message=f"For batch_size:{batch_size}, num_per_stream:{self.num_per_stream} overlap is impossible"
                print(message)
                log_lines.append(message)
        
        if result_path!=None:
            atomic_to_txt(text=log_lines, path=result_path)
            
# MoE model definitions
# 1) Conditional MoE Model 
class ConditionalMoEModel(nn.Module):
    def __init__(self, num_experts, model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, return_val="output"):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = []
        elif isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
            
        # feature_extractor
        self.feature_extractor = FeatureExtractor(feature_extractor_name=model_name, pretrained=pretrained)
        if last_layer_only:
            self.feature_extractor.freeze_weights()
        feature_size = self.feature_extractor.get_feature_size()
        
        # Gating Network
        self.gating_network = FullyConnectedLayers(input_size=feature_size, out_size=num_experts)
        
        # Experts
        self.expert_fc_layers_dict = nn.ModuleDict(
            {
                str(expert_idx):FullyConnectedLayers(input_size=feature_size, hidden_layers=hidden_layers, out_size=num_out) # -> expert_idx should be string!
                for expert_idx in range(num_experts) 
            }
        )
        
        # Set forward_function depending on return_val
        self.set_return_val(return_val)
        
        # Values to use in forwarding
        self.num_out = num_out
    
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
        gate_indexes = self.gating_network(features.detach()).argmax(dim=1)
        
        outputs = torch.zeros(features.size(0), self.num_out, device=features.device)

        # Initial Loop based Approach
        # for sample_idx, gate_idx in enumerate(gate_indexes):
        #     expert_key = str(gate_idx.item())
        #     outputs[sample_idx] = self.expert_fc_layers_dict[expert_key](features[sample_idx])

        # Vectorized group-by expert Approach
        for expert_idx, expert in self.expert_fc_layers_dict.items():
            expert_idx_int = int(expert_idx)

            mask = gate_indexes == expert_idx_int  # [B]
            if mask.any():
                # select only samples routed to this expert
                expert_features = features[mask]        # [N_i, D]
                expert_outputs = expert(expert_features)  # [N_i, C]

                # scatter back
                outputs[mask] = expert_outputs

        return outputs
    
    def forward_return_features_and_outputs(self, tensors):
        features = self.feature_extractor(tensors)
        gate_indexes = self.gating_network(features.detach()).argmax(dim=1)
        
        outputs = torch.zeros(features.size(0), self.num_out, device=features.device)

        # Initial Loop based Approach
        # for sample_idx, gate_idx in enumerate(gate_indexes):
        #     expert_key = str(gate_idx.item())
        #     outputs[sample_idx] = self.expert_fc_layers_dict[expert_key](features[sample_idx])

        # Vectorized group-by expert Approach
        for expert_idx, expert in self.expert_fc_layers_dict.items():
            expert_idx_int = int(expert_idx)

            mask = gate_indexes == expert_idx_int  # [B]
            if mask.any():
                # select only samples routed to this expert
                expert_features = features[mask]        # [N_i, D]
                expert_outputs = expert(expert_features)  # [N_i, C]

                # scatter back
                outputs[mask] = expert_outputs
            
        return features, outputs

# 2) Matryoshka MoE Model Classifiers
class MatryoshkaClassifiers(nn.Module):
    def __init__(self,
        # Classifier configs
        input_size:int=None, hidden_layers:Union[int, List[int]]=None, out_size:int=None,
        
        # Matryoshka_expert configs
        num_matryoshka_layers=1
        ):
        super.init()
        self.classifiers = nn.ModuleDict(
            {
                str(matryoshka_idx):FullyConnectedLayers(input_size=feature_size, hidden_layers=hidden_layers, out_size=num_out) # -> expert_idx should be string!
                for matryoshka_idx in range(num_matryoshka_layers) 
            }
        )
    
    def initialize(self):
        pass
    
    def custom_forward_offline(self, tensors):
        pass
    
    def custom_forward_online(self, tensors, resource_budget):
        pass
    
# 3) End-to-end Matryoshka MoE Model
class EndToEndMatryoshkaMoEModel(nn.Module):
    def __init__(self, 
        # Default Params For Feature Extractor + Main Classifier
        model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, return_val="output",
        num_experts=1, num_matryoshka_layers=1, temperature_val=1):
        super.init()
        
        # 1) FeatureExtractor
        self.main_model = FeatureAndOutputModel(model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, return_val="output")
        
        # 2) Classifiers
        if hidden_layers is None:
            hidden_layers = []
        elif isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
            
        feature_size = self.main_model.feature_extractor.get_feature_size()
        self.gating_classifier = FullyConnectedLayers(input_size=feature_size, hidden_layers=hidden_layers, out_size=num_experts)
        self.matryoshka_expert_classifiers = {
            matryoshka_idx:MatryoshkaClassifiers(
                # Classifier configs
                input_size=feature_size, hidden_layers=hidden_layers, out_size=num_out,
                
                # Matryoshka_expert configs
                num_matryoshka_layers=num_matryoshka_layers
            )
            for matryoshka_idx in range(num_experts)
        }
        
        # 3) Additional Factors For Forwarding Optimizations
        self.temperature_val = temperature_val
    
    def initialize(self):
        pass
    
    def custom_forward_offline(self, tensors):
        # main model forwarding
        features, outputs = self.main_model(tensors)
        
        # temperature scaling
        
        pass
    
    def custom_forward_online(self, tensors, resource_budget):
        message = "Custom forwarding of end-to-end matryoshka moe model is currently not implemented"
        stop_sys(message, raise_error=True)
        
    def forward(self, *args, **kwargs):
        return self.forward_function(*args, **kwargs)
        
# Normalization params aquire
def get_normalize_params_from_model_name(model_name):
    imagenet_normalize_params = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    google_vit_normalize_params = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    google_vit_model_names = [
        "ViT_Tiny_timm",
        "ViT_Small_timm", "ViT_Small_32_timm",
        "ViT_Base_timm", "ViT_Base_32_timm",
        "ViT_Large_timm", "ViT_Large_32_timm",
    ]
    if model_name in google_vit_model_names:
        normalize_params = google_vit_normalize_params
    else:
        normalize_params = imagenet_normalize_params
    return normalize_params

# Input size aquire
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

# named model generation
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
        elif model_name == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=pretrained)
        elif model_name == "wide_resnet101_2":
            model = models.wide_resnet101_2(pretrained=pretrained)
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
            model = timm.create_model(model_proxy, pretrained=True, num_classes=0)
            
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
            # Remove head layer && Set hidden_layers to head.in_features so classification layer could be added in step - 4
            # feature_size = model.head.in_features
            # feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_size = model.num_features
            feature_extractor = model # For Google ViT, There is a way to just remove feature extractor:use num_classes=0 option!
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
    
# Generate model, load weights, move to sepecific device
def get_feature_extractor_from_params(feature_extractor_name, pretrained=True, weight_path=None, device='cpu'):
    # This is actually a feature_extractor
    feature_extractor = FeatureExtractor(feature_extractor_name=feature_extractor_name, pretrained=pretrained)
    
    if weight_path!=None:
        feature_extractor.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
    
    feature_extractor = feature_extractor.to(device)
    return feature_extractor

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
    # print("Watch point Start")
    if weight_path!=None:
        print(weight_path, device)
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=False))
        # print("Weight load is done!")
    
    model = model.to(device)
    # print("Watch point fin")
    
    return model

def get_feature_and_output_model_from_params(model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, weight_path=None, device='cpu'):
    # This is actually a feature_and_output_model with reuturn_func set to features and outputs
    model = FeatureAndOutputModel(model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, return_val="feature_and_output")
    
    if weight_path!=None:
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=False))
    
    model = model.to(device)
    
    return model

def get_gating_network_from_params(feature_extractor_name, hidden_layers:Union[int, List[int]]=None, num_out:int=5, weight_path=None, device="cpu"):
    # This is actually fc_layers for a specific feature extractor
    gating_network = FullyConnectedLayers(feature_extractor_name=feature_extractor_name, hidden_layers=hidden_layers, out_size=num_out)
    
    if weight_path!=None:
        gating_network.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
        
    gating_network = gating_network.to(device)
    
    return gating_network

def get_vmap_default_model_from_weight_paths_dict_and_params(model_name, pretrained=True, hidden_layers:Union[int, List[int]]=None, num_out=6, last_layer_only=False, weight_paths_dict:dict=None, device='cpu', random_test=False):
    vmap_model = VMapModel1()
    vmap_model.initialize_vmap_model_from_params(
        model_name, pretrained=pretrained, hidden_layers=hidden_layers, num_out=num_out, last_layer_only=last_layer_only, weight_paths_dict=weight_paths_dict, device=device, random_test=random_test,
    )
    return vmap_model

