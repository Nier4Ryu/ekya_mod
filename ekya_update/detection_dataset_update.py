import copy, os
from torch.utils.data import DataLoader
from lite.src.datasets.datasets_for_detection import (
    DetectionDatasetPILV2,
    collate_detection_batch,
    collate_detection_batch_with_metadata,
    load_detection_dataset_dict_from_json_path,
)
from ekya_update.common import stop_sys


def load_detection_dataset_dict_from_path(json_path):
    if json_path is not None and os.path.exists(json_path):
        dataset_dict = load_detection_dataset_dict_from_json_path(json_path=json_path)
    else:
        message = f"detection json_path:{json_path} does not exist"
        stop_sys(message, raise_error=True)
    
    return dataset_dict


def get_detection_dataset_dict_with_samples_from_params(source_dataset_dict, samples):
    dataset_dict = copy.deepcopy(source_dataset_dict)
    dataset_dict["samples"] = [
        copy.deepcopy(sample)
        for sample in samples
    ]
    
    return dataset_dict


def get_empty_detection_dataset_dict_from_source(source_dataset_dict):
    dataset_dict = get_detection_dataset_dict_with_samples_from_params(
        source_dataset_dict=source_dataset_dict,
        samples=[],
    )
    
    return dataset_dict


def get_detection_dataset_num_samples_from_dict(dataset_dict):
    num_samples = len(dataset_dict["samples"])
    
    return num_samples


def get_detection_dataset_dict_with_sample_range_from_params(dataset_dict, start_idx, end_idx):
    samples = [
        copy.deepcopy(sample)
        for sample in dataset_dict["samples"][start_idx:end_idx]
    ]
    sliced_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
        source_dataset_dict=dataset_dict,
        samples=samples,
    )
    
    return sliced_dataset_dict


def get_aligned_detection_dataset_dict_from_target_num_samples(dataset_dict, target_num_samples):
    if target_num_samples is not None:
        original_num_samples = len(dataset_dict["samples"])
        if original_num_samples > target_num_samples:
            aligned_samples = [
                copy.deepcopy(sample)
                for sample in dataset_dict["samples"][:target_num_samples]
            ]
        elif original_num_samples < target_num_samples:
            if original_num_samples > 0:
                num_extra = target_num_samples - original_num_samples
                num_full_repeats = num_extra // original_num_samples
                num_remainder = num_extra % original_num_samples
                aligned_samples = [
                    copy.deepcopy(sample)
                    for sample in dataset_dict["samples"]
                ]
                for repeat_idx in range(num_full_repeats):
                    repeated_samples = [
                        copy.deepcopy(sample)
                        for sample in dataset_dict["samples"]
                    ]
                    aligned_samples.extend(repeated_samples)
                remainder_samples = [
                    copy.deepcopy(sample)
                    for sample in dataset_dict["samples"][:num_remainder]
                ]
                aligned_samples.extend(remainder_samples)
            else:
                message = "Cannot align an empty detection dataset"
                stop_sys(message, raise_error=True)
        else:
            aligned_samples = [
                copy.deepcopy(sample)
                for sample in dataset_dict["samples"]
            ]
        
        aligned_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
            source_dataset_dict=dataset_dict,
            samples=aligned_samples,
        )
    else:
        aligned_dataset_dict = dataset_dict
    
    return aligned_dataset_dict


def get_padded_detection_dataset_dict_from_params(dataset_dict, multiple):
    num_samples = len(dataset_dict["samples"])
    if multiple > 0:
        num_padding_samples = (multiple - (num_samples % multiple)) % multiple
    else:
        message = f"multiple must be positive for detection padding, got:{multiple}"
        stop_sys(message, raise_error=True)
    
    if num_padding_samples > 0:
        if num_samples > 0:
            padded_samples = [
                copy.deepcopy(sample)
                for sample in dataset_dict["samples"]
            ]
            last_sample = copy.deepcopy(dataset_dict["samples"][-1])
            for padding_idx in range(num_padding_samples):
                padding_sample = copy.deepcopy(last_sample)
                padding_sample["sample_id"] = f"{last_sample.get('sample_id', 'sample')}_dummy_{padding_idx}"
                padding_sample["is_dummy"] = True
                padded_samples.append(padding_sample)
            padded_dataset_dict = get_detection_dataset_dict_with_samples_from_params(
                source_dataset_dict=dataset_dict,
                samples=padded_samples,
            )
        else:
            message = "Cannot pad an empty detection dataset"
            stop_sys(message, raise_error=True)
    else:
        padded_dataset_dict = dataset_dict
    
    return padded_dataset_dict


def get_detection_dataloader_from_dataset_dict(
    dataset_dict,
    batch_size,
    shuffle=False,
    num_workers=4,
    return_vals="image_target_and_metadata",
    drop_empty_targets=False,
):
    dataset = DetectionDatasetPILV2(
        dataset_dict=dataset_dict,
        return_vals=return_vals,
        drop_empty_targets=drop_empty_targets,
    )
    
    if return_vals=="image_and_target":
        collate_fn = collate_detection_batch
    elif return_vals=="image_target_and_metadata":
        collate_fn = collate_detection_batch_with_metadata
    else:
        message = f"return_vals:{return_vals} is not supported for Ekya detection dataloader"
        stop_sys(message, raise_error=True)
    
    dataloader_shuffle = False if len(dataset)==0 else shuffle
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return dataloader


def get_detection_train_dataloader_from_dataset_dict(dataset_dict, batch_size, num_workers=4):
    dataloader = get_detection_dataloader_from_dataset_dict(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        return_vals="image_and_target",
        drop_empty_targets=True,
    )
    
    return dataloader


def get_detection_inference_dataloader_from_dataset_dict(dataset_dict, batch_size, num_workers=4):
    dataloader = get_detection_dataloader_from_dataset_dict(
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        return_vals="image_target_and_metadata",
        drop_empty_targets=False,
    )
    
    return dataloader
