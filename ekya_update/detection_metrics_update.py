import contextlib, io
from ekya_update.common import stop_sys


def get_detection_map_metrics_from_dataset_dicts(reference_dataset_dict, prediction_dataset_dict):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        message = f"pycocotools is required for Ekya detection mAP evaluation: {error}"
        stop_sys(message, raise_error=True)
    
    coco_gt_dict, detection_results, metric_counts = get_coco_gt_and_detection_results_from_dataset_dicts(
        reference_dataset_dict=reference_dataset_dict,
        prediction_dataset_dict=prediction_dataset_dict,
    )
    
    if metric_counts["num_gt_boxes"]==0:
        if metric_counts["num_prediction_boxes"]==0:
            metrics = get_empty_detection_metrics_with_score(score=1.0, metric_counts=metric_counts)
        else:
            metrics = get_empty_detection_metrics_with_score(score=0.0, metric_counts=metric_counts)
    elif metric_counts["num_prediction_boxes"]==0:
        metrics = get_empty_detection_metrics_with_score(score=0.0, metric_counts=metric_counts)
    else:
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(detection_results)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = [image["id"] for image in coco_gt_dict["images"]]
            coco_eval.params.catIds = [category["id"] for category in coco_gt_dict["categories"]]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        stats = coco_eval.stats
        metrics = {
            "map": float(stats[0]),
            "map50": float(stats[1]),
            "map75": float(stats[2]),
            "mar100": float(stats[8]),
        }
        metrics.update(metric_counts)
    
    return metrics


def get_empty_detection_metrics_with_score(score, metric_counts):
    metrics = {
        "map": float(score),
        "map50": float(score),
        "map75": float(score),
        "mar100": float(score),
    }
    metrics.update(metric_counts)
    
    return metrics


def get_coco_gt_and_detection_results_from_dataset_dicts(reference_dataset_dict, prediction_dataset_dict):
    reference_samples = reference_dataset_dict["samples"]
    prediction_samples = prediction_dataset_dict["samples"]
    
    if len(reference_samples)==len(prediction_samples):
        pass
    else:
        message = (
            f"reference/prediction sample count mismatch for detection mAP: "
            f"{len(reference_samples)} vs {len(prediction_samples)}"
        )
        stop_sys(message, raise_error=True)
    
    categories = get_coco_categories_from_dataset_dicts(
        reference_dataset_dict=reference_dataset_dict,
        prediction_dataset_dict=prediction_dataset_dict,
    )
    images = []
    annotations = []
    detection_results = []
    annotation_id = 1
    num_gt_boxes = 0
    num_prediction_boxes = 0
    
    for image_idx, reference_sample in enumerate(reference_samples):
        prediction_sample = prediction_samples[image_idx]
        width = int(reference_sample.get("width", prediction_sample.get("width", 0)))
        height = int(reference_sample.get("height", prediction_sample.get("height", 0)))
        images.append({
            "id": image_idx,
            "file_name": reference_sample.get("img_save_path", str(image_idx)),
            "width": width,
            "height": height,
        })
        
        reference_target = reference_sample.get("target", {})
        reference_boxes = reference_target.get("boxes", [])
        reference_labels = reference_target.get("labels", [])
        for box_idx, box in enumerate(reference_boxes):
            category_id = int(reference_labels[box_idx])
            bbox = get_coco_xywh_bbox_from_xyxy_box(box=box)
            annotation = {
                "id": annotation_id,
                "image_id": image_idx,
                "category_id": category_id,
                "bbox": bbox,
                "area": float(bbox[2]*bbox[3]),
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1
            num_gt_boxes += 1
        
        prediction_target = prediction_sample.get("target", {})
        prediction_boxes = prediction_target.get("boxes", [])
        prediction_labels = prediction_target.get("labels", [])
        prediction_scores = prediction_target.get("scores", [1.0 for box in prediction_boxes])
        for box_idx, box in enumerate(prediction_boxes):
            category_id = int(prediction_labels[box_idx])
            bbox = get_coco_xywh_bbox_from_xyxy_box(box=box)
            detection_result = {
                "image_id": image_idx,
                "category_id": category_id,
                "bbox": bbox,
                "score": float(prediction_scores[box_idx]),
            }
            detection_results.append(detection_result)
            num_prediction_boxes += 1
    
    coco_gt_dict = {
        "info": {"description": "Ekya detection evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    metric_counts = {
        "num_images": len(images),
        "num_gt_boxes": num_gt_boxes,
        "num_prediction_boxes": num_prediction_boxes,
    }
    
    return coco_gt_dict, detection_results, metric_counts


def get_coco_categories_from_dataset_dicts(reference_dataset_dict, prediction_dataset_dict):
    category_by_id = {}
    for dataset_dict in [reference_dataset_dict, prediction_dataset_dict]:
        for category_record in dataset_dict.get("categories", []):
            category_id = int(category_record.get("detector_label", category_record.get("id", 0)))
            if category_id > 0:
                category_by_id[category_id] = {
                    "id": category_id,
                    "name": str(category_record.get("category", category_record.get("name", category_id))),
                    "supercategory": str(category_record.get("supercategory", "object")),
                }
            else:
                pass
    
    for dataset_dict in [reference_dataset_dict, prediction_dataset_dict]:
        for sample in dataset_dict["samples"]:
            labels = sample.get("target", {}).get("labels", [])
            for label in labels:
                category_id = int(label)
                if category_id > 0 and category_id not in category_by_id:
                    category_by_id[category_id] = {
                        "id": category_id,
                        "name": str(category_id),
                        "supercategory": "object",
                    }
                else:
                    pass
    
    if len(category_by_id)>0:
        categories = [
            category_by_id[category_id]
            for category_id in sorted(category_by_id.keys())
        ]
    else:
        categories = [{"id": 1, "name": "object", "supercategory": "object"}]
    
    return categories


def get_coco_xywh_bbox_from_xyxy_box(box):
    x0 = float(box[0])
    y0 = float(box[1])
    x1 = float(box[2])
    y1 = float(box[3])
    width = max(0.0, x1-x0)
    height = max(0.0, y1-y0)
    bbox = [x0, y0, width, height]
    
    return bbox
