from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import numpy as np
from tabulate import tabulate

from metrics import compute_recall
# from tidecv import TIDE, datasets


def _mean_valid_precision(values):
    """Average valid COCO precision entries (invalid entries are -1)."""
    values = values[values > -1]
    return float(np.mean(values)) if values.size else float("nan")


def print_per_category_ap(coco_eval, coco_gt):
    """Print MMDetection-style per-category bbox AP metrics."""
    # Shape: (IoU thresholds, recall thresholds, classes, areas, max dets)
    precisions = coco_eval.eval["precision"]
    category_ids = coco_eval.params.catIds
    assert len(category_ids) == precisions.shape[2]

    area_indices = {
        label: coco_eval.params.areaRngLbl.index(label)
        for label in ("all", "small", "medium", "large")
    }
    max_dets_index = coco_eval.params.maxDets.index(100)
    iou50_index = int(
        np.flatnonzero(np.isclose(coco_eval.params.iouThrs, 0.50))[0])
    iou75_index = int(
        np.flatnonzero(np.isclose(coco_eval.params.iouThrs, 0.75))[0])

    rows = []
    for class_index, category_id in enumerate(category_ids):
        category = coco_gt.loadCats([category_id])[0]
        rows.append([
            category["name"],
            _mean_valid_precision(
                precisions[:, :, class_index, area_indices["all"],
                           max_dets_index]),
            _mean_valid_precision(
                precisions[iou50_index, :, class_index,
                           area_indices["all"], max_dets_index]),
            _mean_valid_precision(
                precisions[iou75_index, :, class_index,
                           area_indices["all"], max_dets_index]),
            _mean_valid_precision(
                precisions[:, :, class_index, area_indices["small"],
                           max_dets_index]),
            _mean_valid_precision(
                precisions[:, :, class_index, area_indices["medium"],
                           max_dets_index]),
            _mean_valid_precision(
                precisions[:, :, class_index, area_indices["large"],
                           max_dets_index]),
        ])

    print("\nPer-category bbox AP:")
    print(
        tabulate(
            rows,
            headers=[
                "category", "mAP", "mAP_50", "mAP_75", "mAP_s",
                "mAP_m", "mAP_l"
            ],
            tablefmt="pipe",
            floatfmt=".3f",
            stralign="center",
            numalign="center",
        ))

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Evaluate COCO detections.")
parser.add_argument('--cocoGt', type=str, required=True, help="Path to the ground truth COCO annotations (coco_annotations.json).")
parser.add_argument('--cocoDt', type=str, required=True, help="Path to the detection result file (e.g., detections.json).")
args = parser.parse_args()

# Load the ground truth and detection results
cocoGt = COCO(args.cocoGt)
cocoDt = cocoGt.loadRes(args.cocoDt)

# Perform COCO evaluation
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
print_per_category_ap(cocoEval, cocoGt)

# Compute recall
compute_recall(cocoDt, cocoGt)


# # tide
# gt = datasets.COCO(args.cocoGt)
# results=datasets.COCOResult(args.cocoDt)
# tide = TIDE()
# tide.evaluate_range(gt, results, mode=TIDE.BOX) # Use TIDE.MASK for masks
# tide.summarize()  # Summarize the results as tables in the console
