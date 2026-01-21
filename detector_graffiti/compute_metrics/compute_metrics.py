import numpy as np
import torch


def compute_iou(pred_boxes, target_boxes):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        pred_boxes (Tensor): Predicted boxes in (x1, y1, x2, y2)
        format, shape (N, 4)
        target_boxes (Tensor): Ground truth boxes in (x1, y1, x2, y2)
        format, shape (M, 4)

    Returns:
        Tensor: IoU matrix of shape (N, M) where element (i,j) is IoU between
                pred_boxes[i] and target_boxes[j]
    """
    area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    )

    # (N_pred, N_gt, 2)
    lt = torch.max(pred_boxes[:, None, :2], target_boxes[:, :2])
    # (N_pred, N_gt, 2)
    rb = torch.min(pred_boxes[:, None, 2:], target_boxes[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N_pred, N_gt, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N_pred, N_gt)

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-8)
    return iou


def match_predictions_to_targets(pred_boxes, target_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using greedy IoU matching.

    Args:
        pred_boxes (Tensor): Predicted boxes, shape (N, 4)
        target_boxes (Tensor): Ground truth boxes, shape (M, 4)
        iou_threshold (float): Minimum IoU for a match

    Returns:
        int: Number of correctly matched predictions (true positives)
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0

    ious = compute_iou(pred_boxes, target_boxes)
    matched_gt = set()
    correct = 0

    for pred_idx in range(ious.shape[0]):
        iou_values = ious[pred_idx]
        best_gt_idx = iou_values.argmax().item()
        best_iou = iou_values[best_gt_idx].item()

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            correct += 1

    return correct


def compute_metrics(eval_preds, device=None, iou_threshold=0.5):
    """
    Compute object detection metrics from evaluation predictions.

    Args:
        eval_preds (Tuple[Tensor, Tuple[List[Tensor], List[Tensor]]]):
            Tuple containing:
                - count_boxes (Tensor): Tensor with box counts for each image
                - (preds_batch, labels_batch): Lists of predicted and
                  ground truth boxes
        device (Optional[torch.device]): Device for computation
        iou_threshold (float): IoU threshold for correct detection

    Returns:
        Dict[str, float]: Dictionary containing:
            - precision (float): Precision metric
            - recall (float): Recall metric
            - f1 (float): F1 score
            - false positive percentage (float): Percentage of images
              with false positives
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    count_boxes, (preds_batch, labels_batch) = eval_preds
    count_boxes = [
        (int(count_boxes[i].item()), int(count_boxes[i + 1].item()))
        for i in range(0, len(count_boxes), 2)
    ]

    total_correct = 0
    total_pred = 0
    total_true = 0
    fp_count = 0

    curr_pred = 0
    curr_gt = 0
    for count_pred, count_gt in count_boxes:
        total_pred += count_pred
        total_true += count_gt

        if count_pred > 0 and count_gt > 0:
            end_pred = curr_pred + count_pred
            end_gt = curr_gt + count_gt
            pred = preds_batch[0][curr_pred:end_pred]
            gt = labels_batch[0][curr_gt:end_gt]

            if isinstance(pred, np.ndarray):
                pred = torch.tensor(pred, dtype=torch.float32)
            if isinstance(gt, np.ndarray):
                gt = torch.tensor(gt, dtype=torch.float32)

            correct = match_predictions_to_targets(
                pred, gt, iou_threshold=iou_threshold
            )
            total_correct += correct
            if count_pred - correct:
                fp_count += 1
        else:
            if count_pred > 0 and count_gt == 0:
                fp_count += count_pred

        curr_pred += count_pred
        curr_gt += count_gt

    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_true if total_true > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    false_positive_percentage = fp_count / len(count_boxes)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false positive percentage": false_positive_percentage,
    }
