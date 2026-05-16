from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.models.detection.fcos import FCOS, FCOSClassificationHead, FCOSRegressionHead
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops, generalized_box_iou_loss, sigmoid_focal_loss


class FCOSNoCenternessHead(nn.Module):
    """FCOS head without a centerness loss/branch effect.

    It keeps classification and box regression only. A dummy zero centerness tensor is
    returned only to satisfy torchvision FCOS.forward's expected output dictionary;
    the subclassed post-processing below ignores it.
    """

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int = 4) -> None:
        super().__init__()
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)
        # Remove the learnable centerness branch from optimization.
        self.regression_head.bbox_ctrness = nn.Identity()

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_full((len(matched_idxs_per_image),), -1)
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clamp(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clamp(min=0)]
                gt_classes_targets[matched_idxs_per_image < 0] = -1
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        anchors = torch.stack(anchors)

        foreground_mask = all_gt_classes_targets >= 0
        num_foreground = max(1, int(foreground_mask.sum().item()))

        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foreground_mask, all_gt_classes_targets[foreground_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum") / num_foreground

        pred_boxes = self.box_coder.decode(bbox_regression, anchors)
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foreground_mask],
            all_gt_boxes_targets[foreground_mask],
            reduction="sum",
        ) / num_foreground

        return {"classification": loss_cls, "bbox_regression": loss_bbox_reg}

    def forward(self, features: list[Tensor]) -> dict[str, Tensor]:
        cls_logits = self.classification_head(features)
        # Run only the conv + bbox_reg path from FCOSRegressionHead.
        all_bbox_regression = []
        for x in features:
            bbox_feature = self.regression_head.conv(x)
            bbox_regression = nn.functional.relu(self.regression_head.bbox_reg(bbox_feature))
            n, _, h, w = bbox_regression.shape
            bbox_regression = bbox_regression.view(n, -1, 4, h, w).permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(n, -1, 4)
            all_bbox_regression.append(bbox_regression)
        bbox_regression = torch.cat(all_bbox_regression, dim=1)
        bbox_ctrness = bbox_regression.new_zeros((*bbox_regression.shape[:2], 1))
        return {"cls_logits": cls_logits, "bbox_regression": bbox_regression, "bbox_ctrness": bbox_ctrness}


class FCOSNoCenterness(FCOS):
    """FCOS inference that ranks detections with classification score only."""

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        num_images = len(image_shapes)
        detections = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes, image_scores, image_labels = [], [], []
            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes
                boxes_per_level = self.box_coder.decode(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)
                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            detections.append({"boxes": image_boxes[keep], "scores": image_scores[keep], "labels": image_labels[keep]})
        return detections
