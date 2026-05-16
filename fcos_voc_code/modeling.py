from __future__ import annotations

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSHead

from fcos_no_centerness import FCOSNoCenterness, FCOSNoCenternessHead


def build_fcos_voc(num_classes: int = 20, no_centerness: bool = False, min_size: int = 800, max_size: int = 1333):
    """Build FCOS ResNet-50-FPN for Pascal VOC.

    Uses ImageNet-pretrained ResNet-50 backbone. The detection head is randomly
    initialized because VOC has 20 classes, not COCO's category space.
    """
    base = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        score_thresh=0.05,
        nms_thresh=0.6,
        detections_per_img=100,
        topk_candidates=1000,
    )
    if not no_centerness:
        return base

    model = FCOSNoCenterness(
        backbone=base.backbone,
        num_classes=num_classes,
        anchor_generator=base.anchor_generator,
        head=FCOSNoCenternessHead(
            base.backbone.out_channels,
            base.anchor_generator.num_anchors_per_location()[0],
            num_classes,
        ),
        min_size=min_size,
        max_size=max_size,
        score_thresh=0.05,
        nms_thresh=0.6,
        detections_per_img=100,
        topk_candidates=1000,
        image_mean=base.transform.image_mean,
        image_std=base.transform.image_std,
    )
    return model
