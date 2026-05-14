import torch

from fcos_report.coco import coco_result_from_output


def test_coco_result_schema():
    output = {
        "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([1]),
    }
    result = coco_result_from_output(123, output)
    assert result == [{
        "image_id": 123,
        "category_id": 1,
        "bbox": [1.0, 2.0, 10.0, 20.0],
        "score": 0.8999999761581421,
    }]
