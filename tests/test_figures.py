from fcos_report.make_figures import make_all


def test_make_all_figures(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  seed: 42
  output_dir: {output_dir}
data:
  root: data/coco
  image_size_limit: 1333
  smoke_limit: 2
models:
  fcos:
    torchvision_name: fcos_resnet50_fpn
    weights: FCOS_ResNet50_FPN_Weights.DEFAULT
    score_threshold: 0.35
  retinanet:
    torchvision_name: retinanet_resnet50_fpn
    weights: RetinaNet_ResNet50_FPN_Weights.DEFAULT
    score_threshold: 0.35
evaluation:
  batch_size: 1
  num_workers: 0
  max_detections_per_image: 100
  full_limit: null
figures:
  dpi: 90
  style: seaborn-v0_8-whitegrid
""".format(output_dir=(tmp_path / "outputs").as_posix()),
        encoding="utf-8",
    )
    written = make_all(config_path)
    assert any(path.name == "mechanism_centerness_heatmap.png" for path in written)
    assert all(path.exists() for path in written)
