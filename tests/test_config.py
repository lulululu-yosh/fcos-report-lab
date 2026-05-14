from fcos_report.config import ensure_output_dirs, load_config


def test_default_config_loads():
    config = load_config()
    assert config["models"]["fcos"]["torchvision_name"] == "fcos_resnet50_fpn"
    assert config["models"]["retinanet"]["torchvision_name"] == "retinanet_resnet50_fpn"


def test_output_dirs(tmp_path):
    config = load_config()
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    paths = ensure_output_dirs(config)
    assert paths["figures"].exists()
    assert paths["metrics"].exists()
