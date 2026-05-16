# FCOS on Pascal VOC: train, test export, and no-centerness ablation

Place these files in your project root, for example under `voc_fcos/`, then run from that directory.

## Install

```bash
pip install torch torchvision pycocotools tqdm pillow
```

## Train standard FCOS on VOC2012 trainval

```bash
python train_voc_fcos.py \
  --voc-root ./data \
  --year 2012 \
  --image-set trainval \
  --download \
  --epochs 12 \
  --batch-size 4 \
  --lr 0.005 \
  --output-dir outputs/voc2012_fcos
```

## Train ablation without centerness

```bash
python train_voc_fcos.py \
  --voc-root ./data \
  --year 2012 \
  --image-set trainval \
  --download \
  --epochs 12 \
  --batch-size 4 \
  --lr 0.005 \
  --no-centerness \
  --output-dir outputs/voc2012_fcos_no_centerness
```

## Export VOC test submission files

For standard FCOS:

```bash
python export_voc_test_submission.py \
  --voc-root ./data \
  --year 2012 \
  --image-set test \
  --checkpoint outputs/voc2012_fcos/model_final.pth \
  --output-dir outputs/voc2012_submit_fcos
```

For no-centerness ablation:

```bash
python export_voc_test_submission.py \
  --voc-root ./data \
  --year 2012 \
  --image-set test \
  --checkpoint outputs/voc2012_fcos_no_centerness/model_final.pth \
  --no-centerness \
  --output-dir outputs/voc2012_submit_fcos_no_centerness
```

Each output directory contains 20 files named like `comp4_det_test_person.txt`.
Each line follows the VOC server format:

```text
<image_id> <confidence> <xmin> <ymin> <xmax> <ymax>
```
