# Modified files

## To support swin backbones
* models/backbone.py
* models/swin_transformer.py
* mmcv_custom

## To support eval in the training set
* datasets/coco.py
* datasets/\_\_init\_\_.py

## To support Hybird-branch, tricks and checkpoint
* main.py
* engine.py
* models/deformable_detr.py
* models/deformable_transformer.py

## To support fp16
* models/ops/modules/ms_deform_attn.py
* models/ops/functions/ms_deform_attn_func.py

## To fix a pytorch version bug
* util/misc.py

# Addictional packages needed

* wandb: for logging
* mmdet: for swin backbones
* mmcv: for swin backbones
* timm: for swin backbones

# To train a model

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path>
```

# To eval a model

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path> --eval
```