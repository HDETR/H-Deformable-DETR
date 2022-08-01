# H-Deformable DETR

This is the official implementation of the paper "[DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)". 

# Modified files compared to vanilla Deformable DETR

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

## Addictional packages needed

* wandb: for logging
* mmdet: for swin backbones
* mmcv: for swin backbones
* timm: for swin backbones

# Installation
We test our models under ```python=3.7.10,pytorch=1.10.1,cuda=10.2```. Other versions might be available as well.

1. Clone this repo
```sh
git https://github.com/HDETR/H-Deformable-DETR.git
cd H-Deformable-DETR
```

2. Install Pytorch and torchvision

Follow the instruction on https://pytorch.org/get-started/locally/.
```sh
# an example:
conda install -c pytorch pytorch torchvision
```

3. Install other needed packages
```sh
pip install -r requirements.txt
pip install openmim
mim install mmcv-full
pip install mmdet
```

4. Compiling CUDA operators
```sh
cd models/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

# Model ZOO

We provide a set of baseline results and trained models available for download in the [H-Deformable detr Model Zoo](MODEL_ZOO.md).

# Data

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
coco/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
# Run
## To train a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path>
```

## To eval a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path> --eval --resume <checkpoint path>
```