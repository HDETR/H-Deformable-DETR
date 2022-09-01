# H-Deformable-DETR

This is the official implementation of the paper "[DETRs with Hybrid Matching](https://arxiv.org/abs/2207.13080)". 

Authors: Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, Han Hu

## News

**2022.08.31** Code for [H-Deformable-DETR-mmdet](https://github.com/HDETR/H-Deformable-DETR-mmdet) (**support mmdetection2d**) is released. We will also release the code for [H-Mask-Deformable-DETR](https://github.com/HDETR/H-Mask-Deformable-DETR) soon (**strong results on both instance segmentation and panoptic segmentation**).


## Model ZOO

We provide a set of baseline results and trained models available for download:

### Models with the ResNet-50 backbone
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">query</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">43.7</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">46.8</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">47.0</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">49.0</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/12eps/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">48.7</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/36eps/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">R50</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">50.0</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/r50_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tbody></table>

### Models with Swin Transformer backbones

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">query</th>
<th valign="bottom">epochs</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/swin/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">45.3, 46.0</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a>, <a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps_2.pth">model</a></td>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/swin/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">49.0,49.6</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a>, <a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps_2.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/swin/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">49.3</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/swin/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">51.8</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/12eps/swin/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">50.6</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Tiny</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">53.2</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_tiny_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
<tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/swin/swin_large_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">51.0</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_large_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>

<tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/swin/drop_path0.5_swin_large_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">53.7</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/drop_path0.5_swin_large_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/12eps/swin/swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">54.5</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-baseline/36eps/swin/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">56.3</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/12eps/swin/swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">12</td>
<td align="center">55.9</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">300</td>
<td align="center">36</td>
<td align="center">57.1</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/12eps/swin/swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">900</td>
<td align="center">12</td>
<td align="center">56.1</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_12eps.pth">model</a></td>
</tr>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks</a></td>
<td align="center">Swin Large</td>
<td align="center">900</td>
<td align="center">36</td>
<td align="center">57.4</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/two_stage/deformable-detr-hybrid-branch/36eps/swin/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh">H-Deformable-DETR + tricks [topk=300]</a></td>
<td align="center">Swin Large</td>
<td align="center">900</td>
<td align="center">36</td>
<td align="center">57.6</td>
<td align="center"><a href="https://github.com/HDETR/H-Deformable-DETR/releases/download/v0.1/drop_path0.5_swin_large_hybrid_branch_lambda1_group6_t1500_n900_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage_36eps.pth">model</a></td>
</tr>
</tbody></table>

## Installation
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
cd ../..
```

## Data

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
coco_path/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
## Run
### To train a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path>
```

To train/eval a model with the swin transformer backbone, you need to download the backbone from the [offical repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models) frist and specify argument`--pretrained_backbone_path` like [our configs](./configs/two_stage/deformable-detr-hybrid-branch/36eps/swin).

### To eval a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path> --eval --resume <checkpoint path>
```

### Distributed Run

You can refer to [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) to enable training on multiple nodes.

## Modified files compared to vanilla Deformable DETR

### To support swin backbones
* models/backbone.py
* models/swin_transformer.py
* mmcv_custom

### To support eval in the training set
* datasets/coco.py
* datasets/\_\_init\_\_.py

### To support Hybrid-branch, tricks and checkpoint
* main.py
* engine.py
* models/deformable_detr.py
* models/deformable_transformer.py

### To support fp16
* models/ops/modules/ms_deform_attn.py
* models/ops/functions/ms_deform_attn_func.py

### To fix a pytorch version bug
* util/misc.py

### Addictional packages needed

* wandb: for logging
* mmdet: for swin backbones
* mmcv: for swin backbones
* timm: for swin backbones


## Citing H-Deformable-DETR
If you find H-Deformable-DETR useful in your research, please consider citing:

```bibtex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}

@article{zhu2020deformable,
  title={Deformable detr: Deformable transformers for end-to-end object detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```
