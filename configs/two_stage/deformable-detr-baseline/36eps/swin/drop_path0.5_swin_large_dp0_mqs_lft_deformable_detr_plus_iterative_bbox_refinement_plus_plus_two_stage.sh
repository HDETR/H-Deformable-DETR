#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-baseline/36eps/swin/drop_path0.5_swin_large_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 36 \
    --lr_drop 30 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone swin_large \
    --pretrained_backbone_path /mnt/pretrained_backbone/swin_large_patch4_window7_224_22k.pth \
    --drop_path_rate 0.5 \
    ${PY_ARGS}
