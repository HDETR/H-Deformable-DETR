#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-baseline/24eps/r50_n1800_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 1800 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 24 \
    --lr_drop 20 \
    ${PY_ARGS}
