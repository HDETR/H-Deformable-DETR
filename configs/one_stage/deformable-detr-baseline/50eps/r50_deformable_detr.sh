#!/usr/bin/env bash

set -x

EXP_DIR=exps/one_stage/deformable-detr-baseline/12eps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --num_queries_one2one 300 \
    --num_queries_one2many 0 \
    --k_one2many 0 \
    --epochs 50 \
    --lr_drop 40 \
    ${PY_ARGS}
