# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy
from util import box_ops
import wandb
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import time

import datasets.transforms as T
from bounding_box import BoxList
from boxlist_ops import cat_boxlist
import numpy as np

scaler = torch.cuda.amp.GradScaler()


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    multi_targets = copy.deepcopy(targets)
    # repeat the targets
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]

    # one-to-one first
    (loss_dict, matching_time, assign_time, loss_time,) = criterion(
        outputs=outputs,
        targets=targets,
        outputs_one2many=outputs_one2many,
        multi_targets=multi_targets,
        k_one2many=k_one2many,
    )
    return (
        loss_dict,
        matching_time,
        assign_time,
        loss_time,
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    k_one2many=1,
    lambda_one2many=1.0,
    use_wandb=False,
    use_fp16=False,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    time_for_matching = 0.0
    time_for_assign = 0.0
    time_for_loss = 0.0

    start_time = time.time()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        with torch.cuda.amp.autocast() if use_fp16 else torch.cuda.amp.autocast(
            enabled=False
        ):
            if use_fp16:
                optimizer.zero_grad()
            outputs = model(samples)

            if k_one2many > 0:
                loss_dict, matching_time, assign_time, loss_time = train_hybrid(
                    outputs, targets, k_one2many, criterion, lambda_one2many
                )
            else:
                loss_dict, matching_time, assign_time, loss_time = criterion(
                    outputs, targets, k_one2many=0
                )
            time_for_matching += matching_time
            time_for_assign += assign_time
            time_for_loss += loss_time
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if use_fp16:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            optimizer.zero_grad()
            losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            )
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters())

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

        if use_wandb:
            try:
                wandb.log(loss_dict)
            except:
                pass
    end_time = time.time()
    total_time_cost = end_time - start_time
    print("total time cost for an epoch is:", total_time_cost)
    print("time for matching part for an epoch is:", time_for_matching)
    print("time for linear assign part for an epoch is:", time_for_assign)
    print("time for loss part for an epoch is:", time_for_loss)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    use_wandb=False,
):
    # disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict, eval_matching_time, eval_assign_time, eval_loss_time = criterion(
            outputs, targets, k_one2many=0
        )
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]
    if use_wandb:
        try:
            wandb.log({"AP": stats["coco_eval_bbox"][0]})
            wandb.log(stats)
        except:
            pass

    # recover the model parameters for next training epoch
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    return stats, coco_evaluator


def im_detect_bbox(
    model,
    target_scale=800,
    target_max_size=1333,
    before_transform_targets=None,
    before_transform_samples=None,
    topk=100,
):
    """
    Performs bbox detection on the original image.
    """

    transform = T.Compose(
        [
            T.RandomResize([target_scale], max_size=target_max_size),
            T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ]
    )
    new_samples = []
    for sample, target in zip(before_transform_samples, before_transform_targets):
        new_samples.append(transform(sample, target)[0])

    new_samples = utils.nested_tensor_from_tensor_list(new_samples).to("cuda")

    outputs = model(new_samples)

    image_size_list = []
    image_size_list_tensor = []
    tensors = new_samples.tensors
    masks = new_samples.mask
    for batch in range(tensors.shape[0]):
        cur_masks = masks[batch]
        # [w,h]
        w = sum(cur_masks[0] == False)
        h = sum(cur_masks[:, 0] == False)
        image_size_list.append((w, h))
        image_size_list_tensor.append(torch.tensor([h, w]))
    image_size_list_tensor = torch.stack(image_size_list_tensor).to(tensors.device)

    boxlists_i = []
    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), topk, dim=1
    )
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_size_list_tensor.unbind(1)

    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]
    results = [
        {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)
    ]

    for batch, result in enumerate(results):
        new_BoxList = BoxList(
            bbox=result["boxes"], image_size=image_size_list[batch], mode="xyxy"
        )
        new_BoxList.add_field("scores", result["scores"])
        new_BoxList.add_field("labels", result["labels"])
        boxlists_i.append(new_BoxList)

    return boxlists_i


def im_detect_bbox_hflip(
    model,
    target_scale=800,
    target_max_size=1333,
    before_transform_targets=None,
    before_transform_samples=None,
    topk=100,
):
    """
    Performs bbox detection on the original image.
    """

    transform = T.Compose(
        [
            T.RandomResize([target_scale], max_size=target_max_size),
            T.Compose(
                [
                    T.RandomHorizontalFlip(1.0),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ]
    )
    new_samples = []
    for sample, target in zip(before_transform_samples, before_transform_targets):
        new_samples.append(transform(sample, target)[0])

    new_samples = utils.nested_tensor_from_tensor_list(new_samples).to("cuda")

    outputs = model(new_samples)

    image_size_list = []
    image_size_list_tensor = []
    tensors = new_samples.tensors
    masks = new_samples.mask
    for batch in range(tensors.shape[0]):
        cur_masks = masks[batch]
        # [w,h]
        w = sum(cur_masks[0] == False)
        h = sum(cur_masks[:, 0] == False)
        image_size_list.append((w, h))
        image_size_list_tensor.append(torch.tensor([h, w]))
    image_size_list_tensor = torch.stack(image_size_list_tensor).to(tensors.device)

    boxlists_i = []
    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), topk, dim=1
    )
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = image_size_list_tensor.unbind(1)

    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]
    results = [
        {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)
    ]

    for batch, result in enumerate(results):
        new_BoxList = BoxList(
            bbox=result["boxes"], image_size=image_size_list[batch], mode="xyxy"
        )
        new_BoxList.add_field("scores", result["scores"])
        new_BoxList.add_field("labels", result["labels"])
        boxlists_i.append(new_BoxList)
    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists_i]
    return boxlists_inv


@torch.no_grad()
def remove_boxes(boxlist_ts, min_scale, max_scale):
    new_boxlist_ts = []
    for _, boxlist_t in enumerate(boxlist_ts):
        mode = boxlist_t.mode
        boxlist_t = boxlist_t.convert("xyxy")
        boxes = boxlist_t.bbox
        keep = []
        for j, box in enumerate(boxes):
            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            if (w * h > min_scale * min_scale) and (w * h < max_scale * max_scale):
                keep.append(j)
        new_boxlist_ts.append(boxlist_t[keep].convert(mode))
    return new_boxlist_ts


@torch.no_grad()
def merge_result_from_multi_scales(boxlists, nms_type="nms", vote_thresh=0.65):
    print("vote threshold is:", vote_thresh)
    num_images = len(boxlists)
    results = []
    for i in range(num_images):
        scores = boxlists[i].get_field("scores")
        labels = boxlists[i].get_field("labels")
        boxes = boxlists[i].bbox
        boxlist = boxlists[i]
        result = []
        # skip the background
        for j in range(0, 91):
            inds = (labels == j).nonzero().view(-1)
            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            # 0.6 is not used in soft-vote
            boxlist_for_class = boxlist_nms(
                boxlist_for_class,
                0.6,
                score_field="scores",
                nms_type=nms_type,
                vote_thresh=vote_thresh,
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels",
                torch.full((num_labels,), j, dtype=torch.int64, device=scores.device),
            )
            result.append(boxlist_for_class)
        result = cat_boxlist(result)

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > 1000:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - 1000 + 1,
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)
    return results


@torch.no_grad()
def boxlist_nms(
    boxlist,
    nms_thresh,
    max_proposals=-1,
    score_field="scores",
    nms_type="nms",
    vote_thresh=0.65,
):
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    if nms_type == "nms":
        keep = _box_nms(boxes, score, nms_thresh)
        if max_proposals > 0:
            keep = keep[:max_proposals]
        boxlist = boxlist[keep]
    else:
        if nms_type == "vote":
            boxes_vote, scores_vote = bbox_vote(boxes, score, vote_thresh)
        else:
            boxes_vote, scores_vote = soft_bbox_vote(boxes, score, vote_thresh)
        if len(boxes_vote) > 0:
            boxlist.bbox = boxes_vote
            boxlist.extra_fields[score_field] = scores_vote

    return boxlist.convert(mode)


@torch.no_grad()
def bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:]
            )
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores


@torch.no_grad()
def soft_bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= 0.05)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
                det_accu[:, -1:]
            )
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores


@torch.no_grad()
def im_detect_bbox_aug_vote(
    model,
    samples,
    targets,
    scales,
    scale_ranges,
    max_size,
    before_transform_targets,
    before_transform_samples,
    topk,
    mstest_with_h_flip,
    vote_threshold=0.66,
):
    # add code from ATSS here
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(samples.tensors.shape[0]):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # print("identity transform")
                # The first one is identity transform, no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    boxlists_i = im_detect_bbox(
        model,
        before_transform_samples=before_transform_samples,
        before_transform_targets=before_transform_targets,
        topk=topk,
    )
    add_preds_t(boxlists_i)

    if mstest_with_h_flip:
        print("with h-flip!")
        boxlists_hf = im_detect_bbox_hflip(
            model,
            before_transform_samples=before_transform_samples,
            before_transform_targets=before_transform_targets,
            topk=topk,
        )
        add_preds_t(boxlists_hf)

    for idx, scale in enumerate(scales):
        min_range = scale_ranges[idx][0]
        max_range = scale_ranges[idx][1]
        if scale < 800:
            max_size = 1333

        boxlists_scl = im_detect_bbox(
            model,
            target_scale=scale,
            target_max_size=max_size,
            before_transform_samples=before_transform_samples,
            before_transform_targets=before_transform_targets,
            topk=topk,
        )
        boxlists_scl = remove_boxes(boxlists_scl, min_range, max_range)
        add_preds_t(boxlists_scl)
        if mstest_with_h_flip:
            boxlists_scl_hf = im_detect_bbox_hflip(
                model,
                target_scale=scale,
                target_max_size=max_size,
                before_transform_samples=before_transform_samples,
                before_transform_targets=before_transform_targets,
                topk=topk,
            )
            boxlists_scl_hf = remove_boxes(boxlists_scl_hf, min_range, max_range)
            add_preds_t(boxlists_scl_hf)
    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for _, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat([boxlist_t.get_field("scores") for boxlist_t in boxlist_ts])
        labels = torch.cat([boxlist_t.get_field("labels") for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field("scores", scores)
        boxlist.add_field("labels", labels)
        boxlists.append(boxlist)

    results = merge_result_from_multi_scales(boxlists, "soft-vote", vote_threshold)
    # results = boxlists
    for idx in range(len(results)):
        img_w, img_h = results[idx].size
        cur_boxes = results[idx].bbox
        scale_fct = torch.stack([img_w, img_h, img_w, img_h]).to(cur_boxes.device)
        cur_boxes = cur_boxes / scale_fct[None, :]
        img_h, img_w = targets[idx]["orig_size"]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h]).to(cur_boxes.device)
        cur_boxes = cur_boxes * scale_fct[None, :]
        results[idx].bbox = cur_boxes

    return results


@torch.no_grad()
def evaluate_mstest(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    use_wandb=False,
    scales=None,
    scale_ranges=None,
    max_size=None,
    topk=100,
    mstest_with_h_flip=False,
    vote_threshold=0.66,
):
    # disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    batch_count = 0
    for (
        samples,
        targets,
        before_transform_samples,
        before_transform_targets,
    ) in data_loader:
        samples = samples.to(device)

        # type(samples):<class 'util.misc.NestedTensor'>
        # samples.tensors.shape:torch.Size([2, 3, 873, 1201])
        # samples.mask.shape:torch.Size([2, 873, 1201])
        # outputs = model(samples)
        results = im_detect_bbox_aug_vote(
            model,
            samples,
            targets,
            scales,
            scale_ranges,
            max_size,
            before_transform_targets=before_transform_targets,
            before_transform_samples=before_transform_samples,
            topk=topk,
            mstest_with_h_flip=mstest_with_h_flip,
            vote_threshold=vote_threshold,
        )
        new_results = []
        for result in results:
            dict_form_results = dict()
            dict_form_results["scores"] = result.get_field("scores")
            dict_form_results["labels"] = result.get_field("labels")
            dict_form_results["boxes"] = result.bbox
            new_results.append(dict_form_results)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, new_results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        batch_count += 1
        print("Go through batch ", batch_count)

    # gather the stats from all processes
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # recover the model parameters for next training epoch
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    return stats, coco_evaluator
