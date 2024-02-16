# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


# Losses
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


# Criterion class for computing Detection training losses
class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=False).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist, offset=None, scale_factor=1.0):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist_reshaped = pred_dist.view(b, a, 4, c // 4)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            pred_dist = pred_dist * scale_factor
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def stitch_tiled_predictions(self, pred_scores, pred_distri, anchor_points, tile_anchors,
                                 stride, tile_stride, tile_info, img_wh, preds=None, debug_plot=True, imgs=None):
        """
        Stitch the predictions from the tiled images back together.

        Args:
            pred_scores (torch.Tensor): The prediction scores from the tiled images.
            pred_distri (torch.Tensor): The prediction distributions from the tiled images.
            tile_info (dict): The information about the tiles.
            img_wh (int): The width and height of the original image.

        Returns:
            stitched_pred_scores (torch.Tensor): The prediction scores from the tiled images stitched back together.
            stitched_pred_distri (torch.Tensor): The prediction distributions from the tiled images stitched back together.
                                                    -> Careful: These are still in the tile coordinate frame!
            pred_bboxes (torch.Tensor): The predicted bounding boxes from the tiled images stitched back together.
        """

        import random
        import time
        start = time.time()
        argmin_time = 0
        tile_anchors_frame_conversion_time = 0
        copying_time = 0
        looping_over_tiles_time = 0
        looping_over_anchors_time = 0
        decoding_time = 0
        loop_over_matched_indices_time = 0
        argmin_index_time = 0
        store_anchor_offsets_time = 0
        compute_distances_time = 0
        batch_size = len(tile_info)
        stitched_pred_scores = torch.zeros(batch_size, anchor_points.shape[0], pred_scores.shape[2], device=pred_scores.device)
        stitched_pred_distri = torch.zeros(batch_size, anchor_points.shape[0], pred_distri.shape[2], device=pred_distri.device)
        if preds is not None:
            stitched_predictions = torch.zeros(batch_size, preds.shape[1], anchor_points.shape[0], device=pred_distri.device)
        anchor_offsets = torch.zeros(batch_size, anchor_points.shape[0], 4, device=pred_scores.device)
        # Convert anchors to global frame
        anchors_global_frame = anchor_points * stride
        tile_anchors *= tile_stride
        network_input_wh = int(tile_anchors[-1, 0] + tile_stride / 2)
        tile_batch_idx = 0
        for batch_idx in range(batch_size):
            if debug_plot:
                # blank_image_ = np.ones((img_wh, img_wh, 3))
                blank_image_ = imgs[batch_idx].permute(1, 2, 0).cpu().numpy()
                blank_image = blank_image_.copy()
                for anchor in anchors_global_frame:
                    cv2.circle(blank_image, (int(anchor[0]), int(anchor[1])), 2, (0, 0, 1), -1)
            
            matched_indices = []
            looping_over_tiles_start = time.time()
            matching_mask = torch.zeros_like(anchors_global_frame[:, 0])

            for tile in tile_info[batch_idx]:
                tile_wh = tile.x_max - tile.x_min
                tile_to_input = tile_wh / network_input_wh
                img_to_tile_ratio = img_wh / tile_wh
                downsample_factor = tile_stride / stride * tile_to_input
                tile_anchors_frame_conversion_start = time.time()
                tile_anchors_global_frame = torch.zeros_like(tile_anchors)
                tile_anchors_global_frame[:,0] = tile_to_input * tile_anchors[:, 0] + tile.x_min
                tile_anchors_global_frame[:,1] = tile_to_input * tile_anchors[:, 1] + tile.y_min
                tile_anchors_frame_conversion_end = time.time()
                tile_anchors_frame_conversion_time += tile_anchors_frame_conversion_end - tile_anchors_frame_conversion_start

                color = (random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255)

                looping_over_anchors_start = time.time()
                for i, tile_anchor in enumerate(tile_anchors_global_frame):
                    # Find closest anchor point in anchors_global_frame
                    start_argmin = time.time()
                    closest_anchor_idx_old = torch.argmin(torch.sum((anchors_global_frame - tile_anchor)**2, dim=1))
                    end_argmin = time.time()
                    argmin_time += end_argmin - start_argmin
                    # Compute distances between tile anchor and global frame anchors
                    compute_distances_start = time.time()
                    distances = torch.sum((anchors_global_frame - tile_anchor)**2, dim=1)
                    compute_distances_end = time.time()
                    compute_distances_time += compute_distances_end - compute_distances_start

                    loop_over_matched_indices_start = time.time()
                    # for idx in matched_indices:
                    #     distances[idx] = float('inf')
                    distances[matching_mask == 1] = float('inf')
                    loop_over_matched_indices_end = time.time()

                    # Set distances to infinity where matching mask is 1
                    
                    loop_over_matched_indices_time += loop_over_matched_indices_end - loop_over_matched_indices_start
                    
                    # Find the closest anchor index
                    start_argmin_index = time.time()
                    closest_anchor_idx = torch.argmin(distances)
                    end_argmin_index = time.time()
                    argmin_index_time += end_argmin_index - start_argmin_index
                    # matched_indices.append(closest_anchor_idx.item())
                    matching_mask[closest_anchor_idx] = 1

                    # Store the distance between the tile anchor and the global frame anchor
                    store_anchor_offsets_start = time.time()
                    distance_xy = anchors_global_frame[closest_anchor_idx] - tile_anchor
                    anchor_offsets[batch_idx, closest_anchor_idx, :2] = distance_xy
                    anchor_offsets[batch_idx, closest_anchor_idx, 2:] = distance_xy
                    store_anchor_offsets_end = time.time()
                    store_anchor_offsets_time += store_anchor_offsets_end - store_anchor_offsets_start
                    
                    
                    # stitched_pred_distri[batch_idx, closest_anchor_idx] = adjusted_pred_distri.view(-1)
                    copying_start = time.time()
                    stitched_pred_distri[batch_idx, closest_anchor_idx] = pred_distri[tile_batch_idx, i, :].clone()
                    stitched_pred_scores[batch_idx, closest_anchor_idx] = pred_scores[tile_batch_idx, i, :].clone()
                    if preds is not None:
                        stitched_predictions[batch_idx, :, closest_anchor_idx] = preds[tile_batch_idx, :, i].clone()
                        x, y, w, h = stitched_predictions[batch_idx, 0, closest_anchor_idx] * tile_to_input,\
                                     stitched_predictions[batch_idx, 1, closest_anchor_idx] * tile_to_input, \
                                     stitched_predictions[batch_idx, 2, closest_anchor_idx] * tile_to_input, \
                                     stitched_predictions[batch_idx, 3, closest_anchor_idx] * tile_to_input

                        x = x + tile.x_min
                        y = y + tile.y_min
                        stitched_predictions[batch_idx, 0, closest_anchor_idx] = int(x)
                        stitched_predictions[batch_idx, 1, closest_anchor_idx] = int(y)
                        stitched_predictions[batch_idx, 2, closest_anchor_idx] = int(w)
                        stitched_predictions[batch_idx, 3, closest_anchor_idx] = int(h)
                    copying_end = time.time()
                    copying_time += copying_end - copying_start

                    if debug_plot:
                        cv2.circle(blank_image,
                                   (int(anchors_global_frame[closest_anchor_idx, 0]), int(anchors_global_frame[closest_anchor_idx, 1])),
                                   4, color, 1)
                        cv2.circle(blank_image,
                                   (int(anchors_global_frame[closest_anchor_idx_old, 0]), int(anchors_global_frame[closest_anchor_idx_old, 1])),
                                   1, (1, 0, 0), 1)
                        cv2.circle(blank_image, (int(tile_anchor[0]), int(tile_anchor[1])), 5, color, 1)
                        if preds is not None:
                            current_pred = stitched_predictions[batch_idx, :, closest_anchor_idx]
                            conf = stitched_predictions[batch_idx, 4:, closest_anchor_idx].amax(0)
                            if conf > 0.01:
                                x, y, w, h = stitched_predictions[batch_idx, 0, closest_anchor_idx],\
                                                stitched_predictions[batch_idx, 1, closest_anchor_idx], \
                                                stitched_predictions[batch_idx, 2, closest_anchor_idx], \
                                                stitched_predictions[batch_idx, 3, closest_anchor_idx]
                                x1 = int(x - w / 2)
                                y1 = int(y - h / 2)
                                x2 = int(x + w / 2)
                                y2 = int(y + h / 2)
                                cv2.rectangle(blank_image, (x1, y2), (x2, y1), color, 2)

                looping_over_tiles_end = time.time()
                looping_over_anchors_time += looping_over_tiles_end - looping_over_anchors_start
                if debug_plot:
                    cv2.rectangle(blank_image, (tile.x_min, tile.y_min), (tile.x_max, tile.y_max), color, 2)
                
                tile_batch_idx += 1
            looping_over_tiles_end = time.time()
            looping_over_tiles_time += looping_over_tiles_end - looping_over_tiles_start

            decoding_start = time.time()
            pred_bboxes = self.bbox_decode(anchor_points, stitched_pred_distri, scale_factor=downsample_factor) - anchor_offsets / stride
            decoding_end = time.time()
            decoding_time += decoding_end - decoding_start
            if debug_plot:                
                # Plot pred_boxes in first image
                for box, class_scores, anchor in zip(pred_bboxes[batch_idx], stitched_pred_scores[batch_idx], anchor_points):
                    # Only plot boxes with confidence > 0.3
                    # convert class_scores to probabilities
                    class_scores = torch.nn.functional.softmax(class_scores, dim=0)
                    if class_scores.max() < 0.3:
                        continue
                    else:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(stride*x1), int(stride*y1), int(stride*x2), int(stride*y2)                        
                        color = (random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255)
                        cv2.rectangle(blank_image, (x1, y2), (x2, y1), (1, 0, 0), 2)

                # Convert blank_image to uint8
                blank_image *= 255
                blank_image = blank_image.astype(np.uint8)
                cv2.imshow('anchor points', blank_image)
                cv2.waitKey(0)
            
        end = time.time()
        full_time = end - start
        # print(f"Stitching time: {full_time}, Argmin time: {argmin_time}")
        # print(f"Percentages: Argmin: {argmin_time/full_time*100}% Copying: {copying_time/full_time*100}% \
        #         Tile anchors: {tile_anchors_frame_conversion_time/full_time*100}% \
        #         Looping over tiles: {looping_over_tiles_time/full_time*100}% \
        #         Looping over anchors: {looping_over_anchors_time/full_time*100}%\
        #         Decoding: {decoding_time/full_time*100}% \
        #         Loop over matched indices: {loop_over_matched_indices_time/full_time*100}% \
        #         Argmin index: {argmin_index_time/full_time*100}% \
        #         Store anchor offsets: {store_anchor_offsets_time/full_time*100}%\
        #         Compute distances: {compute_distances_time/full_time*100}% ")
        if preds is not None:
            return stitched_pred_scores, stitched_pred_distri, pred_bboxes, stitched_predictions
        else:
            return stitched_pred_scores, stitched_pred_distri, pred_bboxes

    def __call__(self, preds, batch, boxes_anchors_stride=None, plot=False):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # if isinstance(preds, list) and len(preds) > 1:
        #     feats = preds[1]
        # elif isinstance(preds, tuple):
        #     feats = preds[1]
        # else:
        #     feats = preds

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        if boxes_anchors_stride is not None:
            pred_bboxes, anchor_points, stride_tensor = boxes_anchors_stride

        # if batch['tile_info'] is not None:
        #     tile_anchors, tile_strides = make_anchors(feats, self.stride, 0.5, wh_multiplier=1)
        #     tile_info = batch['tile_info']
        #     img_wh = batch['img'].shape[2]
        #     _, _, _, network_output_wh = feats[0].shape
        #     network_input_wh = int(network_output_wh * self.stride[0])
        #     tile_scale = img_wh / network_input_wh
        #     anchor_points, stride_tensor = make_anchors(feats, self.stride*tile_scale, 0.5, wh_multiplier=8)
        #     stride = stride_tensor[0, 0]
        #     tile_stride = tile_strides[0, 0]
        #     #TODO: This might actually be the best place to stitch the predictions back together
        #     #      The output should be with the actual batchsize and have the fake number of anchor points
        #     #      Furthermore, the pred_scores should undergo the same transformation as the pred_distri
            
        #     #TODO: Also stitch the inference predictions back together in case of validation where preds is a tuple
        #     if isinstance(preds, tuple):
        #         predictions = preds[0]
        #         pred_scores, pred_distri, pred_bboxes, stitched_predictions\
        #               = self.stitch_tiled_predictions(pred_scores, pred_distri, anchor_points,
        #                                               tile_anchors, stride, tile_stride,
        #                                               tile_info, img_wh, imgs=batch['img'], preds=predictions)
        #         concatenated = torch.cat((pred_distri, pred_scores), dim=2)
        #         stitched_feats = torch.reshape(concatenated, (concatenated.shape[0], concatenated.shape[2], int(np.sqrt(concatenated.shape[1])), -1))
        #     else:
        #         pred_scores, pred_distri, pred_bboxes = self.stitch_tiled_predictions(pred_scores, pred_distri, anchor_points,
        #                                                              tile_anchors, stride, tile_stride,
        #                                                              tile_info, img_wh, imgs=batch['img'])    

        else:
            anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
            # pboxes
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # if plot and tile_info is not None:
        #     import random
        #     for full_img_idx in range(len(batch['img'])):
        #         full_img = batch['img'][full_img_idx]
        #         full_img_ = full_img.permute(1, 2, 0).cpu().numpy()
        #         full_img_np = full_img_.copy()
        #         height, width = full_img_np.shape[0], full_img_np.shape[1]
        #         for i in range(len(tile_info[full_img_idx])):
        #             if full_img_idx > 0:
        #                 offset = full_img_idx * len(tile_info[full_img_idx - 1])
        #             else:
        #                 offset = 0
        #             img = batch['tiled_img'][i + offset]                
                    
        #             img = img.permute(1, 2, 0).cpu().numpy()
        #             img_copy = img.copy()

        #             current_tile = tile_info[full_img_idx][i]
        #             tile_x_min = current_tile.x_min
        #             tile_y_min = current_tile.y_min
        #             tile_wh = current_tile.x_max - current_tile.x_min
        #             downsaple_factor = tile_wh / img.shape[0]
        #             stride = int(self.stride[0].cpu().numpy())

        #             # Plot pred_boxes in first image
        #             for box, class_scores, anchor in zip(pred_bboxes[i + offset], pred_scores[i + offset], anchor_points):
        #                 # Only plot boxes with confidence > 0.3
        #                 # convert class_scores to probabilities
        #                 class_scores = torch.nn.functional.softmax(class_scores)
        #                 if class_scores.max() < 0.3:
        #                     continue
        #                 else:
        #                     anchor_x, anchor_y = int(anchor[0]*stride), int(anchor[1]*stride)
        #                     x1, y1, x2, y2 = box
        #                     x1, y1, x2, y2 = int(stride*x1), int(stride*y1), int(stride*x2), int(stride*y2)
        #                     # set values to zero if they are negative
        #                     x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        #                     # if values are greater than 255, set them to 255
        #                     x1, y1, x2, y2 = min(width-1, x1), min(height-1, y1), min(width-1, x2), min(height-1, y2)
                            
        #                     x1_full, y1_full, x2_full, y2_full = int(downsaple_factor * x1 + tile_x_min),\
        #                                                          int(downsaple_factor * y1 + tile_y_min),\
        #                                                          int(downsaple_factor * x2 + tile_x_min),\
        #                                                          int(downsaple_factor * y2 + tile_y_min)
        #                     color = (random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255)
        #                     cv2.rectangle(img_copy, (x1, y2), (x2, y1), color, 2)
        #                     cv2.circle(img_copy, (anchor_x, anchor_y), 5, color, -1)
        #                     cv2.rectangle(full_img_np, (x1_full, y2_full), (x2_full, y1_full), color, 2)
                    
        #             cv2.imshow("First Image", full_img_np)
        #             cv2.imshow("Tiled Image", img_copy)
        #             cv2.waitKey(0)
        # anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5, wh_multiplier=2.0)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # if preds is not None and 'val' in batch:
        #     return loss.sum() * batch_size, loss.detach(), (stitched_predictions, stitched_feats)
        # else:
        #     return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
