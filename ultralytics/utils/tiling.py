import torch
import cv2
import random
import time

import numpy as np
import torch.nn as nn

from dataclasses import dataclass

from ultralytics.utils.tal import dist2bbox, make_anchors

@dataclass
class Tile:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


class Tiler:

    def __init__(self, batch, stride, use_dfl=False):
        self.use_dfl = use_dfl
        self.stride = stride
        self.batch = batch
        self.device = batch['img'].device
        self.reg_max = 16  # Hardcoded!!!!
        self.no = 74  # Hardcoded!!!!
        self.nc = 10 # Hardcoded!!!!
        if self.reg_max > 0:
            self.use_dfl = True
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def remove_image_padding(self, image):
            """
            Remove padding from image.
            
            Args:
                image (torch.Tensor): The image tensor to remove padding from.
                
            Returns:
                naked_image (torch.Tensor): The image tensor with padding removed.
            """
            print(f'image shape before: {image.shape}')
            
            _, _, top_left, top_right, bottom_left = self.get_shape_of_image_without_padding(image)

            naked_image = image[top_left[0]:bottom_left[0], top_left[1]:top_right[1]]

            print(f'image shape after: {naked_image.shape}')
            return naked_image
        
    def get_shape_of_image_without_padding(self, image):
        """
        Get the shape of an image without padding.
        
        Args:
            image (torch.Tensor): The image tensor to get the shape of.
            
        Returns:
            shape (tuple): The shape of the image.
        """
        # The padding pixels all have the same color value, so we can just check
        # the first pixel and remove all pixels with that value
        first_pixel = image[0, 0, 0]
        
        # Find the indices of the top left corner where the padding starts
        width = image.shape[2]
        height = image.shape[1]
        top_left = None

        for i in range(height):
            for j in range(width):
                if image[i, j, 0] != first_pixel and image[i+1, j+1, 0] != first_pixel:
                    top_left = (i, j)
                    break
            if top_left is not None:
                break

        # With the assumption of symmetric padding, we can calculate the other corners
        top_right = (top_left[0], width - top_left[1])
        bottom_left = (height - top_left[0], top_left[1])
        # get the shape of the image without padding
        w, h = (bottom_left[0] - top_left[0], top_right[1] - top_left[1])
        return w, h, top_left, top_right, bottom_left

    def get_avg_bbox_area(self, bboxes):
        """
        Get the average area of all the bounding boxes in a list of bounding boxes.
        
        Args:
            bboxes (tensor): The tensor of bounding boxes.

        Returns:
            avg_bbox_area (float): The average area of the bounding boxes.
        """
        bbox_areas = []
        for bbox in bboxes:
            bbox_areas.append(bbox[2] * bbox[3])
        avg_bbox_area = sum(bbox_areas) / len(bbox_areas)
        return avg_bbox_area

    def get_tile_wh(self, full_image_wh, target_nba, avg_nba):
        """
        Get the tile width and height.
            
        Args:
            full_image_wh (int): The width/height of the full augmented image.
            tagret_nba (float): The target normalized bbox area.
            avg_nba (float): The average normalized bbox area.

        Returns:
            tile_wh (int): The tile width/height.
        """
        scale_factor = torch.sqrt(target_nba / avg_nba)
        tile_wh = torch.ceil(full_image_wh / scale_factor).to(torch.int64)
        return tile_wh

    def get_list_of_tiles(self, og_image, avg_bbox_area, target_nba=0.01,
                            debug_plot=False):
        """
        Return a list of tile objects that will be used later to extract tile
        from the original image.

        Args:
            og_image (torch.Tensor): The original image with padding.
            avg_bbox_area (float): The average area of all the bounding boxes in the current
                                    images normalized by the image resolution.
            target_nba (float): The target normalized bbox area.
            debug_plot (bool): A flag to enable plot for debugging.

        Returns:
            tiles (list): A list of Tile objects.
        """
        square_image_wh = max(og_image.shape[1], og_image.shape[2])
        optimal_tile_overlap = 2 * torch.sqrt(avg_bbox_area*square_image_wh**2)
        tile_wh = self.get_tile_wh(square_image_wh, target_nba, avg_bbox_area)
        w, h, top_left, _, _ = self.get_shape_of_image_without_padding(og_image)

        num_tiles_w = torch.ceil(w / tile_wh).to(torch.int64)
        if num_tiles_w == 1:
            centers_x = torch.tensor([w / 2])
        else:
            optimal_overlap_w = False
            while not optimal_overlap_w:
                centers_x = torch.linspace(tile_wh / 2, w - tile_wh / 2, steps=num_tiles_w)
                x_spacing = centers_x[1] - centers_x[0]
                overlap = tile_wh - x_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_w += 1
                else:
                    optimal_overlap_w = True

        num_tiles_h = torch.ceil(h / tile_wh).to(torch.int64)
        if num_tiles_h == 1:
            centers_y = torch.tensor([h / 2])
        else:
            optimal_overlap_h = False
            while not optimal_overlap_h:
                centers_y = torch.linspace(tile_wh / 2, h - tile_wh / 2, steps=num_tiles_h)
                y_spacing = centers_y[1] - centers_y[0]
                overlap = tile_wh - y_spacing
                if overlap < optimal_tile_overlap:
                    num_tiles_h += 1
                else:
                    optimal_overlap_h = True

        # Adjust for padding
        centers_x += top_left[0]
        centers_y += top_left[1]

        tiles = []

        for x in centers_x:
            for y in centers_y:
                x_min = int(x - tile_wh / 2)
                x_max = int(x_min + tile_wh)
                y_min = int(y - tile_wh / 2)
                y_max = int(y_min + tile_wh)
                tile = Tile(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                tiles.append(tile)

        if debug_plot:
            fake_image = og_image.permute(1, 2, 0).cpu().numpy().copy()
            for tile in tiles:
                # create a tuple color for the rectangle where the values for the 
                # 3 channels are chosen randomly
                color = [random.randint(0, 255)/ 255, random.randint(0, 255)/ 255,
                            random.randint(0, 255)/ 255]
                cv2.rectangle(fake_image, (tile.x_min, tile.y_min),
                                (tile.x_max, tile.y_max), color, 2)
                cv2.imshow('fake_image', fake_image)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

        return tiles

    def extract_tile_from_image(self, image, tile, model_input_size=256):
        """
        Extract a tile from the image and resize it to the desired size.

        Args:
            image (torch.Tensor): The image tensor to extract the tile from.
            tile (Tile): The tile object.
            model_input_size (int): The size of the input to the model.

        Returns:
            tile (torch.Tensor): The extracted tile.
        """
        image_tile = image[:, tile.y_min:tile.y_max, tile.x_min:tile.x_max]

        # Resize the tile to the desired size
        resized_tile = nn.functional.interpolate(image_tile.unsqueeze(0),
                                                    size=(model_input_size,model_input_size),
                                                    mode='bilinear', align_corners=False)
        return resized_tile.squeeze(0)
    
    def bbox_decode(self, anchor_points, pred_dist, scale_factor=1.0):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            pred_dist = pred_dist * scale_factor
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def split_image_batch_into_tiles(self, debug_plot=False):
        """
        Creates tiled images and labels according to the specifications in the input dictionary.
        
        Args:
            debug_plot (bool): A flag to enable plot for debugging.

        Returns:
            tiled_batch (torch.Tensor): The batch of tiled images.
            tiles_dict (dict): A dictionary containing the information about the tiles.
        """

        tiled_batch = torch.empty(0).to(self.batch['img'].device)
        tiles_dict = {}
        bs = self.batch['img'].shape[0]
        for idx in range(bs):
            og_img = self.batch['img'][idx]
            # Get all bboxes in batch['bboxes'] where batch['batch_idx'] == idx
            bboxes = []
            for bbox_idx, batch_idx in enumerate(self.batch['batch_idx']):
                if batch_idx == idx:
                    bboxes.append(self.batch['bboxes'][bbox_idx])
                elif batch_idx > idx:
                    break

            avg_bbox_area = self.get_avg_bbox_area(bboxes)
            tiles_dict[idx] = self.get_list_of_tiles(og_img, avg_bbox_area)

            for tile in tiles_dict[idx]:
                tile_img = self.extract_tile_from_image(og_img, tile)
                tiled_batch = torch.cat((tiled_batch, tile_img.unsqueeze(0)))

            if debug_plot:
                og_image_np_ = og_img.permute(1, 2, 0).cpu().numpy()
                og_image_np = og_image_np_.copy()
                for j in range(len(tiles_dict[idx ])):
                    if idx > 0:
                        offset = idx * len(tiles_dict[idx - 1])
                    else:
                        offset = 0

                    tiled_image_ = tiled_batch[j + offset].permute(1, 2, 0).cpu().numpy()
                    tiled_image = tiled_image_.copy()
                    cv2.imshow('tiled_image', tiled_image)

                    current_tile = tiles_dict[idx][j]
                    cv2.rectangle(og_image_np, (current_tile.x_min, current_tile.y_min),
                                    (current_tile.x_max, current_tile.y_max), (0, 255, 0), 2)
                    cv2.imshow('og_image', og_image_np)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()

        return tiled_batch, tiles_dict
    
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

            matching_mask = torch.zeros_like(anchors_global_frame[:, 0])

            for tile in tile_info[batch_idx]:
                tile_wh = tile.x_max - tile.x_min
                tile_to_input = tile_wh / network_input_wh
                downsample_factor = tile_stride / stride * tile_to_input
                tile_anchors_global_frame = torch.zeros_like(tile_anchors)
                tile_anchors_global_frame[:,0] = tile_to_input * tile_anchors[:, 0] + tile.x_min
                tile_anchors_global_frame[:,1] = tile_to_input * tile_anchors[:, 1] + tile.y_min

                color = (random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255)

                for i, tile_anchor in enumerate(tile_anchors_global_frame):
                    # Find closest anchor point in anchors_global_frame
                    closest_anchor_idx_old = torch.argmin(torch.sum((anchors_global_frame - tile_anchor)**2, dim=1))

                    # Compute distances between tile anchor and global frame anchors
                    distances = torch.sum((anchors_global_frame - tile_anchor)**2, dim=1)

                    # for idx in matched_indices:
                    #     distances[idx] = float('inf')
                    distances[matching_mask == 1] = float('inf')
                    
                    # Find the closest anchor index
                    closest_anchor_idx = torch.argmin(distances)
                    # matched_indices.append(closest_anchor_idx.item())
                    matching_mask[closest_anchor_idx] = 1

                    # Store the distance between the tile anchor and the global frame anchor
                    distance_xy = anchors_global_frame[closest_anchor_idx] - tile_anchor
                    anchor_offsets[batch_idx, closest_anchor_idx, :2] = distance_xy
                    anchor_offsets[batch_idx, closest_anchor_idx, 2:] = distance_xy
                    
                    
                    # stitched_pred_distri[batch_idx, closest_anchor_idx] = adjusted_pred_distri.view(-1)
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

                    if debug_plot:
                        cv2.circle(blank_image,
                                   (int(anchors_global_frame[closest_anchor_idx, 0]), int(anchors_global_frame[closest_anchor_idx, 1])),
                                   4, color, 1)
                        cv2.circle(blank_image,
                                   (int(anchors_global_frame[closest_anchor_idx_old, 0]), int(anchors_global_frame[closest_anchor_idx_old, 1])),
                                   1, (1, 0, 0), 1)
                        cv2.circle(blank_image, (int(tile_anchor[0]), int(tile_anchor[1])), 5, color, 1)
                        if preds is not None:
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

                if debug_plot:
                    cv2.rectangle(blank_image, (tile.x_min, tile.y_min), (tile.x_max, tile.y_max), color, 2)
                
                tile_batch_idx += 1

            pred_bboxes = self.bbox_decode(anchor_points, stitched_pred_distri, scale_factor=downsample_factor) - anchor_offsets / stride
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
            
        if preds is not None:
            return stitched_pred_scores, stitched_pred_distri, pred_bboxes, stitched_predictions
        else:
            return stitched_pred_scores, stitched_pred_distri, pred_bboxes
        
    def get_stitched_batch(self, preds):
        """
        Get a batch of images that has the original image shape again by stitching the tiles back together.

        Args:
            preds (list or tuple): The predictions from the tiled images. If the network is in validation mode, preds will
                                   be a tuple or list of shape [[bs, #classes + 4, #anchors], [bs, #classes + #channels, #anchors_x, #anchors_y]].
                                   If the model is in training mode, preds will be a list of shape [bs, #classes + #channels, #anchors_x, #anchors_y].

        Returns:
            stitched_preds (torch.Tensor): The batch of predictions stitched back together.
            boxes_anchors_stride (tuple): A tuple containing the predicted bounding boxes, the anchor points, and the stride tensor.
        """
        # feats = preds[1] if isinstance(preds, tuple) else preds
        preds_is_multiple = False
        if isinstance(preds, list) and len(preds) > 1:
            feats = preds[1]
            preds_is_multiple = True
        elif isinstance(preds, tuple):
            feats = preds[1]
            preds_is_multiple = True
        else:
            feats = preds

        no = feats[0].shape[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        tile_anchors, tile_strides = make_anchors(feats, self.stride, 0.5, wh_multiplier=1)
        tile_info = self.batch['tile_info']
        img_wh = self.batch['img'].shape[2]
        _, _, _, network_output_wh = feats[0].shape
        network_input_wh = int(network_output_wh * self.stride[0])
        tile_scale = img_wh / network_input_wh
        anchor_points, stride_tensor = make_anchors(feats, self.stride*tile_scale, 0.5, wh_multiplier=8)
        stride = stride_tensor[0, 0]
        tile_stride = tile_strides[0, 0]
        
        if preds_is_multiple:
            predictions = preds[0]
            pred_scores, pred_distri, pred_bboxes, stitched_predictions\
                    = self.stitch_tiled_predictions(pred_scores, pred_distri, anchor_points,
                                                    tile_anchors, stride, tile_stride,
                                                    tile_info, img_wh, imgs=self.batch['img'], preds=predictions)
            concatenated = torch.cat((pred_distri, pred_scores), dim=2)
            stitched_feats = torch.reshape(concatenated, (concatenated.shape[0], concatenated.shape[2], int(np.sqrt(concatenated.shape[1])), -1))

            return (stitched_predictions, [stitched_feats]), (pred_bboxes, anchor_points, stride_tensor)
        else:
            pred_scores, pred_distri, pred_bboxes = self.stitch_tiled_predictions(pred_scores, pred_distri, anchor_points,
                                                                    tile_anchors, stride, tile_stride,
                                                                    tile_info, img_wh, imgs=self.batch['img'])
            concatenated = torch.cat((pred_distri, pred_scores), dim=2)
            stitched_feats = torch.reshape(concatenated, (concatenated.shape[0], concatenated.shape[2], int(np.sqrt(concatenated.shape[1])), -1))

            return [stitched_feats], (pred_bboxes, anchor_points, stride_tensor)
        