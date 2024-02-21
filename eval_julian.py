# import pdb, 
import os
import torch
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size
from ultralytics.utils.offline_tiling import Tiler

from ultralytics import YOLO


def load_test_images(dir):
    images = {}
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            full_image_name = filename[:-(len(filename.split('_')[-1])+1)]\
                + '.' + filename.split('.')[-1]
            if full_image_name not in images.keys():
                images[full_image_name] = []
            
            images[full_image_name].append(os.path.join(dir, filename))
    # Sort the list for each image
    for image in images.keys():
        images[image] = sorted(images[image])
    return images

def plot_results(stitched_preds, filtered_boxes, filtered_conf, og_image, conf_thresh=0.8):

    filtered_image = og_image.copy()
    for tile_idx in stitched_preds:
        tile = stitched_preds[tile_idx]['tile']
        instances = stitched_preds[tile_idx]['predictions']
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(og_image, (tile['x_min'], tile['y_min']), (tile['x_max'], tile['y_max']), color, 2)

        for instance in instances:
            # if instance['conf'] > conf_thresh:
            x1, y1, x2, y2 = instance['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(og_image, (x1, y1), (x2, y2), color, 2)
    cv2.imshow('stitched image', og_image)
    for box, conf in zip(filtered_boxes, filtered_conf):
        if conf > conf_thresh:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('filtered image', filtered_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


test_images = load_test_images("/home/liam/datasets/CARPK/test_tiles_target_nba_0.04_input_size_256_mode_test/images")
tiles_dict = yaml.safe_load(open("/home/liam/datasets/CARPK/test_tiles_target_nba_0.04_input_size_256_mode_test/tiles_dict.yaml"))
model = YOLO('runs/detect/train782/weights/best.pt')  # load a pretrained model (recommended for training)

tiler = Tiler('/home/liam/ultralytics/tiling_config.yaml')
tiles_dict_path = '/home/liam/datasets/CARPK/test_tiles_target_nba_0.04_input_size_256_mode_test/tiles_dict.yaml'
original_image_dir = '/home/liam/datasets/CARPK/CARPK_test/images'
tiles_dict = yaml.safe_load(open(tiles_dict_path))

# TODO: structure the inference in a way that groups the predictions by original image
for image in test_images:
    og_image = cv2.imread(os.path.join(original_image_dir, image))
    result = model(test_images[image], stream=True)
    # image_name = os.path.basename(image)
    stitched_preds, filtered_boxes, filtered_conf \
          = tiler.stitch_tiled_predictions(result, tiles_dict, image)
    plot_results(stitched_preds, filtered_boxes, filtered_conf, og_image)
