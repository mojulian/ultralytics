# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size
from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks
import yaml

import wandb

# wandb.init()
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="ultralytics-gapyolo",
#     )

from ultralytics import YOLO

# Load a model
# model = YOLO('runs/detect/train10/weights/last.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov5p.yaml')  # load a pretrained model (recommended for training)
version_name="5"
model_name="yolov"
scale="p"
# model = YOLO(f'ultralytics/cfg/models/v{version_name}/{model_name}.yaml')  # load a pretrained model (recommended for training)
model = YOLO(f'{model_name}{version_name}{scale}.yaml')  # load a pretrained model (recommended for training)
add_wandb_callbacks(model, project =f"gapyolo-{model_name}{scale}-{version_name}")

# Train the model
model.train(data='VOC.yaml', epochs=1000, imgsz=352)