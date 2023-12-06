# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size
# from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks
import yaml

# import wandb

## wandb.init()
## run = wandb.init(
#     # Set the project where this run will be logged
#     project="ultralytics-gapyolo",
#     )

from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train93/weights/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov5p.yaml')  # load a pretrained model (recommended for training)
# version_name="8"
# model_name="pizzayolo"
# scale="p"
# version_name="8"
# model_name="tinyissimoyolo"
# scale="p"
# model = YOLO(f'{model_name}{version_name}{scale}.yaml')  # load a pretrained model (recommended for training)
# model = YOLO(f'ultralytics/cfg/models/v{version_name}/{model_name}.yaml')  # load a pretrained model (recommended for training)
# add_wandb_callbacks(model, project =f"gapyolo-{model_name}{scale}-{version_name}")
# model = YOLO('tinyissimoyolo.yaml')  # load a pretrained model (recommended for training)

# # Train the model
# model.train(data='VOC.yaml', epochs=2, imgsz=256)

# Size
get_size(model.model.model)

# Count the number of layers
num_layers = sum(1 for _ in model.model.model.modules()) - 1
print(f"Number of layers: {num_layers}")

# Count the number of parameters
num_params = sum(p.numel() for p in model.model.model.parameters())
print(f"Number of parameters: {num_params}")

model.export(format="onnx",imgsz=[256,256], opset=12)  # export the model to ONNX format