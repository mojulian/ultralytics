# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size

from ultralytics import YOLO

version_name="5"
model_name="yolov"
scale="p"
# model = YOLO(f'ultralytics/cfg/models/v{version_name}/{model_name}.yaml')  # load a pretrained model (recommended for training)
model = YOLO(f'{model_name}{version_name}{scale}.yaml')  # load a pretrained model (recommended for training)
# Train the model
model.train(data='VOC.yaml', epochs=1000, imgsz=352)