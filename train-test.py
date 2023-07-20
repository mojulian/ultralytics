# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size
import yaml


from ultralytics import YOLO

# Load a model
# model = YOLO('runs/detect/train10/weights/last.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov5p.yaml')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='VOC.yaml', epochs=100, imgsz=640, resume=True)