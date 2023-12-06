# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size

from ultralytics import YOLO

model = YOLO('runs/detect/train75/weights/best.pt')  # load a pretrained model (recommended for training)
model.val(data='VOC.yaml', imgsz=256)