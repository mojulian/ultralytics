# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size

from ultralytics import YOLO

scale="p"
run_number="19"

model = YOLO(f'runs/detect/train{run_number}/weights/best.pt')  # load a pretrained model (recommended for training)
# Train the model
# model.train(data='VOC.yaml', epochs=1000, imgsz=352)

# Size
get_size(model.model.model)

# Count the number of layers
num_layers = sum(1 for _ in model.model.model.modules()) - 1
print(f"Number of layers: {num_layers}")

# Count the number of parameters
num_params = sum(p.numel() for p in model.model.model.parameters())
print(f"Number of parameters: {num_params}")

model.export(format="onnx",imgsz=[352,352], opset=12)  # export the model to ONNX format