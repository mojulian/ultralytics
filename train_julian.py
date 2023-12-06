# import pdb, 
import os
import torch
import numpy as np
from ultralytics import YOLO

import torch.nn as nn
from ultralytics.nn.tasks import get_size
import yaml

import wandb
wandb.init()
run = wandb.init(
    # Set the project where this run will be logged
    project="gapyolo-tinyissimoyolo",
    )

from ultralytics import YOLO

# Load a model
model = YOLO('tinyissimoyolo.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('tinyissimoyolo.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('tinyissimoyolo.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('pizzayolo.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov5p.yaml')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='VOC.yaml',
            imgsz=256,
            epochs=1500, 
            batch=64,
)

# Size
get_size(model.model.model)

# Count the number of layers
num_layers = sum(1 for _ in model.model.model.modules()) - 1
print(f"Number of layers: {num_layers}")

# Count the number of parameters
num_params = sum(p.numel() for p in model.model.model.parameters())
print(f"Number of parameters: {num_params}")

model.export(format="onnx",imgsz=[256,256], opset=12)  # export the model to ONNX format


# def prune(model, amount=0.3):
#     # Prune model to requested global sparsity
#     import torch.nn.utils.prune as prune
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Conv2d):
#             prune.l1_unstructured(m, name='weight', amount=amount)  # prune
#             prune.remove(m, 'weight')  # make permanent

# print("Script Started")
# def print_size_of_model(model):
#     torch.save(model.half().state_dict(), "temp.pt")
#     print('Size (MB):', os.path.getsize("temp.pt")/1e6)
#     os.remove('temp.pt')

# # download data
# with open("ultralytics/cfg/datasets/VOC.yaml", "r") as f:
#     dataset = yaml.load(f,Loader=yaml.FullLoader)
# exec(dataset['download'])

# device = torch.device("cuda")
# model_name = "yolov5n"
# # model = YOLO('./ultralytics/models/v5/yolov5.yaml')
# model = YOLO(f'./{model_name}.yaml')

# model_parameters = filter(lambda p: p.requires_grad, model.model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])


# # Train
# model.train(data="VOC.yaml", 
#             single_cls=True, 
#             optimizer='Adam', 
#             imgsz=352,
#             cls=0.5, 
#             lr0=1e-3,
#             epochs=500, 
#             batch=64,
#             )

# # Eval
# results = model(imgsz=[352,352], max_det=1, conf=0, source=f"/datasets/mojulian/ultralytics/PascalVoc/{model_name}/images/val",save=True )  # list of Results objects

# # Size
# get_size(model.model.model)

# # Count the number of layers
# num_layers = sum(1 for _ in model.model.model.modules()) - 1
# print(f"Number of layers: {num_layers}")

# # Count the number of parameters
# num_params = sum(p.numel() for p in model.model.model.parameters())
# print(f"Number of parameters: {num_params}")

# model.export(format="onnx",imgsz=[450,45], opset=12)  # export the model to ONNX format

# # pdb.set_trace()
