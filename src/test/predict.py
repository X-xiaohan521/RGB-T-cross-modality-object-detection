from collections import OrderedDict

import torch
from torchvision import transforms

from main.dataset.DroneVehicleYoloDataset import DroneVehicleYoloDataset
from main.model.MyOBB import MyOBB
from src.main.util.yml2model import *

if __name__ == "__main__":
    # load OBB model
    obb = MyOBB()
    weights: OrderedDict = torch.load("../../weight/yolo11l-obb-weight.pt")
    obb.load_backbone_weights(weights)
    obb.load_neck_weights(weights)

    # prepare dataset
    val_dataset = DroneVehicleYoloDataset("../../data/DroneVehicle-DOTA/val",
                                          ["car", "truck", "feright car", "bus", "van"],
                                          transforms.Compose([transforms.ToTensor()]),
                                          (100, 100, 100, 100))

    with torch.no_grad():
        img, _, _, _ = val_dataset[0]
        img = torch.stack([img])
        print(img.shape)
        logits = obb.forward(img)
        print(logits)
        '''
        {
            'boxes': tensor.shape(batch, 64, 6720)   6720：所有 feature map 展平后的总点数
            'scores': tensor.shape(batch, 80, 6720),   80：类别数
            'feats': [tensor.shape(batch, 256, 64, 80), tensor.shape(batch, 512, 32, 40), tensor.shape(batch, 1024, 16, 20)],
            'angle': tensor.shape(batch, 1, 6720)
        }
        '''

        '''
        [
            [
                [-0.0965, -0.6233,  0.6091,  ...,  0.1285, -0.2167,  0.1492],
                [ 0.4217,  0.7295,  0.5450,  ..., -0.1039, -0.1140, -0.0374],
                [ 0.3195,  0.0107, -0.0334,  ...,  0.3313,  0.1838,  0.0814],
                ...,
                [ 0.0814,  0.9663, -0.1334,  ...,  0.0180,  0.1831,  0.1074],
                [-0.2191, -0.4235, -0.0098,  ..., -0.0795,  0.0696,  0.2596],
                [ 0.6938,  1.2013,  0.3939,  ...,  0.1789, -0.0386,  0.1790]
            ]
        ]
        '''