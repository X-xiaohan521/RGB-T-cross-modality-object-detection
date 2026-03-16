import torch
from torchvision import transforms

from main.dataset.DroneVehicleYoloDataset import DroneVehicleYoloDataset
from src.main.util.yml2model import *
from src.main.model.YoloOBB import OBBModel

if __name__ == "__main__":
    # load OBB model
    yml_dict = load_yml("../../config/yolo11-obb.yaml")
    model, sort = parse_model(yml_dict, input_channels=3)   # model with backbone and head
    obb = OBBModel(model)
    weight = torch.load("../../weight/yolo11n-obb-weight.pt")
    obb.load(weight)

    # prepare dataset
    val_dataset = DroneVehicleYoloDataset("../../data/DroneVehicle-DOTA/val",
                                          ["car", "truck", "feright car", "bus", "van"],
                                          transforms.Compose([transforms.ToTensor()]),
                                          (100, 100, 100, 100))

    with torch.no_grad():
        img, _, _, _ = val_dataset[0]
        print(img.shape)
        logits = obb.forward(img)
        print(logits)