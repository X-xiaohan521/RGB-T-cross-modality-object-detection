import torch
from ultralytics import YOLO

yolo = YOLO("../../weight/yolo11l-obb.pt")

print(yolo.state_dict())

torch.save(yolo.state_dict(), "../../weight/yolo11l-obb-weight.pt")