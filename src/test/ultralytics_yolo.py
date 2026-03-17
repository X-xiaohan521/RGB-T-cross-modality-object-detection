from ultralytics import YOLO

yolo = YOLO("../../weight/yolo11l-obb.pt")

print(yolo)