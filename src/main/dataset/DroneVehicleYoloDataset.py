import os

from PIL import Image
from torch.utils.data import Dataset
from src.main.util import xml_to_dict, extract_points


class DroneVehicleYoloDataset(Dataset):
    def __init__(self, data_path: str, classes: list, transform=None):
        super().__init__()
        self.data_path = data_path
        self.classes = classes
        self.transform = transform

        prefix = data_path[data_path.rfind('/')+1:]
        self.img_path = os.path.join(data_path, prefix + "img")
        self.imgr_path = os.path.join(data_path, prefix + "imgr")
        self.label_path = os.path.join(data_path, prefix + "label")
        self.labelr_path = os.path.join(data_path, prefix + "labelr")

        self.img_names = os.listdir(self.img_path)
        self.data_length = len(self.img_names)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = Image.open(os.path.join(self.img_path, img_name))
        imgr = Image.open(os.path.join(self.imgr_path, img_name))

        label_name = img_name[:img_name.rfind(".")] + ".xml"
        label = xml_to_dict(os.path.join(self.label_path, label_name))
        labelr = xml_to_dict(os.path.join(self.labelr_path, label_name))

        for object in label["annotation"]["object"]:
            class_idx = self.classes.index(object["name"])
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = extract_points(object["polygon"])


        return img, imgr, label, labelr

if __name__ == "__main__":
    val_dataset = DroneVehicleYoloDataset("../../../data/DroneVehicle-DOTA/val")
    print(val_dataset[0])
