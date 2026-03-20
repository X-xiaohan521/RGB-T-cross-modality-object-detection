import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.main.util import xml_to_dict, extract_points, absolute2relative
from src.main.util.cut_white_frame import move_points, cut_white_frame


class DroneVehicleYoloDataset(Dataset):
    def __init__(self, data_path: str, classes: list, transform: transforms.Compose=None, white_frame: tuple[int, int, int, int]=None):
        '''
        Load the DroneVehicleYolo dataset from the given path.
        :param data_path: the path to load the dataset from
        :param classes: a list of classes name to which the objects should be classified
        :param transform: the transform to apply to the image
        :param white_frame: the white frame tuple to cut off, defining by the (left, upper, right, lower) pixel coordinate (if any)
        '''
        super().__init__()
        self.data_path = data_path
        self.classes = classes
        self.transform = transform
        self.white_frame = white_frame

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
        # open and transform img(r)
        img_name = self.img_names[idx]
        img = Image.open(os.path.join(self.img_path, img_name))
        imgr = Image.open(os.path.join(self.imgr_path, img_name))

        if self.white_frame is not None:
            img = cut_white_frame(img, self.white_frame)
            imgr = cut_white_frame(imgr, self.white_frame)

        # open and prepare label(r)
        label_name = img_name[:img_name.rfind(".")] + ".xml"
        label = xml_to_dict(os.path.join(self.label_path, label_name))
        labelr = xml_to_dict(os.path.join(self.labelr_path, label_name))

        y_list = []
        for object in label["annotation"]["object"]:
            class_idx = self.classes.index(object["name"])
            poly = object["polygon"]
            if self.white_frame is not None:
                poly = move_points(poly, self.white_frame)
            poly = absolute2relative(img.size, poly)
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = extract_points(poly)
            y = torch.tensor([class_idx, x1, y1, x2, y2, x3, y3, x4, y4])
            y_list.append(y)

        yr_list = []
        for object in labelr["annotation"]["object"]:
            class_idx = self.classes.index(object["name"])
            poly = object["polygon"]
            if self.white_frame is not None:
                poly = move_points(poly, self.white_frame)
            poly = absolute2relative(imgr.size, poly)
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = extract_points(poly)
            yr = torch.tensor([class_idx, x1, y1, x2, y2, x3, y3, x4, y4])
            yr_list.append(yr)

        if self.transform is not None:
            img = self.transform(img)
            imgr = self.transform(imgr)
        y_tensor, yr_tensor = torch.stack(y_list), torch.stack(yr_list)

        return img, imgr, y_tensor, yr_tensor

if __name__ == "__main__":
    val_dataset = DroneVehicleYoloDataset("../../../data/DroneVehicle-DOTA/val",
                                          ["car", "truck", "feright car", "bus", "van"],
                                          transforms.Compose([transforms.ToTensor()]),
                                          (100, 100, 100, 100))
    print(val_dataset[0])
