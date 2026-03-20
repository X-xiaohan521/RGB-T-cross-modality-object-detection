import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import xmltodict as xtd
from torchvision import transforms


class HelmetVocDataset(Dataset):
    def __init__(self, image_path, label_path, classes: list, transform=None):
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.classes = classes
        self.transform = transform

        self.image_names = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # prepare img
        img = Image.open(os.path.join(self.image_path, self.image_names[idx]))
        if self.transform is not None:
            img = self.transform(img)

        # prepare label
        label = []
        label_name = self.image_names[idx][:self.image_names[idx].rfind(".")] + ".xml"
        label_dict = None
        with open(os.path.join(self.label_path, label_name), encoding="utf-8") as f:
            label_content = f.read()
            label_dict = xtd.parse(label_content)
        assert label_dict is not None
        objects = label_dict['annotation']['object']
        for object in objects:
            name = object['name']
            class_id = self.classes.index(name)

            object_bbox = object['bndbox']
            xmax = int(object_bbox['xmax'])
            xmin = int(object_bbox['xmin'])
            ymax = int(object_bbox['ymax'])
            ymin = int(object_bbox['ymin'])

            label.append(torch.tensor([class_id, xmin, ymin, xmax, ymax], dtype=torch.float32))
        label = torch.stack(label, dim=0).flatten()
        return img, label

if __name__ == "__main__":
    train_dataset = HelmetVocDataset("..\\..\\..\\data\\HelmetDataset-VOC\\train\\images",
                                     "..\\..\\..\\data\\HelmetDataset-VOC\\train\\labels",
                                     ["no helmet", "motor", "number", "with helmet"],
                                     transform=transforms.Compose([transforms.ToTensor()]))
    y = []
    x = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        y_pixels = img.shape[1]
        x_pixels = img.shape[2]

        y.append(y_pixels)
        x.append(x_pixels)

    print(max(x))
    print(max(y))
