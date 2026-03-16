from src.main.util.yml2model import *
from src.main.model.YoloOBB import OBBModel

if __name__ == "__main__":
    yml_dict = load_yml("../../config/yolo11-obb.yaml")
    model, sort = parse_model(yml_dict, input_channels=3)   # model with backbone and head
    obb = OBBModel(model)
    weight = torch.load("../../weight/yolo11n-obb-weight.pt")
    obb.load(weight)

    print(obb.state_dict()['model.0.conv.weight'])
    