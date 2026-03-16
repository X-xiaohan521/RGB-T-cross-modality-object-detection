from src.main.util.yml2model import *
from src.main.model.YoloOBB import OBBModel

if __name__ == "__main__":
    torch.serialization.add_safe_globals([OBBModel])

    yml_dict = load_yml("../../config/yolo11-obb.yaml")
    print(yml_dict)

    model = parse_model(yml_dict, input_channels=10)

    weights = torch.load("../../weight/yolo11n-obb.pt")
    model.load_weights(weights)
    print(weights['0.conv.weight'])