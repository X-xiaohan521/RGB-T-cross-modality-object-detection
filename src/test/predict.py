from src.main.util.yml2model import *

if __name__ == "__main__":
    yml_dict = load_yml("../main/config/yolo11-obb.yaml")
    print(yml_dict)

    model = parse_model(yml_dict, input_channels=10)
    print(model)