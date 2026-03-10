from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("../../weight/yolo11n-obb.pt")
    print(model.state_dict())
    result = model.predict("C:\\Users\dujin\\Documents\\Code\\RGB-T-cross-modality-object-detection\data\DroneVehicle-DOTA\\val\\valimg\\00017.jpg", save=True)
    print(result)
