from ultralytics import YOLO

def train():
    model = YOLO("yolov8s.pt")
    model.train(
        data="solidworks.yaml",
        epochs=50,
        imgsz=512,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        val=False
    )

if __name__ == "__main__":
    train()
