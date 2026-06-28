from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="VisDrone.yaml",
    epochs=30,
    imgsz=640,
    batch=4,              # CPU-safe
    device="cpu",
    workers=4,
    classes=[0, 1],       # pedestrian + people only
    project="napsee_train",
    name="yolov8n_napsee",
    exist_ok=True,
)