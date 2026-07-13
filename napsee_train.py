from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="VisDrone.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=4,
    classes=[0, 1],
    project=r"F:\Github\NapSee\secondrun",
    name="visdrone_person",
    exist_ok=True,
    verbose=True,
    plots=True,
    save=True,
    save_period=5,
    val=True,
    patience=20,
    cache=True
)