from ultralytics import YOLO

# Nie trenujemy modelu od zera, tylko "dostrajamy" (fine-tuning) model, który już został wstępnie nauczony rozpoznawania ogólnych obiektów.
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    results = model.train(
        data='plate_dataset.yaml',
        epochs=25,
        imgsz=640
    )