import os
import sys

from ultralytics import YOLO
from ultralytics import settings

sys.path.append(os.getcwd())

data_config = os.path.join(os.getcwd(), "Dataset", "dataset.yaml")
epochs = 15

path_model_runs = os.path.join(os.getcwd(), "Dataset", "output", "SAM")
os.makedirs(path_model_runs, exist_ok=True)

settings.update({
    "runs_dir": path_model_runs,
    "weights_dir": path_model_runs,
})

model = YOLO("yolov8n-seg.pt")  # pre-trained model
results = model.train(data=data_config, epochs=epochs)
