import os
import sys

from ultralytics import YOLO
from ultralytics import settings

sys.path.append(os.getcwd())

data_config = "Dataset/dataset.yaml"
epochs = 10

path_model_runs = "model/SAM"
os.makedirs(path_model_runs, exist_ok=True)

settings.update({
    "runs_dir": path_model_runs,
})

model = YOLO("yolov8n-seg.pt") # pre-trained model
results = model.train(data=data_config, epochs=epochs)

