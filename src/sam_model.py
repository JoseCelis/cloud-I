import os
import sys
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt


sys.path.append(os.getcwd())

data_config = os.path.join(os.getcwd(), "Dataset", "dataset.yaml")
epochs = 15

path_model = os.path.join(os.getcwd(), "model", "SAM", "segment", "train", "weights")
os.makedirs(path_model, exist_ok=True)


# # Train
# model = YOLO("yolov8n-seg.pt")  # pre-trained model
# results = model.train(data=data_config, epochs=epochs)

# predict
image = os.path.join(os.getcwd(), "Dataset", "test", "RGB_4481.png")


trained_model = YOLO(os.path.join(path_model, "last.pt"))
results = trained_model.predict(source=image, conf=0.0001, imgsz=1024)

for result in results:
    mask = result.masks
    mask_unique = mask.data.sum(axis=0)
    mask_unique = mask_unique.numpy()
    mask_unique = (mask_unique >= 1).astype(np.int8)

plt.imshow(mask_unique)
plt.show()