import os
import sys
import numpy as np
sys.path.append(os.getcwd())
from ultralytics import YOLO, SAM, settings


def main():
    # Load a pretrained YOLOv8 model (recommended for cloud detection)
    # model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
    os.makedirs(os.path.join("model", "SAM"), exist_ok=True)

    settings.update({
        "runs_dir": os.path.join("model", "SAM"),
        "weights_dir": os.path.join("model", "SAM"),
    })
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    results = model.train(data=os.path.join(os.getcwd(), "Dataset", "dataset.yaml"),
                          epochs=5,
                          val=True,
                          save=True,
                          )
    # # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # #Load Pre-trained SAM segmentation model.
    # # model = SAM('sam_b.pt')
    # image_path = 'preprocessed_data/RGB_1.npy'
    # image_data = np.load(image_path)
    # result = model.predict(image_data, prompts="Segment clouds in this satellite image")
    # print('result is ready')


if __name__ == "__main__":
    main()
