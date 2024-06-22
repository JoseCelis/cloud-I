import os
import sys
import cv2
import numpy as np

sys.path.append(os.getcwd())


def list_image_files(images_folder: str):
    """
    list all files in  data/ to create the X and target files list
    :param images_folder:
    :return:
    """
    images_list = [os.path.join(images_folder, image_name) for image_name in os.listdir(images_folder)
                   if ".DS_Store" not in image_name
                   ]
    return images_list


def from_mask_image_to_coordinates(mask_file_name: str, cloud_class: str):
    """
    It converts a mask image to many x, y coordinates.
    Used for training a SAM model.
    See: https://www.tutorialspoint.com/opencv_python/opencv_python_image_contours.htm
    See: https://docs.ultralytics.com/datasets/segment/
    """
    image_mask = cv2.imread(mask_file_name, 0)  # 0 for gray scale
    height, width = image_mask.shape

    if np.any(image_mask, where=255):  # convert if there are clouds
        image_mask = cv2.Canny(image_mask, 30, 200)  # Canny edge detection

        contours, _ = cv2.findContours(
            image_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        mask_coordinates = []
        for ncontour in range(len(contours)):
            contour = contours[ncontour]
            normalized_contour = contour.reshape(-1, 2) / [width, height]
            flatten_contour = normalized_contour.flatten()
            flatten_contour = np.insert(flatten_contour.astype("str"), 0, cloud_class)
            mask_coordinates.append(flatten_contour)
    else:
        mask_coordinates = None
    return mask_coordinates


def generate_labels_files(mask_path: str, label_path: str, cloud_class: str):
    images_filenames = list_image_files(mask_path)
    for index in range(len(images_filenames)):
        print(images_filenames[index])
        mask_fname = images_filenames[index]
        labels = from_mask_image_to_coordinates(mask_fname, cloud_class=str(cloud_class))

        if labels is not None:
            label_fname = mask_fname.split("/")[-1].split(".")[0] + ".txt"
            label_fname = label_fname.replace("MASK_", "RGB_")
            with open(os.path.join(label_path, label_fname), "w") as label_file:
                for mask_coords in labels:
                    label_file.write(" ".join(mask_coords) + "\n")


if __name__ == "__main__":

    train_mask_path = "Dataset/masks/train"
    val_mask_path = "Dataset/masks/val"
    train_label_path = "Dataset/labels/train/"
    val_label_path = "Dataset/labels/val/"

    [os.makedirs(path, exist_ok=True) for path in [train_label_path, val_label_path]]
    cloud_class = 0

    generate_labels_files(mask_path=train_mask_path, label_path=train_label_path, cloud_class=str(cloud_class))
    generate_labels_files(mask_path=val_mask_path, label_path=val_label_path, cloud_class=str(cloud_class))
