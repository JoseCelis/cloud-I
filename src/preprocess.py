import os
import cv2
import sys
import rasterio
import numpy as np
from tqdm import tqdm
import tensorflow as tf

sys.path.append(os.getcwd())


def list_image_files(images_folder: str):
    """
    list all folders in  data/ to create the X and target files list
    :param images_folder:
    :return:
    """
    bands_clp_list = [[os.path.join(images_folder, path, 'bands.tif'), os.path.join(images_folder, path, 'CLP.tif')]
                      for path in os.listdir(images_folder) if "." not in path]
    return bands_clp_list


def read_crop(image_file_name, crop=None, bands=None):
    """
    Read rasterio `crop` for the given `bands`
    :param image_file_name: full path of the image.
    :param crop: Tuple or list containing the area to be cropped (px, py, w, h).
    :param bands: List of `bands` to read from the dataset.
    :return: A numpy array containing the read image `crop` (bands * h * w).
    """
    rasterio_dataset = rasterio.open(image_file_name, mode='r')
    if bands is None:
        bands = [i for i in range(1, rasterio_dataset.count + 1)]
    else:
        assert len(bands) <= rasterio_dataset.count, \
            "`bands` cannot contain more bands than the number of bands in the dataset."
        assert max(bands) <= rasterio_dataset.count, \
            "The maximum value in `bands` should be smaller or equal to the band count."

    window = None
    if crop is not None:
        assert len(crop) == 4, "`crop` should be a tuple or list of shape (px, py, w, h)."
        px, py, w, h = crop
        w = rasterio_dataset.width - px if (px + w) > rasterio_dataset.width else w
        h = rasterio_dataset.height - py if (py + h) > rasterio_dataset.height else h
        assert (px + w) <= rasterio_dataset.width, "The crop (px + w) is larger than the dataset width."
        assert (py + h) <= rasterio_dataset.height, "The crop (py + h) is larger than the dataset height."
        window = rasterio.windows.Window(px, py, w, h)
    image_metadata = rasterio_dataset.meta
    image_metadata.update(count=len(bands))
    if crop is not None:
        image_metadata.update({'height': window.height, 'width': window.width,
                               'transform': rasterio.windows.transform(window, rasterio_dataset.transform)})
    return rasterio_dataset.read(bands, window=window), image_metadata


def preprocess_image(image: np.array):
    """
    Calculates the log1p to reduce the variance and normalize to save the results in int8 format
    :param image:
    :return: norm_log_image
    """
    log_image = np.log1p(image)
    min_rescaled_log_image = log_image - log_image.min()
    if min_rescaled_log_image.max() != 0:
        norm_log_image = np.round((min_rescaled_log_image / min_rescaled_log_image.max()) * 255).astype('uint8')
    else:
        norm_log_image = np.zeros(np.shape(image))
    norm_log_image = norm_log_image.transpose((1, 2, 0))
    return norm_log_image


def preprocess_mask(mask: np.array, threshold_prob: float):
    """
    Creates binary file using threshold_prob
    :param mask:
    :param threshold_prob:
    :return: binary mask
    """
    binary_mask = (mask > (threshold_prob * 255)).astype(int)
    binary_mask = binary_mask.transpose((1, 2, 0))
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask


def augment_data(image, mask):
    """
    We perform rotations and flips to increase the number of images. For every image we read, we get 5 extra ones:
    90, 180 and 270 degrees rotations, and vertical and horizontal flip. Giving a total of six images.
    """
    image_list, mask_list = [], []
    image_list.append(image)
    mask_list.append(mask)
    image_list.append(np.rot90(image))
    mask_list.append(np.rot90(mask))
    image_list.append(np.rot90(np.rot90(image)))
    mask_list.append(np.rot90(np.rot90(mask)))
    image_list.append(np.rot90(np.rot90(np.rot90(image))))
    mask_list.append(np.rot90(np.rot90(np.rot90(mask))))
    image_list.append(np.flip(image, 0))
    mask_list.append(np.flip(mask, 0))
    image_list.append(np.flip(image, 1))
    mask_list.append(np.flip(mask, 1))
    return image_list, mask_list


def from_mask_image_to_coordinates(image_mask, cloud_class: str):
    """
    It converts a mask image to many x, y coordinates.
    Used for training a SAM model.
    See: https://www.tutorialspoint.com/opencv_python/opencv_python_image_contours.htm
    See: https://docs.ultralytics.com/datasets/segment/
    """
    image_mask = image_mask * 255  # multiply with 255 for cv process
    height, width = np.squeeze(image_mask).shape

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


def save_image_files(preprocessed_folder, counter_batch, image_array, mask_array, labels):
    """
    Save augmented data: images, masks and labels.
    If originally we have 100 images, the original image 'k' will be saved to the file 'k.png' and the augmented data
    in the files '100+k.png', '200+k.png', '300+k.png', '400+k.png', '500+k.png'. If the original image belongs to the
    train, all augmented images will be saved in the train folder as well.
    """
    image_folder = preprocessed_folder.format(image_mask_or_label='images')
    mask_folder = preprocessed_folder.format(image_mask_or_label='masks')
    label_folder = preprocessed_folder.format(image_mask_or_label='labels')
    img = tf.keras.utils.array_to_img(image_array)
    img_file_name = os.path.join(image_folder, f"{counter_batch}.png")
    img.save(img_file_name)
    mask = tf.keras.utils.array_to_img(mask_array)
    mask_file_name = os.path.join(mask_folder, f"{counter_batch}.png")
    mask.save(mask_file_name)
    if labels is not None:
        label_file_name = os.path.join(label_folder, f"{counter_batch}.txt")
        with open(label_file_name, "w") as label_file:
            for mask_coords in labels:
                label_file.write(" ".join(mask_coords) + "\n")
    return None


def main():
    input_images_folder = 'data'  # images are in Digital Numbers (DN)
    dataset_folder = 'Dataset/'
    folder_structure_list = ['Dataset/images/train/', 'Dataset/images/val/', 'Dataset/masks/train/',
                             'Dataset/masks/val/', 'Dataset/labels/train/', 'Dataset/labels/val/']
    [os.makedirs(folder, exist_ok=True) for folder in folder_structure_list]
    train_val_split = 0.8  # we are using 80-20 train validation split

    bands = [3, 2, 1]  # RGB bands of the original tiff file
    threshold_prob = 0.4  # bit we use to detect as truth in the mask files

    images_list = list_image_files(input_images_folder)
    for counter, image_mask_pair in tqdm(enumerate(images_list)):
        image, _ = read_crop(image_mask_pair[0], bands=bands)
        mask, _ = read_crop(image_mask_pair[1], bands=[1])
        image = preprocess_image(image)
        mask = preprocess_mask(mask, threshold_prob)
        aug_image_list, aug_mask_list = augment_data(image, mask)
        train_val_folder = 'train/' if counter < train_val_split * len(images_list) else 'val/'
        output_folder = os.path.join(dataset_folder, '{image_mask_or_label}/', train_val_folder)
        for i, image_maks_pair in enumerate(zip(aug_image_list, aug_mask_list)):
            counter_batch = i * len(images_list) + counter
            label = from_mask_image_to_coordinates(image_maks_pair[1], cloud_class=str(0))
            save_image_files(output_folder, counter_batch, image_maks_pair[0], image_maks_pair[1], label)


if __name__ == "__main__":
    main()
