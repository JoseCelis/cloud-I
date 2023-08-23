import os
import sys
import time
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import rasterio
from itertools import product


os.chdir(os.getcwd())


def list_image_files(images_folder: str):
    """
    list all files in the images_folder to create the X and target files list
    :param images_folder:
    :return:
    """
    rgb_files = [file for file in os.listdir(images_folder) if ('Sentinel2L2A_sen2cor_' in file) and ('_RGB' in file)]
    mask_files = [file for file in os.listdir(images_folder) if 'cloud_mask_' in file]
    rgb_files.sort()
    mask_files.sort()
    assert len(mask_files) == len(rgb_files), "The number of mask files and RGB files are different. "
    return rgb_files, mask_files


def get_random_pixel_coords(original_image_height: int, original_image_width: int, crop_width: int, crop_height: int,
                            ncrops: int):
    """
    Creates random coordinates to obtain ncrops from the images
    :param original_image_height:
    :param original_image_width:
    :param crop_width:
    :param crop_height:
    :param ncrops:
    :return: pixel_coords = (left, top, tilesize, tilesize)
    """
    pixel_x = np.random.randint(0, original_image_width - crop_width, size=ncrops)
    pixel_y = np.random.randint(0, original_image_height - crop_height, size=ncrops)

    coords = list(zip(pixel_x, pixel_y, crop_width * np.ones(ncrops, dtype=np.int8),
                      crop_height * np.ones(ncrops, dtype=np.int8)))
    return coords


def preprocess_image(image: np.array):
    log_image = np.log1p(image)
    min_rescaled_log_image = log_image - log_image.min()
    if min_rescaled_log_image.max() != 0:
        norm_log_image = np.round((min_rescaled_log_image / min_rescaled_log_image.max()) * 255).astype('uint8')
    else:
        norm_log_image = np.zeros(np.shape(image))
    norm_log_image = norm_log_image.transpose((1, 2, 0))
    return norm_log_image


def preprocess_mask(mask: np.array, bit: int):
    binary_mask = (mask & bit) / bit
    binary_mask = binary_mask.transpose((1, 2, 0))
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask


def read_crop(image_file_name, crop=None, bands=None):
    """
    Read rasterio `crop` for the given `bands`
    Args:
        image_file_name: full path of the image.
        crop: Tuple or list containing the area to be cropped (px, py, w, h).
        bands: List of `bands` to read from the dataset.
    Returns:
        A numpy array containing the read image `crop` (bands * h * w).
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


def main():
    input_images_folder = './images'
    preprocessed_image_folder = './proprocessed_images'
    NCROPS = 64
    IMAGE_HEIGHT = IMAGE_WIDTH = 10980
    CROP_HEIGHT = CROP_WIDTH = 1024  # pixels of the new images, must be power of 2 so that the U-net runs
    bands = [3, 2, 1]
    bit = 4  # bit we use to detect as truth in the mask files

    X, y = list_image_files(input_images_folder)
    pixel_coords = get_random_pixel_coords(original_image_height=IMAGE_HEIGHT,original_image_width=IMAGE_WIDTH,
                                           crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT, ncrops=NCROPS)
    os.makedirs(preprocessed_image_folder, exist_ok=True)
    for image_file_name, mask_file_name in zip(X, y):
        for counter, pixel_coord in enumerate(pixel_coords):
            image, _ = read_crop(os.path.join(input_images_folder, image_file_name), pixel_coord, bands)
            mask, _ = read_crop(os.path.join(input_images_folder, mask_file_name), pixel_coord, bands=[1])
            image = preprocess_image(image)
            mask = preprocess_mask(mask, bit)
            np.save(os.path.join(preprocessed_image_folder,
                                 f"{image_file_name.replace('.tif', '')}_{counter}.npy"), image)
            np.save(os.path.join(preprocessed_image_folder,
                                 f"{mask_file_name.replace('.tif', '')}_{counter}.npy"), mask)


if __name__ == "__main__":
    main()
