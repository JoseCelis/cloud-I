import os
import sys
import rasterio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


sys.path.append(os.getcwd())


def list_image_files(images_folder: str):
    """
    list all folders in  data/ to create the X and target files list
    :param images_folder:
    :return:
    """
    bands_clp_list = [[os.path.join(images_folder, path, 'bands.tif'), os.path.join(images_folder, path, 'CLP.tif')]
                      for path in os.listdir(images_folder)]
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
    We perform rotations and flips to increase the number of images, for every image we read, we get 5 extra ones
    90, 180, 270, vertical_flip, horizontal_flip
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


def save_image_files(preprocessed_image_folder, counter, aug_image_list, aug_mask_list, n_files):
    for i, image_maks_pair in enumerate(zip(aug_image_list, aug_mask_list)):
        counter_batch = i * n_files + counter
        np.save(os.path.join(preprocessed_image_folder, f"RGB_{counter_batch}.npy"), image_maks_pair[0])
        np.save(os.path.join(preprocessed_image_folder, f"MASK_{counter_batch}.npy"), image_maks_pair[1])
    return None


def main():
    input_images_folder = 'data'
    preprocessed_image_folder = 'preprocessed_data'
    bands = [3, 2, 1]
    threshold_prob = 0.4  # bit we use to detect as truth in the mask files

    images_list = list_image_files(input_images_folder)
    os.makedirs(preprocessed_image_folder, exist_ok=True)
    for counter, image_mask_pair in tqdm(enumerate(images_list)):
        image, _ = read_crop(image_mask_pair[0], bands=bands)
        mask, _ = read_crop(image_mask_pair[1], bands=[1])
        image = preprocess_image(image)
        mask = preprocess_mask(mask, threshold_prob)
        aug_image_list, aug_mask_list = augment_data(image, mask)
        save_image_files(preprocessed_image_folder, counter, aug_image_list, aug_mask_list, len(images_list))


if __name__ == "__main__":
    main()
