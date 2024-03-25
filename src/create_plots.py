import os
import sys
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

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


def main():
    input_images_folder = 'failure'
    preprocessed_image_folder = 'preprocessed_data'
    bands = [3, 2, 1]
    threshold_prob = 0.4  # bit we use to detect as truth in the mask files

    image_mask_pairs = list_image_files(input_images_folder)
    for image_mask_pair in image_mask_pairs:
        # os.makedirs(preprocessed_image_folder, exist_ok=True)
        raw_image, _ = read_crop(image_mask_pair[0], bands=bands)
        cl_probs, _ = read_crop(image_mask_pair[1], bands=[1])
        image = preprocess_image(raw_image)
        mask = preprocess_mask(cl_probs, threshold_prob)
        print('unique raw image:', np.unique(raw_image))
        print('unique cloud probs:', np.unique(cl_probs))
        print(image_mask_pair)

    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    ax[0].imshow(tf.keras.utils.array_to_img(raw_image.transpose((1, 2, 0))), aspect="auto")
    ax[0].title.set_text("Raw Image")
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].hist(raw_image[0].flatten(), bins=100, color='red', alpha=0.5, label='Red band')
    ax[1].hist(raw_image[1].flatten(), bins=100, color='green', alpha=0.5, label='Green band')
    ax[1].hist(raw_image[2].flatten(), bins=100, color='blue', alpha=0.5, label='Blue band')
    ax[1].title.set_text("Bands distribution in Raw Image")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel('Digital Numbers')
    ax[1].set_ylabel('Counts')

    f1 = ax[2].imshow(tf.keras.utils.array_to_img(cl_probs.transpose((1, 2, 0))), cmap='gray', aspect="auto", vmin=0, vmax=255)
    ax[2].title.set_text("Cloud probabilities")
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')
    fig.subplots_adjust(wspace=0.2)
    fig.colorbar(f1, ax=ax[2])


    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    ax[0].imshow(tf.keras.utils.array_to_img(image), aspect="auto")
    ax[0].title.set_text("Processed Image")
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].hist(image[0].flatten(), bins=50, color='red', alpha=0.5, label='Red band')
    ax[1].hist(image[1].flatten(), bins=50, color='green', alpha=0.5, label='Green band')
    ax[1].hist(image[2].flatten(), bins=50, color='blue', alpha=0.5, label='Blue band')
    ax[1].title.set_text("Bands distribution in Processed Image")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel('Pixel values')
    ax[1].set_ylabel('Counts')

    f2 = ax[2].imshow(tf.keras.utils.array_to_img(mask), cmap='gray', aspect="auto", vmin=0, vmax=1)
    ax[2].title.set_text("Cloud mask")
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')
    fig.subplots_adjust(wspace=0.2)
    fig.colorbar(f2, ax=ax[2])


if __name__ == "__main__":
    main()
