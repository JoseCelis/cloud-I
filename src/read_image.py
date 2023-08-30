import os
import rasterio
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Input, InputLayer, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.chdir(os.getcwd())

def open_rasterio_file(path):
    """
    Open rasterio file using read mode
    """
    return rasterio.open(path, mode='r')


def read_crop(ds, crop=None, bands=None):
    """
    Read rasterio `crop` for the given `bands`..
    Args:
        ds: Rasterio dataset.
        crop: Tuple or list containing the area to be cropped (px, py, w, h).
        bands: List of `bands` to read from the dataset.
    Returns:
        A numpy array containing the read image `crop` (bands * h * w).
    """
    ds = open_rasterio_file(ds)
    if bands is None:
        bands = [i for i in range(1, ds.count+1)]
    else:
        assert len(bands) <= ds.count, "`bands` cannot contain more bands than the number of bands in the dataset."
        assert max(bands) <= ds.count, "The maximum value in `bands` should be smaller or equal to the band count."

    window = None
    if crop is not None:
        assert len(crop) == 4, "`crop` should be a tuple or list of shape (px, py, w, h)."
        px, py, w, h = crop
        w = ds.width - px if (px + w) > ds.width else w
        h = ds.height - py if (py + h) > ds.height else h
        assert (px + w) <= ds.width, "The crop (px + w) is larger than the dataset width."
        assert (py + h) <= ds.height, "The crop (py + h) is larger than the dataset height."
        window = rasterio.windows.Window(px, py, w, h)
    meta = ds.meta
    meta.update(count=len(bands))
    if crop is not None:
        # make the aoi more smooth so data is easier correctly downloaded
        meta.update({'height': window.height, 'width': window.width,
                     'transform': rasterio.windows.transform(window, ds.transform)})
    return ds.read(bands, window=window), meta


def main():
    main_path = 'images'
    crop = (0, 0, 1000, 1000)
    crop = None
    rgb, meta_rgb = read_crop(os.path.join(main_path,
                                           'Sentinel2L2A_sen2cor_49RFN_20190806_clouds=6.2%_area=99%_RGB.tif'), crop,
                              bands=[1, 2, 3])
    log_rgb = np.log1p(rgb)
    min_rescaled_log_rgb = log_rgb - log_rgb.min()
    norm_log_rgb = np.round((min_rescaled_log_rgb / min_rescaled_log_rgb.max()) * 255).astype('uint8')
    label, meta_label = read_crop(os.path.join(main_path, 'cloud_mask_49RFN_20190806_clouds=61%.tif'), crop)
    label_bin = (label > 0).astype(int)
    print(f'The shape of the rgb file is {norm_log_rgb.shape}\nthe shape of the label file is {label_bin.shape}')


if __name__ == "__main__":
    main()
