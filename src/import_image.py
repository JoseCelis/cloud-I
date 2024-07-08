import os
import sys
import yaml
import logging
import tarfile
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from sentinelhub import CRS, BBox, DataCollection, MimeType, SentinelHubRequest, SHConfig, generate_evalscript

sys.path.append(os.getcwd())


def read_yaml_file(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_credentials():
    """
    load the credentials and fill the sentinel hub config
    :return: configuration credentials
    """
    config = SHConfig()
    if os.getenv('CLIENT_ID') and os.getenv('CLIENT_SECRET'):
        config.sh_client_id = os.getenv('CLIENT_ID')
        config.sh_client_secret = os.getenv('CLIENT_SECRET')
    return config


def request_image_and_mask_to_sentinel2_L2A(bbox, im_date, sh_config, data_folder='data'):
    """
    max_values = {MimeType.TIFF: 65535, MimeType.PNG: 255, MimeType.JPG: 255, MimeType.JP2: 10000}
    :param bbox: bounding box
    :param im_date: Date of the image
    :param sh_config: sentinel hub config filled with the credentials
    :return: None
    """
    data_collection = DataCollection.SENTINEL2_L2A
    evalscript = generate_evalscript(
        data_collection=data_collection,
        meta_bands=["CLP"],  # cloud probabilities
        merged_bands_output="bands",  # all bands
        prioritize_dn=True  # use Digital Numbers (DN)
    )
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=data_collection, time_interval=im_date)],
        responses=[SentinelHubRequest.output_response("bands", MimeType.TIFF),
                   SentinelHubRequest.output_response("CLP", MimeType.TIFF)],
        bbox=bbox,
        resolution=(10, 10),
        config=sh_config,
        data_folder=data_folder
    )
    logging.debug(f'Downloading image from coordinates {bbox}')
    _ = request.get_data(save_data=True)[0]
    return None


def extract_tar_file(tar_file, path):
    """
    Extracts image and cloud probabilities to the folder of the request.
    :param tar_file:
    :param path:
    :return:
    """
    if tarfile.is_tarfile(os.path.join(path, tar_file)):
        opened_tar = tarfile.open(os.path.join(path, tar_file))
        opened_tar.extractall(path)
        opened_tar.close()
    else:
        print("The tar file you entered is not a tar file")


def main():
    images_folder = "data"
    os.makedirs(images_folder, exist_ok=True)

    sh_config = load_credentials()
    bbox_dicts = read_yaml_file("settings/coordinates.yaml")
    for bbox_dict in tqdm(bbox_dicts):
        bbox = BBox(tuple(bbox_dict["coords"]), crs=CRS(bbox_dict["CRS"]))
        request_image_and_mask_to_sentinel2_L2A(bbox, bbox_dict["date"], sh_config, data_folder=images_folder)
    logging.info("Extracting .tar files . . .")
    [extract_tar_file("response.tar", os.path.join(images_folder, path)) for path in os.listdir(images_folder)]


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
