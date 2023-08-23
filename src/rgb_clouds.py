import numpy as np
from PIL import Image
from pathlib import Path


image_path = "../fotos_cloud/38-Cloud_training"

def convert_tif_png(tif_filename: Path, out_folder:Path):
    # Open mask with PIL
    arr_tif = np.array(Image.open(tif_filename))

    # change values from 255 to 1
    im = Image.fromarray(np.where(arr_tif==255, 1, 0))
    im.save(out_folder/tif_filename.with_suffix('.png').name)
    return im


def main():
    # create the rgb output dir
    if not (Path(image_path)/'labels').exists():
        (Path(image_path)/'labels').mkdir()

    # loop trough the red patches and create the corresponding rgb ones
    for gt_patch in (Path(image_path)/'train_gt').iterdir():
        convert_tif_png(gt_patch, Path(image_path)/'labels')


if __name__ == "__main__":
    main()
