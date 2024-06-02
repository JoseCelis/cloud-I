import click
import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, FCN_model


@click.command()
@click.option(
    "--model_name",
    required=True,
    type=str,
    help="It can be 'ann','rf', 'unet', 'fcn' or 'sam'.",
)
def main(model_name):
    logging.info('starting model')
    test_image_array = np.load(f'Dataset_npy/test/RGB_4481.npy')
    test_mask_array = np.load(f'Dataset_npy/test/MASK_4481.npy')

    # TODO: implement sam
    model_dict = {'ann': ANN_model(), 'rf': RF_model(), 'unet': UNET_model(), 'fcn': FCN_model()}
    if model_name in model_dict.keys():
        if model_name in ['ann', 'rf']:
            test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
            test_mask_array = np.ravel(test_mask_array.reshape(-1, test_mask_array.shape[-1]))
        else:
            test_image_array = np.array([test_image_array])
            test_mask_array = np.array([test_mask_array])
        suffix = 'pkl' if model_name == 'rf' else 'h5'
        if os.path.exists(os.path.join('models', f'{str.upper(model_name)}.{suffix}')):
            model = model_dict[model_name]
            predictions = model.make_predictions(test_image_array)
            iou_score = model.intersection_over_union(test_mask_array, predictions)
            print(iou_score)
        else:
            raise ValueError(f"model {model_name} does not exist.")
    else:
        raise ValueError(f"model {model_name} not defined.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
