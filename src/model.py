import os
import sys
import click
import logging
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, FCN_model
from src.predict import plot_results


@click.command()
@click.option(
    "--model_name",
    required=True,
    type=str,
    help="It can be 'ann','rf', 'unet', 'fcn' or 'sam'.",
)
def main(model_name):
    logging.info('starting model')

    # TODO: implement sam
    model_dict = {'ann': ANN_model(), 'rf': RF_model(), 'unet': UNET_model(), 'fcn': FCN_model()}
    if model_name in model_dict.keys():
        model = model_dict[model_name]
        model.run()

        test_image_array = np.load(f'Dataset_npy/test/RGB_4481.npy')
        input_test_mask_array = np.load(f'Dataset_npy/test/MASK_4481.npy')
        test_image_array = test_image_array[tf.newaxis, :]
        test_mask_array = input_test_mask_array[tf.newaxis, :]
        predictions = model.make_predictions(test_image_array, use_saved_model=False)
        iou_score = model.intersection_over_union(test_mask_array, predictions)
        plot_results(test_image_array, input_test_mask_array, predictions, iou_score, model_name)
    else:
        raise ValueError(f"model {model_name} not defined.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
