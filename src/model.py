import os
import sys
import click
import logging
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, SEGNET_model
from src.predict import plot_results


@click.command()
@click.option(
    "--model_name",
    required=True,
    type=str,
    help="It can be 'ann','rf', 'unet' or 'segnet'.",
)
@click.option(
    "--use_weights",
    required=False,
    type=str,
    default=None,
    help="location of '.keras' file with the weights.",
)
def main(model_name, use_weights):
    logging.info('starting model')

    # TODO: implement sam
    model_dict = {'ann': ANN_model(), 'rf': RF_model(), 'unet': UNET_model(), 'segnet': SEGNET_model()}
    if model_name in model_dict.keys():
        model = model_dict[model_name]
        model.run(use_weights=use_weights)

        test_folder = 'Dataset/test/'
        image = tf.keras.utils.load_img(os.path.join(test_folder, 'RGB_4481.png'))
        mask = tf.keras.utils.load_img(os.path.join(test_folder, 'MASK_4481.png'), color_mode='grayscale')
        test_image_array = tf.keras.utils.img_to_array(image, dtype=np.uint8)
        input_test_mask_array = tf.keras.utils.img_to_array(mask, dtype=bool).astype(np.uint8)
        if model_name == 'rf':
            test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
            test_mask_array = np.ravel(input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1]))
        elif model_name == 'ann':
            test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
            test_mask_array = input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1])
        else:
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
