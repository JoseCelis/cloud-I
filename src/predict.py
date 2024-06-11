import click
import os
import sys
import logging
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, FCN_model


def plot_results(test_image_array, input_test_mask_array, predictions, iou_score, model_name):
    images_folder = 'images/'
    os.makedirs(images_folder, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    ax[0].imshow(test_image_array.reshape(1024, 1024, 3))
    ax[0].title.set_text("Image")
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].imshow(input_test_mask_array, interpolation='nearest', cmap='Greys_r')
    ax[1].title.set_text("Mask")
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    ax[2].imshow(predictions.reshape(1024, 1024, 1), interpolation='nearest', cmap='Greys_r')
    ax[2].title.set_text("Predicted Mask")
    ax[2].set_xlabel('X')
    ax[2].set_ylabel('Y')

    fig.subplots_adjust(wspace=0.2)
    fig.suptitle(f'model={model_name}; iou_score={iou_score:.4};', fontsize=16)
    fig.savefig(os.path.join(images_folder, f'prediction_model={model_name:.4}_iou={iou_score}.png'))
    return None


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
    input_test_mask_array = np.load(f'Dataset_npy/test/MASK_4481.npy')

    # TODO: implement sam
    model_dict = {'ann': ANN_model(), 'rf': RF_model(), 'unet': UNET_model(), 'fcn': FCN_model()}
    if model_name in model_dict.keys():
        if model_name == 'rf':
            test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
            test_mask_array = np.ravel(input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1]))
        elif model_name == 'ann':
            test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
            test_mask_array = input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1])
        else:
            test_image_array = test_image_array[tf.newaxis, :]
            test_mask_array = np.array([input_test_mask_array])
        suffix = 'pkl' if model_name == 'rf' else 'keras'
        if os.path.exists(os.path.join('models', f'{str.upper(model_name)}.{suffix}')):
            model = model_dict[model_name]
            predictions = model.make_predictions(test_image_array)
            iou_score = model.intersection_over_union(test_mask_array, predictions)
            plot_results(test_image_array, input_test_mask_array, predictions, iou_score, model_name)
        else:
            raise ValueError(f"model {model_name} does not exist.")
    else:
        raise ValueError(f"model {model_name} not defined.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
