import os
import sys
import click
import logging
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, SEGNET_model, YOLO_model


def plot_results(test_image_array, input_test_mask_array, predictions, iou_score, model_name):
    images_folder = "images/"
    os.makedirs(images_folder, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    ax[0].imshow(test_image_array.reshape(1024, 1024, 3))
    ax[0].title.set_text("Image")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].imshow(input_test_mask_array, interpolation="nearest", cmap="Greys_r")
    ax[1].title.set_text("Mask")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    ax[2].imshow(predictions.reshape(1024, 1024, 1), interpolation="nearest", cmap="Greys_r")
    ax[2].title.set_text("Predicted Mask")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")

    fig.subplots_adjust(wspace=0.2)
    fig.suptitle(f"model={model_name}; iou_score={iou_score:.4};", fontsize=16)
    fig.savefig(os.path.join(images_folder, f"prediction_model={model_name}_iou={iou_score:.4}.png"))
    return None


def prepare_test_data(model_name):
    test_folder = "Dataset/test/"
    image = tf.keras.utils.load_img(os.path.join(test_folder, "RGB_4481.png"))
    mask = tf.keras.utils.load_img(os.path.join(test_folder, "MASK_4481.png"), color_mode="grayscale")
    test_image_array = tf.keras.utils.img_to_array(image, dtype=np.uint8)
    input_test_mask_array = tf.keras.utils.img_to_array(mask, dtype=bool).astype(np.uint8)
    if model_name == "rf":
        test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
        test_mask_array = np.ravel(input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1]))
    elif model_name == "ann":
        test_image_array = test_image_array.reshape(-1, test_image_array.shape[-1])
        test_mask_array = input_test_mask_array.reshape(-1, input_test_mask_array.shape[-1])
    elif model_name == "yolo":
        test_image_array = test_image_array
        test_mask_array = input_test_mask_array
    else:
        test_image_array = test_image_array[tf.newaxis, :]
        test_mask_array = input_test_mask_array[tf.newaxis, :]
    return input_test_mask_array, test_image_array, test_mask_array


@click.command()
@click.option(
    "--model_name",
    required=True,
    type=str,
    help="It can be 'ann','rf', 'unet', 'segnet' or 'yolo'.",
)
@click.option(
    "--use_weights",
    required=False,
    type=str,
    default=None,
    help="location of '.keras' file with the weights.",
)
@click.option(
    "--train_model",
    required=False,
    type=bool,
    default=True,
    help="Do you want to train the model?. default=True",
)
def main(model_name, use_weights, train_model):
    logging.info("starting model")
    input_test_mask_array, test_image_array, test_mask_array = prepare_test_data(model_name)

    model_dict = {"ann": ANN_model(), "rf": RF_model(), "unet": UNET_model(), "segnet": SEGNET_model(),
                  "yolo": YOLO_model()}
    if model_name in model_dict.keys():
        model = model_dict[model_name]
        if train_model:
            model.run(use_weights=use_weights)
            predictions = model.make_predictions(test_image_array, use_saved_model=False)
        else:
            predictions = model.make_predictions(test_image_array, use_saved_model=True)
        iou_score = model.intersection_over_union(test_mask_array, predictions)
        plot_results(test_image_array, input_test_mask_array, predictions, iou_score, model_name)
    else:
        raise ValueError(f"model {model_name} not defined.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
