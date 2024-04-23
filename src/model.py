import click
import os
import sys
import logging
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
    if model_name == "ann":
        model = ANN_model()
    elif model_name == "rf":
        model = RF_model()
    elif model_name == "unet":
        model = UNET_model()
    elif model_name == "fcn":
        model = FCN_model()
    elif model_name == "sam":
        model = "HERE ADD SAM MODEL"  # TODO: implement sam
    else:
        raise ValueError(f"model {model_name} not defined.")
    model.run()


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
