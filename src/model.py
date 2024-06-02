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

    # TODO: implement sam
    model_dict = {'ann': ANN_model(), 'rf': RF_model(), 'unet': UNET_model(), 'fcn': FCN_model()}
    if model_name in model_dict.keys():
        model = model_dict[model_name]
        model.run()
    else:
        raise ValueError(f"model {model_name} not defined.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
