import os
import sys
import logging
import numpy as np
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model


def main():
    logging.info('starting model')
    # ann_model = ANN_model()
    # ann_model.run()
    # rf_model = RF_model()
    # rf_model.run()
    rf_model = UNET_model()
    rf_model.run()



if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
