import os
import sys
import logging
import numpy as np
sys.path.append(os.getcwd())
from src.model_class import ANN_model, RF_model, UNET_model, FCN_model


def main():
    logging.info('starting model')
    # model = ANN_model()
    # model = RF_model()
    model = UNET_model()
    # model = FCN_model()
    model.run()



if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    main()
