import importlib
import os

from datasets.hdf5 import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model

logger = utils.get_logger('UNet3DPredictor')

def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('unet3d.predictor')
    predictor_class = getattr(m, class_name)

    # model: UNet3D, loader: test_loader, output_file: data.h5, config: config.yaml
    return predictor_class(model, loader, output_file, config, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])

    logger.info('Loading HDF5 datasets...')

    test_loader = get_test_loaders(config)['test']
    for i, data_pair in enumerate(test_loader):
        output_file = 'predict_' + str(i) + '.h5'
        predictor = _get_predictor(model, data_pair, output_file, config)
        predictor.predict()

if __name__ == '__main__':
    main()
