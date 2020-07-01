import h5py
import hdbscan
import numpy as np
import time
import torch

from unet3d.utils import get_logger
from unet3d.utils import unpad

logger = get_logger('UNet3DTrainer')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError


class StandardPredictor(_AbstractPredictor):

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)

    def predict(self):
        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(self.loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self._volume_shape(self.loader.dataset)
        prediction_maps_shape = (out_channels,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        avoid_block_artifacts = self.predictor_config.get('avoid_block_artifacts', True)
        logger.info(f'Avoid block artifacts: {avoid_block_artifacts}')

        # create destination H5 file
        h5_output_file = h5py.File(self.output_file, 'w')
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads, h5_output_file)

        # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        if avoid_block_artifacts:
                            # unpad in order to avoid block artifacts in the output probability maps
                            u_prediction, u_index = unpad(pred, index, volume_shape)
                            # accumulate probabilities into the output prediction array
                            prediction_map[u_index] += u_prediction
                            # count voxel visits for normalization
                            normalization_mask[u_index] += 1
                        else:
                            # accumulate probabilities into the output prediction array
                            prediction_map[index] += pred
                            # count voxel visits for normalization
                            normalization_mask[index] += 1

        # save results to
        self._save_results(prediction_maps, normalization_masks, output_heads, h5_output_file, self.loader.dataset)
        # close the output H5 file
        h5_output_file.close()

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset):
        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding:
                pad_width = dataset.pad_width
                logger.info(f'Dataset loaded with mirror padding, pad_width: {pad_width}. Cropping before saving...')

                prediction_map = prediction_map[:, pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

            logger.info(f'Saving predictions to: {output_file}/{prediction_dataset}...')
            output_file.create_dataset(prediction_dataset, data=prediction_map, compression="gzip")
