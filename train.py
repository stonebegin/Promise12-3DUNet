import importlib

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.hdf5 import get_train_loaders
from unet3d.config import load_config
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import get_model
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import get_logger, get_tensorboard_formatter
from unet3d.utils import get_number_of_learnable_parameters


def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    skip_train_validation = trainer_config.get('skip_train_validation', False)

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.get('tensorboard_formatter', None))

    # start training from scratch
    return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                            config['device'], loaders, trainer_config['checkpoint_dir'],
                            max_num_epochs=trainer_config['epochs'],
                            max_num_iterations=trainer_config['iters'],
                            validate_after_iters=trainer_config['validate_after_iters'],
                            log_after_iters=trainer_config['log_after_iters'],
                            eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                            logger=logger, tensorboard_formatter=tensorboard_formatter,
                            skip_train_validation=skip_train_validation)

def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    # TODO: The initialization parameters need to be updated
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')             # MultiStepLR
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)

        # lr = 0.0002
        # epoch: 0 - 10 lr=0.0002
        # epoch: 10 - 30 lr = 0.0002 * 0.2 = 0.00004
        # epoch: 30 - 60 lr = 0.00004 * 0.2 = 0.000008

def main():
    # Create main logger
    logger = get_logger('UNet3DTrainer')

    # Load and log experiment configuration
    config = load_config()  # Set DEFAULT_DEVICE and config file
    logger.info(config)     # Log configure from train_config_4d_input.yaml

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True  # Ensure the repeatability of the experiment
        torch.backends.cudnn.benchmark = False     # Benchmark mode improves the computation speed, but results in slightly different network feedforward results

    # Create the model
    model = get_model(config)
    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    # loaders: {'train': train_loader, 'val': val_loader}
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    # Create model trainer
    trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                              logger=logger)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()
