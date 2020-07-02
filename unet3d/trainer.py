import logging
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from unet3d.utils import DefaultTensorboardFormatter
from . import utils


class UNet3DTrainer:

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=1,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None, tensorboard_formatter=None, skip_train_validation=False):
        if logger is None:
            self.logger = utils.get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs               
        self.max_num_iterations = max_num_iterations      
        self.validate_after_iters = validate_after_iters  
        self.log_after_iters = log_after_iters             
        self.validate_iters = validate_iters               
        self.eval_score_higher_is_better = eval_score_higher_is_better
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations               
        self.num_epoch = num_epoch                         
        self.skip_train_validation = skip_train_validation 


    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            self.logger.info(
                # Iterations: 0-max_num_iterations, Batch: 0-len(train_loader), Epoch: [1-max_num_epochs/max_num_epochs]
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs}]')

            input, target = self._split_training_batch(t)

            # 前向传播，返回output和计算出的loss ---> DiceLoss(output, target)
            output, loss = self._forward_pass(input, target)

            train_losses.update(loss.item(), 1) # the second parameter is batch_size

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                else:
                    self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(eval_score.item(), 1)

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                self._log_images(input, target, output)

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                self.logger.info(f'Validation iteration {i}')

                input, target = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), 1)

                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), 1)

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            self._log_stats('val', val_losses.avg, val_scores.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                input = input.float()
                return input.to(self.device)

        t = _move_to_device(t)
        input, target = t
        return input, target

    def _forward_pass(self, input, target):
        # forward pass
        output = self.model(input)

        # print(output.detach().numpy().shape)

        # compute the loss
        loss = self.loss_criterion(output, target)


        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
