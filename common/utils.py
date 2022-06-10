from __future__ import absolute_import, division

import os
import torch
import numpy as np
from tensorboardX import SummaryWriter


# self define tools
class Summary(object):
    def __init__(self, directory):
        self.directory = directory
        self.epoch = 0
        self.writer = None
        self.phase = 0
        self.train_iter_num = 0
        self.train_realpose_iter_num = 0
        self.train_fakepose_iter_num = 0
        self.test_iter_num = 0
        self.test_MPI3D_iter_num = 0

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def summary_train_iter_num_update(self):
        self.train_iter_num = self.train_iter_num + 1

    def summary_train_realpose_iter_num_update(self):
        self.train_realpose_iter_num = self.train_realpose_iter_num + 1

    def summary_train_fakepose_iter_num_update(self):
        self.train_fakepose_iter_num = self.train_fakepose_iter_num + 1

    def summary_test_iter_num_update(self):
        self.test_iter_num = self.test_iter_num + 1

    def summary_test_MPI3D_iter_num_update(self):
        self.test_MPI3D_iter_num = self.test_MPI3D_iter_num + 1

    def summary_epoch_update(self):
        self.epoch = self.epoch + 1

    def summary_phase_update(self):
        self.phase = self.phase + 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterList(object):
    """Computes and stores the average and current value"""

    def __init__(self, total_meter):
        self.total_meter = total_meter
        self.dict = {}
        for i in range(self.total_meter):
            self.dict[str(i)] = AverageMeter()

    def reset(self):
        for i in range(self.total_meter):
            self.dict[str(i)].reset()

    def update(self, val_lst, n=1):
        for i, val in enumerate(val_lst):
            self.dict[str(i)].update(val, n)


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epochs - self.decay_epoch)


class LambdaLRIter():
    def __init__(self, epochs, offset, decay_epoch, loader_length):
        self.total_iter = epochs * loader_length
        self.offset_iter = offset * loader_length
        self.decay_iter = decay_epoch * loader_length

    def step(self, current_iter):
        return 1.0 - max(0, current_iter + self.offset_iter - self.decay_iter) / (self.total_iter - self.decay_iter)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
