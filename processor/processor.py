#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import random
import os.path as osp
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

from .io import IO
import subprocess
class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()
        self.label = []
        

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.set_seed(self.arg.seed)

    def get_gpu_memory_map(self):
        """Get the current gpu usage.

        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    def load_optimizer(self):
        pass

    def test_conf(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label, index, _ in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.mean().item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)

        

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
        rank = self.result.argsort()
        rank = rank[:, -1]
        plt.figure(figsize=(5,5))
        confusion = confusion_matrix(self.label, rank)
        print(confusion[0, :].sum())
        confusion = confusion / confusion[0, :].sum()
        confusion = 100 * confusion
        plt.matshow(confusion, cmap=plt.cm.Greens) 
        plt.colorbar()
        # for i in range(len(confusion)): 
        #     for j in range(len(confusion)):
        #         string = str(round(confusion[i,j],1))
        #         plt.annotate(string, xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=8)
        plt.title('Ours', fontsize=18)
        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)         
        plt.savefig(osp.join(self.arg.work_dir, 'confusion.jpg'), bbox_inches='tight')

    def save_model(self, model, name):
        model_path = '{}/{}'.format(self.work_dir, name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sensinet_state_dict': self.sensinet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_sensinet_state_dict': self.optimizer_sensinet.state_dict(),
            'meta_epoch': self.meta_info['epoch'],
            'meta_iter': self.meta_info['iter']
            }, model_path)
        self.print_log('The model has been saved as {}.'.format(model_path))

    def load_weights(self):
        # self.arg.phase = 'test'
        # self.arg.weights = osp.join(self.arg.work_dir, 'best_model.pt')
        if self.arg.weights:
            checkpoint = torch.load(self.arg.weights)
            self.model.load_state_dict(checkpoint)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.sensinet.load_state_dict(checkpoint['sensinet_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.optimizer_sensinet.load_state_dict(checkpoint['optimizer_sensinet_state_dict'])
            # self.arg.start_epoch = checkpoint['meta_epoch']
            # self.meta_info['meta_iter'] = checkpoint['meta_iter']

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        hit_top_k_cls = []
        hit_top_k_cls_num = []
        for cls in range(self.arg.model_args['num_classes']):
            hit_top_k_cls.append([(l in rank[i, -k:]) * (l == cls) for i, l in enumerate(self.label)])
            hit_top_k_cls_num.append([l == cls for i, l in enumerate(self.label)])
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy_cls = [sum(hit_top_k_cls[i]) * 1.0 / sum(hit_top_k_cls_num[i]) for i in range(self.arg.model_args['num_classes'])]
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            filename = 'best_model.pt'
            self.io.save_model(self.model, filename)

        self.train_writer.add_scalar('accuracy/test_acc', 100 * accuracy, self.meta_info['epoch'])
        for i in range(self.arg.model_args['num_classes']):
            self.train_writer.add_scalar('accuracy/test_acc_cls_' + str(i), 100 * accuracy_cls[i], self.meta_info['epoch'])

        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        self.io.print_log('\tBest accuracy Top{}: {:.2f}%'.format(k, 100 * self.best_acc))

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True, pin_memory=True)
            self.data_loader['meta_train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True, pin_memory=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device), pin_memory=True)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--train_meta_feeder_args', action=DictAction, default=dict(), help='the arguments of meta data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--warmup_epoch', type=int, default=0, help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--alpha_factor', type=float, default=0.1, help='initial learning rate')

        parser.add_argument('--seed', type=int, default=1, help='the model will be used')
        #endregion yapf: enable

        return parser
