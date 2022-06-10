#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import os
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .utils import save_json
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_CYCLE_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.label_info_file = os.path.join(self.arg.work_dir, 'label_info.json')
        self.record_all_loss_info_file = os.path.join(self.arg.work_dir, 'record_all_loss_info.json')
        self.best_acc = 0
        self.lr_ratio = 0
        if os.path.isdir(os.path.join('runs', self.arg.work_dir)):
            print('log_dir: ', os.path.join('runs', self.arg.work_dir), 'already exist')
            # answer = input('delete it? y/n:')
            answer = 'y'
            if answer == 'y':
                shutil.rmtree(os.path.join('runs', self.arg.work_dir))
                print('Dir removed: ', os.path.join('runs', self.arg.work_dir))
                # input('Refresh the website of tensorboard by pressing any keys')
            else:
                print('Dir not removed: ', os.path.join('runs', self.arg.work_dir))
        self.train_writer = SummaryWriter(os.path.join('runs', self.arg.work_dir), 'train')
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        # if 'debug' not in self.arg.train_feeder_args:
        #     self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False, pin_memory=True)
            if 'clothing_dataset' in self.arg.feeder:
                self.data_loader['small_bs_train'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker * torchlight.ngpu(
                        self.arg.device),
                    drop_last=False, pin_memory=True)
            self.data_loader['meta_train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_meta_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False, pin_memory=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device), pin_memory=True)

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] >= self.arg.step[-1]:
                if self.meta_info['epoch'] % self.arg.cycle_step == 0:
                    self.lr_ratio = 1 - self.lr_ratio 
                # ratio = (self.meta_info['epoch'] % self.arg.cycle_step) / self.arg.cycle_step
                lr = self.lr_ratio * 1e-2 + (1 - self.lr_ratio) * 1e-3
            else:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr
        

    def save_dataset(self):
        np.save(os.path.join(self.arg.work_dir, 'train_data.npy'), self.data_loader['train'].dataset.train_data)
        np.save(os.path.join(self.arg.work_dir, 'train_labels.npy'), self.data_loader['train'].dataset.train_labels)
        np.save(os.path.join(self.arg.work_dir, 'true_labels.npy'), self.data_loader['train'].dataset.true_labels)
        
        np.save(os.path.join(self.arg.work_dir,'meta_train_data.npy'), self.data_loader['meta_train'].dataset.train_data)
        np.save(os.path.join(self.arg.work_dir,'meta_train_labels.npy'), self.data_loader['meta_train'].dataset.train_labels)
        np.save(os.path.join(self.arg.work_dir,'meta_true_labels.npy'), self.data_loader['meta_train'].dataset.true_labels)

        np.save(os.path.join(self.arg.work_dir,'test_data.npy'), self.data_loader['test'].dataset.test_data)
        np.save(os.path.join(self.arg.work_dir,'test_labels.npy'), self.data_loader['test'].dataset.test_labels)

    def save(self):
        save_json(self.label_info_file, self.label_info)
        saverecord = {}
        for key, value in self.record.items():
            saverecord[key] = value.cpu().detach().numpy().tolist() 
        save_json(self.record_all_loss_info_file, saverecord)

    def check_save(self):
        Feeder_Check = import_class(self.arg.feeder_check)
        self.arg.check_train_feeder_args['root'] = self.arg.pretrain_dir
        self.data_loader['check_save_train'] = torch.utils.data.DataLoader(
                dataset=Feeder_Check(**self.arg.check_train_feeder_args),
                batch_size=self.arg.batch_size // 8,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False, pin_memory=True)
        self.compare(self.data_loader['train'].dataset, self.data_loader['check_save_train'].dataset)
        self.arg.check_meta_train_feeder_args['root'] = self.arg.pretrain_dir
        self.data_loader['check_save_meta_train'] = torch.utils.data.DataLoader(
            dataset=Feeder_Check(**self.arg.check_meta_train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device),
            drop_last=False, pin_memory=True)
        self.compare(self.data_loader['meta_train'].dataset, self.data_loader['check_save_meta_train'].dataset)
        if self.arg.test_feeder_args:
            self.arg.check_test_feeder_args['root'] = self.arg.pretrain_dir
            self.data_loader['check_save_test'] = torch.utils.data.DataLoader(
                dataset=Feeder_Check(**self.arg.check_test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device), pin_memory=True)
            self.compare(self.data_loader['test'].dataset, self.data_loader['check_save_test'].dataset, 'test')
        print('check save complete')


    def compare(self, dataset, dataset_save, mode = 'train'):
        if mode == 'train':
            datas = dataset.train_data
            train_labels = dataset.train_labels
            true_labels = dataset.true_labels            
        else:
            datas = dataset.test_data
            train_labels = dataset.test_labels
            true_labels = dataset.test_labels
        
        datas_save = dataset_save.data        
        train_labels_save = dataset_save.labels        
        true_labels_save = dataset_save.true_labels

        if (datas - datas_save).sum() != 0:
            print('datas save error')
        
        if (train_labels - train_labels_save).sum() != 0:
            print('train_labels save error')

        if (true_labels - true_labels_save).sum() != 0:
            print('true_labels save error')

        
    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        if self.meta_info['epoch'] >= self.arg.step[-1]:
            if 'clothing_dataset' not in self.arg.feeder:
                loader = self.data_loader['check_save_train']
            else:
                loader = self.data_loader['small_bs_train']
        else:
            loader = self.data_loader['train']

        loss_value = []

        for data, label, index, _ in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            with torch.no_grad():
                for idx in range(index.shape[0]):
                    cost_v = torch.reshape(loss, (len(loss), 1))
                    self.record[index[idx].item()] = torch.cat((self.record[index[idx].item()], cost_v.data[idx]))

            # backward
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_writer.add_scalar('train/loss', self.iter_info['loss'], self.meta_info['iter'])
            self.train_writer.add_scalar('train/lr', self.lr, self.meta_info['iter'])
            self.train_writer.add_scalar('train/epoch', self.meta_info['epoch'], self.meta_info['iter'])

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            self.record = {}
            self.label_info = {}
            loader = self.data_loader['train']
            if 'clothing_dataset' not in self.arg.feeder:
                self.save_dataset()
                self.check_save()
                print('initialize label info')
                false_value = [0 for idx in range(self.arg.model_args['num_classes'])]
                true_value = [0 for idx in range(self.arg.model_args['num_classes'])]
                for batch_idx, (inputs, targets, index, true_targets) in tqdm(enumerate(loader)):
                    with torch.no_grad():
                        for idx in range(index.shape[0]):
                            self.label_info[index[idx].item()] = [targets[idx].item(), true_targets[idx].item()]
                            if targets[idx].item() == true_targets[idx].item():
                                true_value[targets[idx].item()] += 1
                            else:
                                false_value[targets[idx].item()] += 1
                percent = [false_value[idx] / (false_value[idx] + true_value[idx]) for idx in range(self.arg.model_args['num_classes'])]
                print('noise rate in per class', percent, '\n')

            for i in range(len(loader.dataset)):
                self.record[i] = torch.zeros((0)).cuda()

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')
            self.save()
            # save model
            filename = 'last_epoch_model.pt'
            self.io.save_model(self.model, filename)
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test_conf()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    def test(self, evaluation=True):

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

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        # check_feeder
        parser.add_argument('--feeder_check', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--pretrain_dir', default='', type=str, help='checkfile save dir')
        parser.add_argument('--check_train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--check_meta_train_feeder_args', action=DictAction, default=dict(), help='the arguments of meta data loader for training')
        parser.add_argument('--check_test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--cycle_step', type=int, default=15, help='weight decay for optimizer')

        return parser
