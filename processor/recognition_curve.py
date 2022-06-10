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
import shutil

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from tensorboardX import SummaryWriter
from .processor import Processor
import torch.nn.functional as F
import json
from .utils import save_json
import higher
import time
from datetime import datetime 

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

class REC_Curve_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.arg.curvenet_args['num_cls'] = self.arg.model_args['num_classes']
        self.arg.curvenet_args['input_dim'] = self.arg.staend_epoch - self.arg.stastart_epoch
        self.curvenet = self.io.load_model(self.arg.curvenet, **(self.arg.curvenet_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.label_info_file = os.path.join(self.arg.work_dir, 'label_info.json')
        self.record_all_loss_info_file = os.path.join(self.arg.work_dir, 'record_all_loss_info.json')
        self.weight_info_file = os.path.join(self.arg.work_dir, 'weight_info.json')
        self.loss_info_file = os.path.join(self.arg.work_dir, 'loss_info.json')    

        if 'WideResNet' in self.arg.model:
            self.wider = True
        else:
            self.wider = False    

        self.best_acc = 0
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

    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev)
        #self.meta_model = self.meta_model.to(self.dev)
        self.curvenet = self.curvenet.to(self.dev)
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        # model parallel
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)
            self.curvenet = nn.DataParallel(self.curvenet, device_ids=self.gpus)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.arg.train_feeder_args['root'] = self.arg.pretrain_dir
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False, pin_memory=True)
            self.arg.train_meta_feeder_args['root'] = self.arg.pretrain_dir
            self.data_loader['meta_train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_meta_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False, pin_memory=True)
        if self.arg.test_feeder_args:
            self.arg.test_feeder_args['root'] = self.arg.pretrain_dir
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device), pin_memory=True)
        
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

        if self.arg.optimizer_curvenet =='SGD':
            self.optimizer_curvenet = optim.SGD(
                self.curvenet.parameters(),
                lr=self.arg.curvenet_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.curvenet_weight_decay)
        elif self.arg.optimizer_curvenet == 'Adam':
            self.optimizer_curvenet = optim.Adam(
                    self.curvenet.parameters(),
                    lr=self.arg.curvenet_lr,
                    weight_decay=self.arg.curvenet_weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.meta_info['epoch'] < self.arg.warmup_epoch:
            self.lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warmup_epoch
        elif self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] < 5:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / 5
            else:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def adjust_curvenet_lr(self):
        if self.arg.optimizer_curvenet == 'SGD' and self.arg.step:
            curvenet_lr = self.arg.curvenet_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer_curvenet.param_groups:
                param_group['lr'] = curvenet_lr
            self.curvenet_lr = curvenet_lr
        else:
            self.curvenet_lr = self.arg.curvenet_lr
    
    def froze(self, fmodel):
        self.gradient_steps = self.arg.fro_stage
        if self.arg.fro_stage == 0:
            return

        if not self.wider:
            fmodel.bn1.eval()
            for m in [fmodel.conv1,
                    fmodel.bn1]:
                for param in m.parameters():                    
                    param.requires_grad = False

            for i in range(1, self.arg.fro_stage):
                m = getattr(fmodel, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        else:
            for m in [fmodel.conv1]:
                for param in m.parameters():                    
                    param.requires_grad = False

            for i in range(1, self.arg.fro_stage):
                m = getattr(fmodel, f'block{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def drw(self):
        if self.arg.drw:
            idx = 1 if self.meta_info['epoch'] > self.arg.step[0] else 0
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.dev)
            self.epoch_info['per_cls_weights'] =  per_cls_weights
        else:
            self.per_cls_weights = None

    def train(self):
        self.model.train()
        self.adjust_lr()
        self.adjust_curvenet_lr()
        self.drw()
        loader = self.data_loader['train']
        train_meta_loader = self.data_loader['meta_train']
        train_meta_loader_iter = iter(train_meta_loader)

        train_loss = 0
        meta_loss = 0

        loss_value = []
        clean_label_weight_epoch = torch.zeros((0)).cuda()
        noise_label_weight_epoch = torch.zeros((0)).cuda()
        cls_label_weight_epoch = [torch.zeros((0)).cuda() for i in range(self.arg.model_args['num_classes'])]
        cls_label_weight_epoch_all = [torch.zeros((0)).cuda() for i in range(self.arg.model_args['num_classes'])]
        start_time = datetime.now() 

        for data, label, index, _ in loader:

            clean_label_weight_iter = torch.zeros((0)).cuda()
            noise_label_weight_iter = torch.zeros((0)).cuda()
            cls_label_weight_iter = [torch.zeros((0)).cuda() for i in range(self.arg.model_args['num_classes'])]

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # train the curvenet
            if self.meta_info['epoch'] < self.arg.step[0]:
                with higher.innerloop_ctx(self.model, self.optimizer) as (fmodel, diffopt):
                    # frozen bottom layers for saving time
                    self.froze(fmodel)

                    # virtual train
                    output = fmodel(data)
                    cost = F.cross_entropy(output, label, reduction='none')
                    cost_v = torch.reshape(cost, (len(cost), 1))

                    with torch.no_grad():
                        v_net_in = [self.record[index[idx].item()].unsqueeze(0) for idx in range(index.shape[0])]
                        v_net_in = torch.cat(v_net_in, dim=0)
                        # flags = [self.mean_flag[index[idx].item()] for idx in range(index.shape[0])]
                        # flags = torch.cat(flags, dim=0)

                    # v_lambda = self.curvenet(v_net_in, label, flags)
                    v_lambda = self.curvenet(v_net_in, label)
                    for idx in range(index.shape[0]): 
                        v_lambda[idx] = self.alpha * v_lambda[idx] + (1 - self.alpha) * self.weight_info[index[idx].item()][-1]
                    l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
                    diffopt.step(l_f_meta)
                    
                    # meta train
                    l_g_meta = 0
                    try:
                        inputs_val, targets_val, _, _ = next(train_meta_loader_iter)
                    except StopIteration:
                        train_meta_loader_iter = iter(train_meta_loader)
                        inputs_val, targets_val, _, _ = next(train_meta_loader_iter)
                    
                    targets_val = targets_val.type(torch.LongTensor)
                    inputs_val, targets_val = inputs_val.to(self.dev), targets_val.to(self.dev)
                    y_g_hat = fmodel(inputs_val)
                    loss = F.cross_entropy(y_g_hat, targets_val, reduction='none')
                    l_g_meta += loss.mean()
                    
                    # optimize the curvenet
                    t1 = time.time()
                    self.optimizer_curvenet.zero_grad()
                    l_g_meta.backward()
                    self.optimizer_curvenet.step()
                    meta_loss += l_g_meta.item()
                    t2 = time.time()
                    
                    self.iter_info['meta time'] = '{:.6f}'.format((t2 - t1)*1000)
                    self.train_writer.add_scalar('Meta/loss', l_g_meta, self.meta_info['iter'])

            # Actual Train
            output = self.model(data)
            cost_w = F.cross_entropy(output, label, reduction='none', weight=self.per_cls_weights)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))            

            with torch.no_grad():
                v_net_in = [self.record[index[idx].item()].unsqueeze(0) for idx in range(index.shape[0])]
                v_net_in = torch.cat(v_net_in, dim=0)
                # flags = [self.mean_flag[index[idx].item()] for idx in range(index.shape[0])]
                # flags = torch.cat(flags, dim=0)

                # w_new = self.curvenet(v_net_in, label, flags)
                w_new = self.curvenet(v_net_in, label)

                # record weight information
                for idx in range(index.shape[0]): 
                    if self.weight_info[index[idx].item()].shape[0] != 0:
                        w_new.data[idx] = self.alpha * w_new.data[idx] + (1 - self.alpha) * self.weight_info[index[idx].item()][-1] 
                    cls_label_weight_epoch_all[self.label_info[index[idx].item()][0]] = torch.cat((cls_label_weight_epoch[self.label_info[index[idx].item()][0]], w_new.data[idx]))    
                    if self.label_info[index[idx].item()][0] == self.label_info[index[idx].item()][1]:
                        clean_label_weight_iter = torch.cat((clean_label_weight_iter, w_new.data[idx]))
                        clean_label_weight_epoch = torch.cat((clean_label_weight_epoch, w_new.data[idx]))
                        cls_label_weight_iter[self.label_info[index[idx].item()][0]] = torch.cat((cls_label_weight_iter[self.label_info[index[idx].item()][0]], w_new.data[idx]))
                        cls_label_weight_epoch[self.label_info[index[idx].item()][0]] = torch.cat((cls_label_weight_epoch[self.label_info[index[idx].item()][0]], w_new.data[idx]))
                    else:
                        noise_label_weight_iter = torch.cat((noise_label_weight_iter, w_new.data[idx]))
                        noise_label_weight_epoch = torch.cat((noise_label_weight_epoch, w_new.data[idx]))
                    
                    self.weight_info[index[idx].item()] = torch.cat((self.weight_info[index[idx].item()], w_new.data[idx]))
                    self.record_meta[index[idx].item()] = torch.cat((self.record_meta[index[idx].item()], cost_v.data[idx]))

            loss = torch.sum(cost_v * w_new)/len(cost_v)
            train_loss += loss.item()
            
            # backward
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()           

            # statistics
            if clean_label_weight_iter.shape[0] != 0:
                self.train_writer.add_scalar('Meta/true_weight', clean_label_weight_iter.sum() / (clean_label_weight_iter.shape[0] + 1e-6), self.meta_info['iter'])
            if noise_label_weight_iter.shape[0] != 0:
                self.train_writer.add_scalar('Meta/false_weight', noise_label_weight_iter.sum() / (noise_label_weight_iter.shape[0] + 1e-6), self.meta_info['iter'])

            for i in range(self.arg.model_args['num_classes']):
                if cls_label_weight_iter[i].shape[0] != 0:
                    self.train_writer.add_scalar('MetaImblance_iter/num_cls_' + str(i), cls_label_weight_iter[i].sum() / (cls_label_weight_iter[i].shape[0] + 1e-6), self.meta_info['iter'])

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_writer.add_scalar('train/loss', self.iter_info['loss'], self.meta_info['iter'])
            self.train_writer.add_scalar('train/lr', self.lr, self.meta_info['iter'])
            self.train_writer.add_scalar('train/alpha', self.alpha, self.meta_info['iter'])
            self.train_writer.add_scalar('train/curvenet_lr', self.curvenet_lr, self.meta_info['iter'])
            self.train_writer.add_scalar('train/epoch', self.meta_info['epoch'], self.meta_info['iter'])
        
        self.train_writer.add_scalar('Meta/true_weight_epoch', clean_label_weight_epoch.sum() / (clean_label_weight_epoch.shape[0] + 1e-6), self.meta_info['epoch'])
        self.train_writer.add_scalar('Meta/false_weight_epoch', noise_label_weight_epoch.sum() / (noise_label_weight_epoch.shape[0] + 1e-6), self.meta_info['epoch'])
        for i in range(self.arg.model_args['num_classes']):
            self.train_writer.add_scalar('MetaImblance_epoch/num_cls_' + str(i), cls_label_weight_epoch[i].sum() / (cls_label_weight_epoch[i].shape[0] + 1e-6), self.meta_info['epoch'])

        end_time=datetime.now() 
        self.epoch_info['Time consumption'] = (end_time-start_time).seconds
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def save(self):        
        saverecord = {}
        record_meta = {}
        weight_info = {}        
        for key, value in self.record_meta.items():
            record_meta[key] = value.cpu().detach().numpy().tolist()
            weight_info[key] = self.weight_info[key].cpu().detach().numpy().tolist()
        save_json(self.label_info_file, self.label_info)
        save_json(self.loss_info_file, record_meta)
        save_json(self.weight_info_file, weight_info)
        save_json(self.record_all_loss_info_file, saverecord)

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

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            self.record = {}
            self.saverecord = {}
            self.label_info = {}
            self.record_meta = {}
            self.weight_info = {}
            self.pre_loss_cls = {}
            self.mean_flag = {}

            for i in range(self.arg.model_args['num_classes']):
                self.pre_loss_cls[i] = []

            # sample statistics
            loader = self.data_loader['train']
            with open(os.path.join(self.arg.pretrain_dir,'label_info.json'), encoding='utf-8-sig', errors='ignore') as f:
                label_info = json.load(f, strict=False)
                false_value = [0 for idx in range(self.arg.model_args['num_classes'])]
                true_value = [0 for idx in range(self.arg.model_args['num_classes'])]

                for key, value in label_info.items():
                    self.label_info[int(key)] = [int(value[0]), int(value[1])]
                    if int(value[0]) == int(value[1]):
                        true_value[int(value[0])] += 1
                    else:
                        false_value[int(value[1])] += 1
                self.cls_num_list = [false_value[idx] + true_value[idx] for idx in range(self.arg.model_args['num_classes'])]
                percent = [false_value[idx] / (false_value[idx] + true_value[idx]) for idx in range(self.arg.model_args['num_classes'])]
                print(true_value, '\n')
                print(false_value, '\n')
                print(percent, '\n')
                print(self.cls_num_list)

            for i in range(len(loader.dataset)):
                self.record_meta[i] = torch.zeros((0)).cuda()
                self.weight_info[i] = torch.zeros((1)).cuda() + self.arg.start_weight

            # load loss curves of samples
            with open(os.path.join(self.arg.pretrain_dir,'record_all_loss_info.json'), encoding='utf-8-sig', errors='ignore') as f:
                record_read = json.load(f, strict=False)
                with torch.no_grad():
                    for key, value in record_read.items():
                        self.pre_loss_cls[self.label_info[int(key)][0]].append(value)
                        self.record[int(key)] = torch.from_numpy(np.array(value)).float().cuda()
            
            for i in range(self.arg.model_args['num_classes']):
                self.pre_loss_cls[i] = torch.from_numpy(np.mean(np.array(self.pre_loss_cls[i]), axis=0)).to(self.dev)
            
            # Normalize loss curves
            for key, value in self.record.items():
                self.record[int(key)] = self.record[int(key)] - self.pre_loss_cls[self.label_info[int(key)][0]]

            record_all = [value.unsqueeze(0) for key, value in self.record.items()]
            record_all = torch.cat(record_all, dim=0)

            record_all = record_all[:, self.arg.stastart_epoch:self.arg.staend_epoch]

            for key, value in self.record.items():
                self.saverecord[key] = value.cpu().detach().numpy().tolist() 

            for i in range(len(loader.dataset)):        
                self.record[i] = self.record[i][self.arg.stastart_epoch:self.arg.staend_epoch].to(torch.float)

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch
                self.alpha = 1
                if self.meta_info['epoch'] >= self.arg.step[0]:
                    self.alpha = 0

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)  or ((epoch + 1) in self.arg.step):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')
            self.save()
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

        # curvenet
        parser.add_argument('--fro_stage', default=0, type=int, help='the curvenet will be used')
        parser.add_argument('--curvenet', default=None, help='the curvenet will be used')
        parser.add_argument('--curvenet_args', action=DictAction, default=dict(), help='the arguments of curvenet')
        parser.add_argument('--curvenet_lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--curvenet_weight_decay', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--stastart_epoch', default=5, type=int, help='number of total epochs to run')
        parser.add_argument('--staend_epoch', default=299, type=int, help='number of total epochs to run')
        parser.add_argument('--pretrain_dir', default='', type=str, help='number of total epochs to run')
        parser.add_argument('--optimizer_curvenet', default='Adam', help='type of optimizer')
        parser.add_argument('--start_weight', default=1.0, type=float, help='type of optimizer') 
        parser.add_argument('--smooth', default=1, type=int, help='type of optimizer') 
        parser.add_argument('--drw', default=False, type=bool, help='type of optimizer') 

        return parser
