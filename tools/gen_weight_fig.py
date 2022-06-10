import json
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path as osp
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('dir', help='work dir path')
    parser.add_argument('dir_weight', help='work dir path')
    args = parser.parse_args()
    return args

args = parse_args()

colormap = cm.gist_ncar
root_dir = args.dir
label_file = osp.join(root_dir, 'label_info.json')
loss_file =  osp.join(root_dir, 'record_all_loss_info.json')
weight_file =  osp.join(args.dir_weight, 'weight_info.json')
fig_filename = root_dir.split('/')[-1] + '.jpg'
print(fig_filename)

def load_json(filename):    
    with open(filename,encoding='utf-8-sig', errors='ignore') as f:
        data = json.load(f, strict=False)
    return data

def norm(data, mean, std):
    data = (data-mean) / std
    return data

weight = load_json(weight_file)
label = load_json(label_file)
loss = load_json(loss_file)

num_classes = 10
if 'cifar10' in root_dir.split('/'):
    num_classes = 10
elif 'cifar100' in root_dir.split('/'):
    num_classes = 100

colors = [colormap(i) for i in np.linspace(0, 1, num_classes * 2+3)]
record_weight_class = {}
record_weight_class_false = {}
#epochs = 120
epochs = len(loss['0'])
weight_epochs = 120
print(num_classes)
for i in range(num_classes):
    record_weight_class[i] = np.zeros((0, weight_epochs))
    record_weight_class_false[i] = np.zeros((0, weight_epochs))
record_weight_false = np.zeros((0, weight_epochs))

for key, value in tqdm(label.items()):
    if value[0] == value[1]:
        record_weight_class[value[0]] = np.concatenate((record_weight_class[value[0]], np.array(weight[key])[None, :weight_epochs]), axis=0) 
    else:
        record_weight_class_false[value[0]]  = np.concatenate((record_weight_class_false[value[0]], np.array(weight[key])[None, :weight_epochs]), axis=0)

x = np.arange(0, epochs)
cols = 5
rows = 2
figsize = (3.2*cols, 2.5*rows)
axes = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

axes = trim_axs(axes, num_classes)
labels = [i for i in range(num_classes)]

x = np.arange(0, record_weight_class[0].shape[1])
iter_num = range(10) if num_classes == 10 else range(0, 100, 30)
iter_num = range(num_classes)
for index, i in enumerate(iter_num):
    axes[i].plot(x, record_weight_class[i].mean(axis=0), '-', label='cls_' + str(i), color=colors[index])
    axes[i].fill_between(x, record_weight_class[i].mean(axis=0) - record_weight_class[i].std(axis=0), record_weight_class[i].mean(axis=0) + record_weight_class[i].std(axis=0), alpha=0.2, color=colors[index])
    if record_weight_class_false[i].shape[0]:
        axes[i].plot(x, record_weight_class_false[i].mean(axis=0), '*', label='noise_' + str(i), color=colors[index + num_classes])
        axes[i].fill_between(x, record_weight_class_false[i].mean(axis=0) - record_weight_class_false[i].std(axis=0), record_weight_class_false[i].mean(axis=0) + record_weight_class_false[i].std(axis=0), alpha=0.2, color=colors[index + num_classes])

    axes[i].legend(fontsize=15)
    axes[i].set_xlabel('epoch', fontsize=15)
    axes[i].set_ylabel('weight', fontsize=15)
    axes[i].tick_params(axis='both', which='major', labelsize=15)

plt.savefig(osp.join(args.dir_weight, fig_filename), bbox_inches='tight')