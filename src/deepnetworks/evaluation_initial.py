'''
Title: Training of models (Classification problem)
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ===========================================================
# ============== MODULES & PATHS ============================
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
import models

_DATASETS = {'cifar10': (datasets.CIFAR10,2,10), 
			'mnist':(datasets.MNIST,0,10),
			'fashion':(datasets.FashionMNIST,0,10),
			'kmnist':(datasets.KMNIST,0,10),
			'emnist':(datasets.EMNIST,0,10)
			}

# ===========================================================
# ============== IO & MODEL FILES ===========================

def parse_boolean(value):
    out = True if value.lower() == 'true' else False
    return out

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)

parser.add_argument('--num-exps', default=1, type=int)
parser.add_argument('--test-batch-size', default=20, type=int)

parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', type=int, default=0)

parser.add_argument('--PATHdata', required=True)
parser.add_argument('--dataset', choices=['mnist','fashion','kmnist','emnist'])

parser.add_argument('--model', default="MLP", choices=['MLP', 'CNN'])
parser.add_argument('--layers', default=[100,100], nargs="+", type=int)
parser.add_argument('--bias', default=False, type=parse_boolean)

args = parser.parse_args()

# ==========================================================
# ===================== OUTPUTS ============================
outputs_folder = args.PATH + 'results/' + args.title + '/outputs_initial/'
if not os.path.isdir(outputs_folder): os.makedirs(outputs_folder)

# ===========================================================
# ===================== DATASET =============================
num_workers  = _DATASETS[args.dataset][1] 
num_classes = _DATASETS[args.dataset][2]

transform = transforms.ToTensor()

if args.dataset == 'emnist':
	test_data    = _DATASETS[args.dataset][0](root=args.PATHdata,split='mnist',train=False,transform=transform)
else:
	test_data    = _DATASETS[args.dataset][0](root=args.PATHdata,train=False,transform=transform)

test_loader  = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)

# ===========================================================
# ================= RUN EXPERIMENTS =========================

num_layers = len(args.layers)

for exp in range(args.num_exps):
	print('Exp '+str(exp+1) + '------------------------------')
	# =======================================================
	# ================= NETWORK MODEL  ======================
	cfg_model = {'in_size':784, 'hidden_sizes': args.layers, 'out_size':num_classes, 'bias': args.bias}

	model = models._LIST_MODELS[args.model](**cfg_model)

	if args.device_type == 'gpu':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True

	if (args.device_type == "gpu") and torch.has_cudnn:
		device = torch.device("cuda:{}".format(args.gpu_number))
	else:
		device = torch.device("cpu")

	model.to(device)

	# ===========================================================
	# =============== SAVE ACTIVATIONS ==========================

	model.eval()

	idx = 0
	save_data={}

	for data, target in test_loader:
		data   = data.to(device)
		target = target.to(device)
		output = model(data)

		for kk in output.keys():
			save_data[kk]=output[kk].detach().cpu().numpy()
		
		save_data['target']=target.cpu().numpy()
		np.save(outputs_folder + 'exp_' + str(exp+1) + '_batch_' + str(idx) +'.npy',save_data)
		
		idx += 1
		save_data={}