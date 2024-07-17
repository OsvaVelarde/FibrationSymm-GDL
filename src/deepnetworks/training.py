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
parser.add_argument('--num-train-epochs', default=30, type=int)
parser.add_argument('--train-batch-size', default=20, type=int)
parser.add_argument('--test-batch-size', default=20, type=int)
parser.add_argument('--lr-rate', default=0.01, type=float, help='learning rate')

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
outputs_folder = args.PATH  + 'results/' + args.title + '/outputs_final/'
training_folder = args.PATH + 'results/' + args.title + '/training/'
weights_folder = args.PATH  + 'pretrained_models/' + args.title + '/'

if not os.path.isdir(outputs_folder): os.makedirs(outputs_folder)
if not os.path.isdir(training_folder): os.makedirs(training_folder)
if not os.path.isdir(weights_folder): os.makedirs(weights_folder)

# ===========================================================
# ===================== DATASET =============================
num_workers  = _DATASETS[args.dataset][1] 
num_classes = _DATASETS[args.dataset][2]

transform = transforms.ToTensor()

if args.dataset == 'emnist':
	train_data   = _DATASETS[args.dataset][0](root=args.PATHdata,split='mnist',train=True, download=False, transform=transform)
	test_data    = _DATASETS[args.dataset][0](root=args.PATHdata,split='mnist',train=False,transform=transform)
else:
	train_data   = _DATASETS[args.dataset][0](root=args.PATHdata, train=True, download=False, transform=transform)
	test_data    = _DATASETS[args.dataset][0](root=args.PATHdata,train=False,transform=transform)

train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=num_workers)
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

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_rate)

	# =======================================================
	# ==================== TRAINING =========================

	performance = np.zeros((args.num_train_epochs,3))

	for epoch in range(args.num_train_epochs):
		# ---------------------------------------------------
		model.train()
		train_loss = 0.0

		for data, target in train_loader:
			data   = data.to(device)
			target = target.to(device)

			optimizer.zero_grad()
			output = model(data)

			loss = criterion(output['h'+str(num_layers+1)], target)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()*data.size(0)
		    
		train_loss = train_loss/len(train_loader.dataset)

		# -------------------------------------------------------
		model.eval()

		test_loss = 0.0
		accs = []

		for data, target in test_loader:
			data   = data.to(device)
			target = target.to(device)
			output = model(data)

			loss = criterion(output['h'+str(num_layers+1)], target)
			test_loss += loss.item()*data.size(0)

			_, pred = torch.max(output['h'+str(num_layers+1)], 1)
			correct = np.squeeze(pred.eq(target.data.view_as(pred)))
			acc = correct.sum() / len(target)
			accs.append(acc.item())

		test_loss = test_loss/len(test_loader.dataset)
		test_acc = np.mean(accs)

		# -------------------------------------------------------

		print('Epoch '+ str(epoch) + ' Acc: ' + str(test_acc))

		performance[epoch,0] = epoch+1
		performance[epoch,1] = train_loss
		performance[epoch,2] = test_acc

	# -------------------------------------------------------
	with open(training_folder + 'exp_' + str(exp+1) + '.npy', 'wb') as f:
		np.save(f, performance)

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

	# ===========================================================
	# =================== SAVE WEIGHTS ==========================

	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	torch.save(model.state_dict(), weights_folder + 'exp_' + str(exp+1) + '.pt')