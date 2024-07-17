'''
Title: Fibrations of graph
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ===========================================================
# ============== MODULES & PATHS ============================
import argparse
import os

import numpy as np 
import pandas as pd
import json

_DATASETS = {'cifar10': 10, 
			'mnist':10,
			'fashion':10,
			'kmnist':10,
			'emnist':10
			}

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)
parser.add_argument('--num-exps', default=1, type=int)
parser.add_argument('--layers', default=[100,100], nargs="+", type=int)
parser.add_argument('--dataset', choices=['mnist','fashion','kmnist','emnist'])
parser.add_argument('--epoch', choices=['initial','final'])

args = parser.parse_args()

num_classes = _DATASETS[args.dataset]
num_layers = len(args.layers) + 1

# ===========================================================
# ===========================================================

rows = []
for idx_ll, ll in enumerate(args.layers):
	rows.extend(['L'+str(idx_ll+1)+'_'+str(nn) for nn in range(ll)])
	
rows.extend(['L'+str(idx_ll+2) + '_' + str(nn) for nn in range(num_classes)])
rows.append('target')

# rows = [L1_00, L1_01, ..., L1_XX, L2_00, L2_01, ..., L2_YY, ..., L3_00, ..., L3_09, target]
# columns = [1,2,3,...,N, mean, mean_00, mean_01, mean_02, ..., mean_09]

for exp in range(args.num_exps):

	datafolder = args.PATH + 'results/' + args.title + '/outputs_' +  args.epoch + '/exp_' + str(exp+1) + '/'
	files = os.listdir(datafolder)

	data = {ii+1:[] for ii in range(num_layers)}
	targets = []

	for ff in files:
		data_batch = np.load(datafolder + ff,allow_pickle=True)

		for kk in data.keys():
			data[kk].append(data_batch[()]['h'+str(kk)])

		targets.append(data_batch[()]['target'])

	for kk in data.keys():
		data[kk] = np.concatenate(data[kk],axis=0).T

	targets = np.concatenate(targets,axis=0)
	data = np.concatenate(list(data.values()), axis=0)
	data = np.vstack((data,targets))

	data = pd.DataFrame(data, index=rows)

	data['mean'] = data.mean(axis=1)

	for _class in range(num_classes):
		samples_in_class = data.loc[:,data.loc['target'] == _class]
		data['mean_'+str(_class)] = samples_in_class.mean(axis=1)

	data.to_csv(args.PATH + 'results/' + args.title + '/outputs_' + args.epoch + '/exp_' + str(exp+1) + '.csv')

# ===========================================================
# ===========================================================