'''
Title: Fibrations of graph
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ============================================================
# ============== MODULES & PATHS =============================

import argparse
import pandas as pd
from numpy.random import normal
from numpy import mean, std

import matplotlib.pyplot as plt

# ============================================================
# ============== IO & MODEL FILES ============================

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)
parser.add_argument('--epoch', choices=['initial','final'])

args = parser.parse_args()

# ============================================================

df_class = pd.read_csv(args.PATH + 'results/' + args.title + '/num_fibers_' + args.epoch + '_per_class.txt', sep=' ', header=None, names=['exp','class','layer','fibers'])
df_gral  = pd.read_csv(args.PATH + 'results/' + args.title + '/num_fibers_' + args.epoch + '_gral.txt', sep=' ', header=None, names=['exp','class','layer','fibers'])
df = pd.concat([df_class,df_gral],axis=0,ignore_index=True)
gk = df.groupby(['class','layer'])

# ============================================================

fig_mean, axs_mean = plt.subplots(1,1, figsize=(8,8))
fig_class, axs_class = plt.subplots(2,5, figsize=(20,8), sharex=True, sharey=True)
std_plot = 0.01

axs_mean.set_xlabel('Layer')
axs_mean.set_ylabel('Num of fibers')
axs_mean.set_xlim([0.5,2.5])
axs_mean.set_xticks([1,2])
axs_mean.set_xticklabels(['L1', 'L2'])

fig_class.supxlabel('Layer', ha='center', va='center')
fig_class.supylabel('Num of fibers', ha='center', va='center', rotation='vertical')

color_scatter = {1:'blue',2:'red'}

# ============================================================

for name, gg in gk:
	idx_ll = int(name[1][1])
	values = gg['fibers'].tolist()

	x = normal(idx_ll,std_plot,len(values))
	mean_values = mean(values)
	std_values = std(values)

	if name[0]=='-':
		axs_mean.scatter(x,values,c=color_scatter[idx_ll])
		axs_mean.errorbar(idx_ll, mean_values, yerr=std_values, c=color_scatter[idx_ll])
	else:
		_class=int(name[0])
		i = _class//5
		j = _class %5
		axs_class[i,j].scatter(x,values,c=color_scatter[idx_ll])
		axs_class[i,j].errorbar(idx_ll, mean_values, yerr=std_values, c=color_scatter[idx_ll])
		axs_class[i,j].set_xlim([0.5,2.5])
		axs_class[i,j].set_xticks([1,2])
		axs_class[i,j].set_xticklabels(['L1', 'L2'])

# ============================================================

fig_class.savefig(args.PATH + 'results/' + args.title + '/plots/num_fibers_' + args.epoch + '_per_class.svg',format='svg')
fig_mean.savefig(args.PATH + 'results/' + args.title + '/plots/num_fibers_' + args.epoch + '_gral.svg',format='svg')
