'''
Title: Fibrations of graph
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ===================================================================
# ===================== MODULES & PATHS =============================

import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===================================================================
# ====================== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)
parser.add_argument('--num-exps', default=1, type=int)
parser.add_argument('--layers', default=[100,100], nargs="+", type=int)
parser.add_argument('--epoch', choices=['initial','final'])

args = parser.parse_args()

# ===================================================================
# ======= ACTIVITY VS SIZE OF FIBERS IN GRAL CASE ===================

num_classes = 10
num_layers = len(args.layers)

epsilon_l = {'L1':0.1,'L2':0.1}
max_activity = {'initial':1,'final':6} #4 # network 03-04:6
max_prob_activity = {'initial':3,'final':0.4} #4 # network 03-04:6
max_size_fibers = 90 #250 # network 03-04:90
num_points = 1000 #2000 # network 03-04:1000

# Plot ----------------------------------------------
fig_act = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(3, 6)
ax_main=[None,None]
ax_xDist=[None,None]
ax_yDist=[None,None]

for ll in range(num_layers):	
	ax_main[ll] = plt.subplot(gs[1:3, 3*ll:3*ll+2])
	ax_xDist[ll] = plt.subplot(gs[0, 3*ll:3*ll+2],sharex=ax_main[ll])
	ax_yDist[ll] = plt.subplot(gs[1:3, ll*3 + 2],sharey=ax_main[ll])
	    
	ax_main[ll].set(xlabel="Activity")
	ax_main[ll].set(ylabel="Size")
	ax_main[ll].set(xlim=[0,max_activity[args.epoch]])
	ax_main[ll].set(ylim=[0,max_size_fibers])

# ===================================================================

for exp in range(args.num_exps):
	filename = args.PATH + 'results/' + args.title + '/outputs_' + args.epoch + '/exp_' + str(exp+1) +'.csv'
	data = pd.read_csv(filename, index_col=0)

	samples = data.iloc[:,:10000]

	for ll in range(num_layers):
		# Calculation ---------------------------------------
		samples_layer = samples.loc[samples.index.str.startswith('L'+str(ll+1))]

		max_l = samples_layer.max().max()
		histograms = []

		bins = np.arange(0, max_l + epsilon_l['L'+str(ll+1)], epsilon_l['L'+str(ll+1)])

		for col in samples_layer.columns:
			hist, bin_edges = np.histogram(samples_layer[col], bins=bins)
			histograms.append(np.column_stack((bin_edges[:-1],hist)))

		result = np.vstack(histograms)
		random_points = np.random.choice(result.shape[0], size=num_points, replace=False)

		# Plot ----------------------------------------------	    
		ax_main[ll].scatter(result[random_points,0],result[random_points,1],marker='.', color='blue')

		ax_xDist[ll].hist(result[:,0],bins=6,align='mid',alpha=0.3, color='blue', density=True)
		ax_xDist[ll].set(ylabel='count')
		ax_xDist[ll].set_ylim([0,max_prob_activity[args.epoch]])

		ax_yDist[ll].hist(result[:,1],bins=40,orientation='horizontal',align='mid',alpha=0.3, color='blue', density=True)
		ax_yDist[ll].set(xlabel='count')
		ax_yDist[ll].set(xscale='log')

fig_act.savefig(args.PATH + 'results/' + args.title + '/plots/act_vs_size_' + args.epoch +'.svg',format='svg')
plt.close(fig_act)
# ===================================================================