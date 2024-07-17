'''
Title: Fibrations of graph
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ===========================================================
# ============== MODULES & PATHS ============================

import argparse
import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)
parser.add_argument('--num-exps', default=1, type=int)
parser.add_argument('--layers', default=[100,100], nargs="+", type=int)
parser.add_argument('--epoch', choices=['initial','final'])

args = parser.parse_args()

# ===================================================================
# ===== DISTANCE MATRIX AND ACTIVITY PER CLUSTER (PER CLASS) ========
num_classes = 10
num_layers = len(args.layers)
aggl_thr_class = 0.04 # network 03-04:0.04

f = open(args.PATH + 'results/' + args.title + '/num_fibers_' + args.epoch + '_per_class.txt', "w")
max_act_plot = {'initial':0.4, 'final':1.5}

for exp in range(args.num_exps):
	filename = args.PATH + 'results/' + args.title + '/outputs_' + args.epoch + '/exp_' + str(exp+1) +'.csv'
	data = pd.read_csv(filename, index_col=0)
	samples = data.iloc[:,:10000]

	for cl in range(num_classes):
		samples_in_class = samples.loc[:,samples.loc['target'] == cl]
		mean_class = data.loc[:,'mean_'+str(cl)]

		#fig_v1, axs_v1 = plt.subplots(1,2)
		fig_v2, axs_v2 = plt.subplots(1,2)

		for ll in range(num_layers):
			samples_layer = samples_in_class.loc[samples_in_class.index.str.startswith('L'+str(ll+1))]
			mean_layer = mean_class.loc[mean_class.index.str.startswith('L'+str(ll+1))]

			# Clustering -------------------------------------------------
			distance_real = cdist(samples_layer, samples_layer, metric='cityblock')/1000
			np.random.seed(42)
			model = AgglomerativeClustering(distance_threshold=aggl_thr_class, n_clusters=None, linkage='average',metric='precomputed')
			clusters = model.fit_predict(distance_real)

			# Plot -------------------------------------------------
			#index_order = sorted(range(len(clusters)), key=lambda k: (clusters[k], k))
			#dist = axs_v1[ll].imshow(distance_real[index_order][:, index_order], cmap='viridis', interpolation='nearest')
			#divider = make_axes_locatable(axs_v1[ll])
			#axtop = divider.append_axes("top", size=1.2, pad=0.3, sharex=axs_v1[ll])
			#axtop.plot(mean_layer.reindex(['L'+str(ll+1) + '_' + str(kk) for kk in index_order]).values)
			#axtop.margins(x=0)
			#plt.tight_layout()
			#cbar = plt.colorbar(dist, ax=axs_v1[ll], orientation='horizontal', fraction=0.03, pad=0.1)
			#dist.set_clim(0,max_act_plot[args.epoch])

			mean_layer.sort_values(ascending=True, inplace=True)
			index_order = [int(kk[3:]) for kk in mean_layer.index.tolist()]
			dist = axs_v2[ll].imshow(distance_real[index_order][:, index_order], cmap='viridis', interpolation='nearest')
			divider = make_axes_locatable(axs_v2[ll])
			axtop = divider.append_axes("top", size=1.2, pad=0.3, sharex=axs_v2[ll])
			axtop.plot(mean_layer.values)
			axtop.margins(x=0)
			plt.tight_layout()
			plt.colorbar(dist, ax=axs_v2[ll], orientation='horizontal', fraction=0.03, pad=0.1)
			dist.set_clim(0,max_act_plot[args.epoch])

			# Saving results
			print(exp, cl, 'L'+str(ll+1), max(clusters),file=f)

		# Saving Plot 
		#fig_v1.savefig(args.PATH + 'results/' + args.title + '/plots/' + args.epoch + '/dist_mtx_exp_' + str(exp+1) + '_class_' + str(cl)+ '_clust.svg',format='svg')
		fig_v2.savefig(args.PATH + 'results/' + args.title + '/plots/' + args.epoch + '/dist_mtx_exp_' + str(exp+1) + '_class_' + str(cl)+ '_fiber.svg',format='svg')
		#plt.close(fig_v1)
		plt.close(fig_v2)

f.close()