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
# ======= DISTANCE MATRIX AND ACTIVITY PER CLUSTER (GRAL) ===========
num_classes = 10
num_layers = len(args.layers)
aggl_thr_gral = 0.3 #0.25 # network 03-04:0.3

f = open(args.PATH + 'results/' + args.title + '/num_fibers_' + args.epoch + '_gral.txt', "w")
max_act_plot = {'initial':0.4, 'final':1.5}

for exp in range(args.num_exps):
	filename = args.PATH + 'results/' + args.title + '/outputs_' + args.epoch + '/exp_' + str(exp+1) +'.csv'
	data = pd.read_csv(filename, index_col=0)
	samples = data.iloc[:,:10000]

	# fig_v1, axs_v1 = plt.subplots(1,2)
	fig_v2, axs_v2 = plt.subplots(1,2)

	for ll in range(num_layers):
		samples_layer = samples.loc[samples.index.str.startswith('L'+str(ll+1))]
		mean_layer = data.loc[:,'mean'].loc[data.loc[:,'mean'].index.str.startswith('L'+str(ll+1))]

		# Clustering -------------------------------------------------
		distance = cdist(samples_layer, samples_layer, metric='cityblock')/10000
		distance_real = np.copy(distance)

		np.random.seed(42)
		model = AgglomerativeClustering(distance_threshold=aggl_thr_gral, n_clusters=None, linkage='average',metric='precomputed')
		clusters = model.fit_predict(distance_real)

		# Plot -------------------------------------------------
		# index_order = sorted(range(len(clusters)), key=lambda k: (clusters[k], k))
		# dist = axs_v1[ll].imshow(distance_real[index_order][:, index_order], cmap='viridis', interpolation='nearest')
		# divider = make_axes_locatable(axs_v1[ll])
		# axtop = divider.append_axes("top", size=1.2, pad=0.3, sharex=axs_v1[ll])
		# axtop.plot(mean_layer.reindex(['L'+str(ll+1) + '_' + str(kk) for kk in index_order]).values)
		# axtop.margins(x=0)
		# plt.tight_layout()
		# plt.colorbar(dist, ax=axs_v1[ll], orientation='horizontal', fraction=0.03, pad=0.1)

		mean_layer.sort_values(ascending=True, inplace=True)
		index_order = [int(kk[3:]) for kk in mean_layer.index.tolist()]
		dist = axs_v2[ll].imshow(distance_real[index_order][:, index_order], cmap='viridis', interpolation='nearest')
		divider = make_axes_locatable(axs_v2[ll])
		axtop = divider.append_axes("top", size=1.2, pad=0.3, sharex=axs_v2[ll])
		axtop.plot(mean_layer.values)
		axtop.margins(x=0)
		plt.tight_layout()

		plt.colorbar(dist, ax=axs_v2[ll], orientation='horizontal', fraction=0.03, pad=0.1)
		dist.set_clim(0,1.5)
		print(exp, '-', 'L'+str(ll+1), max(clusters),file=f)

	#fig_v1.savefig(args.PATH + 'results/' + args.title + '/plots/' + args.epoch + '/dist_mtx_exp_' + str(exp+1) + '_clust.svg',format='svg')
	fig_v2.savefig(args.PATH + 'results/' + args.title + '/plots/' + args.epoch + '/dist_mtx_exp_' + str(exp+1) + '_fiber.svg',format='svg')
	#plt.close(fig_v1)
	plt.close(fig_v2)

f.close()
# ===================================================================