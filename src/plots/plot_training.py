'''
Title: Plot training of the networks.
Author: Osvaldo M Velarde
Project: Fibrations in DL Models
'''

# ===========================================================
# ============== MODULES & PATHS ============================
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)

args = parser.parse_args()

# ===========================================================
# ===================== PLOTTING ============================

train_folder  = args.PATH + 'results/' + args.title + '/training/'
plot_folder = args.PATH + 'results/' + args.title + '/plots/training/'

if not os.path.isdir(plot_folder): os.makedirs(plot_folder)

name_files = os.listdir(train_folder)
num_exps = len(name_files)
performance_exp = []

for idx_exp, exp in enumerate(name_files):
	performance_exp.append(np.load(train_folder + exp))

performance_exp = np.stack(performance_exp,axis=0)

mean = np.mean(performance_exp,axis=0)
std  = np.std(performance_exp,axis=0)

fig, axs = plt.subplots(1,1,figsize=(5,5))

axs.plot(mean[:,0], mean[:,2], 'black', label = 'std ' + str(std[-1,2]))
axs.fill_between(mean[:,0], mean[:,2]-std[:,2], mean[:,2]+std[:,2])
axs.set_ylim(0.6,1.0)
axs.legend()

print('Final Performance:',mean[-1][-1],std[-1][-1])
fig.savefig(plot_folder + 'training.svg',format='svg')