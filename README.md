# Structure

./
	/src
	/results
	/cfgfiles

# =====================================================================================

# Bash files
1) evaluation_in_mlps.sh idx num_exps

* Evaluation the "network_idx" - "num_exps" independent times (src/deepnetworks/evaluation_initial.py). This code generates the folder
	/results/network_idx/outputs_initial: 	Files .npy with activation of nodes (evaluation before the training) 

* Activations of the nodes (src/fibrations_in_mlps/activations_to_csv.py). 
This code combine the information of the folder 'results/network_idx/outputs_initial/exp_nn/' in the file 'results/network_idx/outputs_initial/exp_nn.csv':
	COLUMNS = [Empty, 1,2,3,...,N, mean, mean_00, mean_01, mean_02, ..., mean_09]
	ROWS = [L1_00, L1_01, ..., L1_XX, L2_00, L2_01, ..., L2_YY, ..., L3_00, ..., L3_09, target]

# -------------------------------------------------------------------------------------

2) fibrations_in_mlps.sh idx num_exps

* Training the "network_idx" - "num_exps" independent times (src/deepnetworks/training.py). This code generates the folders
	/results/network_idx/outputs_final: 	Files .npy with activation of nodes (evaluation after the training) 
	/results/network_idx/training: 	Files .npy with loss functions vs epochs
	/pretrained_models/network_idx: Files .pt with the weights of the network after the training.

* Activations of the nodes (src/fibrations_in_mlps/activations_to_csv.py). 
This code combine the information of the folder 'results/network_idx/outputs_final/exp_nn/' in the file 'results/network_idx/outputs_final/exp_nn.csv':
	COLUMNS = [Empty, 1,2,3,...,N, mean, mean_00, mean_01, mean_02, ..., mean_09]
	ROWS = [L1_00, L1_01, ..., L1_XX, L2_00, L2_01, ..., L2_YY, ..., L3_00, ..., L3_09, target]

# -------------------------------------------------------------------------------------


# =====================================================================================
# Cfgfiles
idx: (int) index of network

1) struct_network_idx.cfg
	Information about MODEL, LAYERS, BIAS, DEVICE_TYPE, GPU_NUMBER

2) train_network_idx.cfg
	Information about PATHDATA, DATASET, NUM_EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LR_RATE

Examples:
idx 	LAYERS		DATASET		
01		(256 256)	Fashion
02		(256 256)	KMNIST
03		(100 100)	EMNIST
04		(100 100)	MNIST

MODEL='MLP'	
BIAS='false'	
NUM_EPOCHS=30	
TRAIN_BATCH_SIZE=30	
TEST_BATCH_SIZE	LR_RATE=30

3) fibrations_parameters.cfg

# src


