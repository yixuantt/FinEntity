import os
import torch
# Training set, validation set division ratio
dev_split_size = 0.8

# Whether to fine tuning the entire BERT
full_fine_tuning = True

# hyper-parameter
lr = 3e-4
lr_crf = 3e-5
crf_learning_rate = 1e-2 #1e-2 
eps = 1e-6
weight_decay = 0.01
clip_grad = 5
train_batch_size = 16
epoch_num = 20
min_epoch_num = 5

per_gpu_train_batch_size = 16
n_gpu = 1
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
warm_up_ratio = 0.01
gpu = '1'
last_state_dim =256

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
    
    
crf = "crf"
bert = "bert"